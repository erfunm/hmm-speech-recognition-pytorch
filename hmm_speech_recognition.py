import argparse
import json
import os
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DEFAULT_N_STATES = 8
DEFAULT_N_MFCC = 13
DEFAULT_FRAME_MS = 30
DEFAULT_HOP_MS = 10
DEFAULT_ITERATIONS = 15
DEFAULT_VALIDATION_SIZE = 0.30
DEFAULT_RANDOM_SEED = 42


def load_audio_directory(data_dir):
    """Load .wav/.mp3 files and use the final filename token as the label."""
    data_dir = Path(data_dir)
    audio_data = []
    labels = []
    sample_rate = None

    files = sorted(list(data_dir.glob("*.wav")) + list(data_dir.glob("*.mp3")))
    if not files:
        raise ValueError(f"No .wav or .mp3 files found in {data_dir}")

    for path in files:
        audio, file_sr = sf.read(path)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        if sample_rate is None:
            sample_rate = file_sr
        elif file_sr != sample_rate:
            raise ValueError(
                f"Inconsistent sample rates: expected {sample_rate} Hz, "
                f"but {path.name} is {file_sr} Hz."
            )

        label = path.stem.split("_")[-1]
        audio_data.append(audio)
        labels.append(label)

    return audio_data, labels, sample_rate


def extract_mfcc(audio, sample_rate, n_mfcc=13, frame_ms=30, hop_ms=10):
    """Apply pre-emphasis, amplitude normalisation, and MFCC extraction."""
    if len(audio) < 2:
        raise ValueError("Audio sample is too short for feature extraction.")

    pre_emphasis = 0.97
    emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    peak = np.max(np.abs(emphasized))
    if peak > 0:
        emphasized = emphasized / peak

    n_fft = max(2, int(sample_rate * frame_ms / 1000))
    hop_length = max(1, int(sample_rate * hop_ms / 1000))

    features = librosa.feature.mfcc(
        y=emphasized,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hamming",
    ).T

    return np.asarray(features, dtype=np.float32)


def extract_dataset_features(audio_data, sample_rate, n_mfcc, frame_ms, hop_ms):
    features = []
    for audio in audio_data:
        features.append(extract_mfcc(audio, sample_rate, n_mfcc, frame_ms, hop_ms))
    return features


def pad_sequences(sequences, device):
    max_length = max(sequence.shape[0] for sequence in sequences)
    n_features = sequences[0].shape[1]

    data = torch.zeros(len(sequences), max_length, n_features, device=device)
    mask = torch.zeros(len(sequences), max_length, dtype=torch.bool, device=device)

    for i, sequence in enumerate(sequences):
        length = sequence.shape[0]
        data[i, :length] = torch.tensor(sequence, dtype=torch.float32, device=device)
        mask[i, :length] = True

    return data, mask


class GaussianHMM(nn.Module):
    """Left-to-right HMM with diagonal Gaussian emission distributions."""

    def __init__(self, n_states, n_features, device=None):
        super().__init__()

        if n_states < 2:
            raise ValueError("n_states must be at least 2.")

        self.n_states = n_states
        self.n_features = n_features
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.register_buffer("initial_log_probs", torch.full((n_states,), float("-inf")))
        self.register_buffer("transition_log_probs", torch.full((n_states, n_states), float("-inf")))
        self.register_buffer("transition_mask", torch.zeros((n_states, n_states), dtype=torch.bool))
        self.register_buffer("means", torch.zeros((n_states, n_features)))
        self.register_buffer("variances", torch.ones((n_states, n_features)))

        self.to(self.device)
        self._initialize_topology()

    def _initialize_topology(self):
        self.initial_log_probs.fill_(float("-inf"))
        self.initial_log_probs[0] = 0.0

        self.transition_mask.zero_()
        for state in range(self.n_states):
            self.transition_mask[state, state] = True
            if state + 1 < self.n_states:
                self.transition_mask[state, state + 1] = True

    def initialize_parameters(self, sequences, masks):
        valid_frames = sequences[masks]
        global_mean = valid_frames.mean(dim=0)
        global_var = valid_frames.var(dim=0, unbiased=False).clamp(min=1e-4)

        self.means.copy_(global_mean.repeat(self.n_states, 1))
        self.variances.copy_(global_var.repeat(self.n_states, 1))

        lengths = masks.sum(dim=1).float()
        expected_duration = max(float(lengths.mean().item()) / self.n_states, 1.5)
        self_prob = float(np.exp(-1.0 / max(expected_duration - 1.0, 1.0)))
        self_prob = min(max(self_prob, 0.50), 0.98)
        next_prob = 1.0 - self_prob

        self.transition_log_probs.fill_(float("-inf"))
        for state in range(self.n_states):
            if state == self.n_states - 1:
                self.transition_log_probs[state, state] = 0.0
            else:
                self.transition_log_probs[state, state] = np.log(self_prob)
                self.transition_log_probs[state, state + 1] = np.log(next_prob)

    def emission_log_prob(self, sequence):
        sequence = sequence[:, None, :]
        means = self.means[None, :, :]
        variances = self.variances[None, :, :].clamp(min=1e-6)

        log_det = torch.sum(torch.log(variances), dim=2)
        mahalanobis = torch.sum((sequence - means) ** 2 / variances, dim=2)
        constant = self.n_features * np.log(2 * np.pi)

        return -0.5 * (constant + log_det + mahalanobis)

    def forward(self, sequence):
        emissions = self.emission_log_prob(sequence)
        length = sequence.shape[0]
        alpha = torch.full((length, self.n_states), float("-inf"), device=self.device)

        alpha[0] = self.initial_log_probs + emissions[0]

        for t in range(1, length):
            scores = alpha[t - 1][:, None] + self.transition_log_probs
            alpha[t] = torch.logsumexp(scores, dim=0) + emissions[t]

        log_likelihood = torch.logsumexp(alpha[-1], dim=0)
        return alpha, log_likelihood

    def backward(self, sequence):
        emissions = self.emission_log_prob(sequence)
        length = sequence.shape[0]
        beta = torch.zeros((length, self.n_states), device=self.device)

        for t in range(length - 2, -1, -1):
            scores = (
                self.transition_log_probs
                + emissions[t + 1][None, :]
                + beta[t + 1][None, :]
            )
            beta[t] = torch.logsumexp(scores, dim=1)

        return beta

    def baum_welch_step(self, sequences, masks):
        gamma_counts = torch.zeros(self.n_states, device=self.device)
        gamma_x = torch.zeros(self.n_states, self.n_features, device=self.device)
        gamma_x2 = torch.zeros(self.n_states, self.n_features, device=self.device)
        transition_counts = torch.zeros(self.n_states, self.n_states, device=self.device)

        for padded_sequence, sequence_mask in zip(sequences, masks):
            length = int(sequence_mask.sum().item())
            sequence = padded_sequence[:length]

            alpha, log_likelihood = self.forward(sequence)
            beta = self.backward(sequence)
            emissions = self.emission_log_prob(sequence)

            log_gamma = alpha + beta - log_likelihood
            gamma = torch.exp(log_gamma)

            gamma_counts += gamma.sum(dim=0)
            gamma_x += torch.sum(gamma[:, :, None] * sequence[:, None, :], dim=0)
            gamma_x2 += torch.sum(gamma[:, :, None] * sequence[:, None, :] ** 2, dim=0)

            for t in range(length - 1):
                log_xi = (
                    alpha[t][:, None]
                    + self.transition_log_probs
                    + emissions[t + 1][None, :]
                    + beta[t + 1][None, :]
                    - log_likelihood
                )
                transition_counts += torch.exp(log_xi)

        safe_counts = gamma_counts.clamp(min=1e-8)
        new_means = gamma_x / safe_counts[:, None]
        new_variances = gamma_x2 / safe_counts[:, None] - new_means ** 2
        new_variances = new_variances.clamp(min=1e-4)

        self.means.copy_(new_means)
        self.variances.copy_(new_variances)

        transition_counts = transition_counts * self.transition_mask
        for state in range(self.n_states):
            row = transition_counts[state]
            total = row.sum()
            if total > 0:
                probs = row / total
                self.transition_log_probs[state].fill_(float("-inf"))
                allowed = self.transition_mask[state]
                self.transition_log_probs[state, allowed] = torch.log(probs[allowed].clamp(min=1e-10))

    def average_log_likelihood(self, sequences, masks):
        total = 0.0
        for padded_sequence, sequence_mask in zip(sequences, masks):
            length = int(sequence_mask.sum().item())
            _, log_likelihood = self.forward(padded_sequence[:length])
            total += float(log_likelihood.item())
        return total / len(sequences)

    def fit(self, feature_sequences, n_iter=15):
        sequences, masks = pad_sequences(feature_sequences, self.device)
        self.initialize_parameters(sequences, masks)

        progress = tqdm(range(n_iter), desc="Baum-Welch")
        for iteration in progress:
            self.baum_welch_step(sequences, masks)
            score = self.average_log_likelihood(sequences, masks)
            progress.set_postfix(iteration=iteration + 1, avg_log_likelihood=f"{score:.2f}")

    def viterbi_decode(self, sequence):
        sequence = torch.tensor(sequence, dtype=torch.float32, device=self.device)
        emissions = self.emission_log_prob(sequence)
        length = sequence.shape[0]

        delta = torch.full((length, self.n_states), float("-inf"), device=self.device)
        backpointer = torch.zeros((length, self.n_states), dtype=torch.long, device=self.device)

        delta[0] = self.initial_log_probs + emissions[0]

        for t in range(1, length):
            scores = delta[t - 1][:, None] + self.transition_log_probs
            delta[t], backpointer[t] = torch.max(scores, dim=0)
            delta[t] += emissions[t]

        final_state = torch.argmax(delta[-1])
        best_score = float(delta[-1, final_state].item())

        path = torch.zeros(length, dtype=torch.long, device=self.device)
        path[-1] = final_state
        for t in range(length - 2, -1, -1):
            path[t] = backpointer[t + 1, path[t + 1]]

        return best_score, path.cpu().numpy()


def train_word_models(features, labels, n_states, n_iter, device):
    models = {}

    for label in sorted(set(labels)):
        word_features = [feature for feature, item_label in zip(features, labels) if item_label == label]
        if not word_features:
            continue

        print(f"\nTraining HMM for '{label}' ({len(word_features)} samples)")
        model = GaussianHMM(n_states=n_states, n_features=word_features[0].shape[1], device=device)
        model.fit(word_features, n_iter=n_iter)
        models[label] = model

    return models


def predict_word(models, feature_sequence):
    best_label = None
    best_score = float("-inf")

    for label, model in models.items():
        score, _ = model.viterbi_decode(feature_sequence)
        if score > best_score:
            best_score = score
            best_label = label

    return best_label, best_score


def evaluate(models, features, labels):
    predictions = []
    for feature_sequence in tqdm(features, desc="Evaluating"):
        predicted_label, _ = predict_word(models, feature_sequence)
        predictions.append(predicted_label)

    accuracy = accuracy_score(labels, predictions)
    ordered_labels = sorted(set(labels) | set(predictions))
    matrix = confusion_matrix(labels, predictions, labels=ordered_labels)

    return accuracy, matrix, ordered_labels, predictions


def save_confusion_matrix(matrix, labels, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_models(models, model_dir, config):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    for label, model in models.items():
        torch.save(model.state_dict(), model_dir / f"{label}_hmm.pt")

    with open(model_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and evaluate left-to-right Gaussian HMMs for isolated-word speech recognition."
    )
    parser.add_argument("--train-dir", required=True, help="Directory containing labelled training audio files.")
    parser.add_argument("--test-dir", help="Optional directory containing labelled test audio files.")
    parser.add_argument("--model-dir", default="models", help="Directory for saved HMM state dictionaries.")
    parser.add_argument("--results-dir", default="results", help="Directory for confusion matrices and metrics.")
    parser.add_argument("--n-states", type=int, default=DEFAULT_N_STATES)
    parser.add_argument("--n-mfcc", type=int, default=DEFAULT_N_MFCC)
    parser.add_argument("--frame-ms", type=int, default=DEFAULT_FRAME_MS)
    parser.add_argument("--hop-ms", type=int, default=DEFAULT_HOP_MS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser


def main():
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading training audio...")
    audio, labels, sample_rate = load_audio_directory(args.train_dir)

    train_audio, val_audio, train_labels, val_labels = train_test_split(
        audio,
        labels,
        test_size=args.validation_size,
        stratify=labels,
        random_state=args.seed,
    )

    print("Extracting MFCC features...")
    train_features = extract_dataset_features(
        train_audio, sample_rate, args.n_mfcc, args.frame_ms, args.hop_ms
    )
    val_features = extract_dataset_features(
        val_audio, sample_rate, args.n_mfcc, args.frame_ms, args.hop_ms
    )

    models = train_word_models(
        train_features,
        train_labels,
        n_states=args.n_states,
        n_iter=args.iterations,
        device=device,
    )

    print("\nValidation results")
    val_accuracy, val_matrix, val_order, _ = evaluate(models, val_features, val_labels)
    print(f"Validation accuracy: {val_accuracy:.2%}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(
        val_matrix,
        val_order,
        results_dir / "validation_confusion_matrix.png",
        "Validation Confusion Matrix",
    )

    metrics = {"validation_accuracy": val_accuracy}

    if args.test_dir:
        print("\nLoading external test audio...")
        test_audio, test_labels, test_sample_rate = load_audio_directory(args.test_dir)

        if test_sample_rate != sample_rate:
            print(f"Resampling test audio from {test_sample_rate} Hz to {sample_rate} Hz")
            test_audio = [
                librosa.resample(item, orig_sr=test_sample_rate, target_sr=sample_rate)
                for item in test_audio
            ]

        test_features = extract_dataset_features(
            test_audio, sample_rate, args.n_mfcc, args.frame_ms, args.hop_ms
        )

        print("Test results")
        test_accuracy, test_matrix, test_order, _ = evaluate(models, test_features, test_labels)
        print(f"Test accuracy: {test_accuracy:.2%}")
        print(f"Recognition error rate: {(1.0 - test_accuracy):.2%}")

        save_confusion_matrix(
            test_matrix,
            test_order,
            results_dir / "test_confusion_matrix.png",
            "Test Confusion Matrix",
        )
        metrics["test_accuracy"] = test_accuracy
        metrics["test_error_rate"] = 1.0 - test_accuracy

    config = {
        "n_states": args.n_states,
        "n_mfcc": args.n_mfcc,
        "frame_ms": args.frame_ms,
        "hop_ms": args.hop_ms,
        "iterations": args.iterations,
        "validation_size": args.validation_size,
        "sample_rate": sample_rate,
        "labels": sorted(set(labels)),
    }
    save_models(models, args.model_dir, config)

    with open(results_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\nSaved models to: {args.model_dir}")
    print(f"Saved results to: {args.results_dir}")


if __name__ == "__main__":
    main()
