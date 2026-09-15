# HMM Speech Recognition with PyTorch

An isolated-word speech recognition system based on **Mel-Frequency Cepstral Coefficients (MFCCs)** and **left-to-right Hidden Markov Models (HMMs)** implemented with PyTorch.

This repository is a cleaned, standalone version of work completed for the **Speech Recognition (EEEM030)** module at the **University of Surrey** in December 2024. The original coursework investigated an 11-word recognition task using MFCC acoustic features, Baum-Welch re-estimation, and Viterbi decoding. The coursework report recorded **53.64% test accuracy** for the selected 8-state configuration.

## What the project demonstrates

- Speech/audio preprocessing with pre-emphasis and amplitude normalisation
- MFCC extraction from raw audio using Librosa
- Left-to-right Gaussian HMMs implemented with PyTorch tensors
- Log-domain forward and backward algorithms for numerical stability
- Baum-Welch / EM parameter re-estimation
- Viterbi decoding for isolated-word recognition
- Validation and test evaluation with accuracy, recognition error rate, and confusion matrices
- GPU execution when CUDA is available

## My contribution

This was group coursework. My documented contributions were:

- Collaborated on implementation of the HMM training pipeline
- Worked on HMM re-estimation and fine-tuning
- Implemented/contributed to the Viterbi decoding work
- Wrote the abstract, Viterbi, and conclusion sections and helped edit the report

The repository code has been reorganised from the original Colab coursework so it can run as a normal Python project without Google Drive-specific paths or duplicated experimental blocks.

## Repository structure

```text
hmm-speech-recognition-pytorch/
├── hmm_speech_recognition.py   # Training, Baum-Welch, Viterbi, and evaluation
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excludes datasets, models, and generated results
├── data/
│   └── README.md               # Expected dataset layout
├── docs/
│   └── README.md               # Optional coursework-report location
├── models/                     # Generated model checkpoints
└── results/                    # Generated metrics and confusion matrices
```

## Installation

```bash
python -m venv .venv
```

Activate the environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Data format

Place labelled audio files under `data/train` and, optionally, `data/test`.

The final underscore-separated token in the filename is used as the class label:

```text
speaker01_heed.wav  -> heed
speaker03_hood.wav  -> hood
```

The training files should use a consistent sample rate. Test audio is automatically resampled to the training sample rate if needed.

## Run the project

Train the HMMs and evaluate them on a validation split:

```bash
python hmm_speech_recognition.py --train-dir data/train
```

Train and also evaluate on an external test directory:

```bash
python hmm_speech_recognition.py \
  --train-dir data/train \
  --test-dir data/test \
  --n-states 8 \
  --n-mfcc 13 \
  --iterations 15
```

Generated checkpoints are saved under `models/`, while metrics and confusion matrices are written to `results/`.

## Coursework result

The original coursework compared several HMM state configurations. The selected **8-state model** achieved:

- Validation accuracy: **55.10%**
- Test accuracy: **53.64%**
- Recognition error rate: **46.36%**

These numbers are reported from the original coursework experiment. Exact reproduction depends on using the same dataset and split.

## Technical notes

The model uses diagonal Gaussian emission distributions over MFCC features. Forward-backward and Viterbi computations are performed in log space to reduce numerical underflow. The HMM topology permits self-transitions and transitions to the next state, which reflects the temporal progression of speech in an isolated word.

## Academic context

This project is intended to demonstrate speech-processing and sequence-modelling fundamentals. The original dataset is not redistributed here. If you publish the coursework report, first check the course rules and make sure group-member attribution is appropriate.
