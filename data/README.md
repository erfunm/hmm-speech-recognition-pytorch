# Data

The original coursework audio is not included in this repository.

Expected layout:

```text
data/
├── train/
│   ├── speaker01_heed.wav
│   ├── speaker02_heed.wav
│   ├── speaker01_hid.wav
│   └── ...
└── test/
    ├── speaker11_heed.wav
    ├── speaker12_hid.wav
    └── ...
```

The script uses the final underscore-separated token in each filename as the word label. For example, `speaker01_heed.wav` is assigned the label `heed`.

Do not publish course-provided audio unless its licence or the university explicitly allows redistribution.
