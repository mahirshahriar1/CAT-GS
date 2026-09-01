# Data directory

Place datasets here (or point the paths in `config.py` elsewhere):

```
data/
├── CREMAD/
│   ├── train.csv
│   ├── test.csv
│   ├── AudioWAV/                # .wav clips
│   └── Image-01-FPS/            # frames extracted at 1 fps, one folder per clip
├── AVMNIST/
│   ├── audio/{train_data.npy, test_data.npy}
│   ├── image/{train_data.npy, test_data.npy}
│   ├── train_labels.npy
│   └── test_labels.npy
├── VGGSound/
│   ├── vggsound.csv
│   ├── video/frames/<split>/Image-01-FPS/<clip_id>/
│   └── audio/
├── AVE_Dataset/
│   ├── trainSet.txt / testSet.txt / valSet.txt
│   ├── Audio-1004-SE/           # precomputed spectrogram .pkl files
│   └── Image-01-FPS-SE/         # frames, one folder per clip
├── Mnist/
│   ├── mnist/                   # downloaded automatically by torchvision
│   └── colored_mnist/mnist_10color_jitter_var_0.030.npy
├── URFUNNY/                     # produced by: python -m dataset.URFunnyDataset
│   ├── audio_features_{train,dev,test}.pkl
│   ├── visual_features_{train,dev,test}.pkl
│   ├── text_features_{train,dev,test}.pkl
│   └── labels_{train,dev,test}.pkl
└── MOSI/
    └── mosi_raw.pkl             # MultiBench-format CMU-MOSI pickle
```

- UR-FUNNY V2 raw pickles: https://github.com/ROC-HCI/UR-FUNNY
- CMU-MOSI processed pickle: https://github.com/pliang279/MultiBench

