# eegnet_pytorch
EEGNet implementation in PyTorch. It implements only the latest to date version of EEGNet which employs depthwise and separable convolution layers. Also because input signals are 1D and PyTorch allows to use 1D every layer has been implemented as 1D version. Original Army Research Laboratory (ARL) implementation uses 2D versions.

**Original implementation** - https://github.com/vlawhern/arl-eegmodels

**Original paper**: https://iopscience.iop.org/article/10.1088/1741-2552/aace8c


**Directory structure:**
```
./
├── config
│   ├── ConfigCheckpoints.py
│   ├── ConfigDataset.py
│   ├── ConfigEEGNet.py
│   ├── ConfigHyperparams.py
│   └── __init__.py
├── data
│   ├── Base_EEG_BCI_Dataset.py
│   ├── EEG_BCI_doi10_6084_m9_figshare_c_3917698_v1.py
│   └── __init__.py
├── model
│   ├── eegnet_pt.py
│   ├── __init__.py
│   └── SeparableConv.py
└── run_experiment.py
```

./run_experiment.py - holds EEGNetTrain() class which makes all preparations before the training and fits the model. Supports checkpoints

./model - holds EEGNet implementation and Separable and Depthwise convolution layers

./data - holds wrappers for EEG BCI mental imagery datasets published here - https://doi.org/10.6084/m9.figshare.c.3917698.v1

./config - holds different commandline argument configs for EEGNetTrain() class

