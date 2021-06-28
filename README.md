# eegnet_pytorch
EEGNet implementation in PyTorch. It implements only the latest to date version of EEGNet which employs depthwise and separable convolution layers. Also because input signals are 1D and PyTorch allows to use 1D every layer has been implemented as 1D version. Original Army Research Laboratory (ARL) implementation uses 2D versions.

**Original implementation** - https://github.com/vlawhern/arl-eegmodels

**Original paper**: https://iopscience.iop.org/article/10.1088/1741-2552/aace8c

    ---
    EEGNet Parameters:

      nb_classes      : int, number of classes to classify
      Chans           : number of channels in the EEG data
      Samples         : sample frequency (Hz) in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. 
                        ARL recommends to set this parameter to be half of the sampling rate. 
                        For the SMR dataset in particular since the data was high-passed at 4Hz ARL used a kernel length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
