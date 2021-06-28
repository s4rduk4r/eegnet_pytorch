from model.eegnet_pt import EEGNet


class ConfigEEGNet:
    def __init__(self, nb_classes:int, channels: int, samples: int,
                 kernel_length: int, f1: int, d:int,
                 dropout_rate: float):
        self.nb_classes = nb_classes
        self.channels = channels
        self.samples = samples
        self.kernel_length = kernel_length
        self.f1 = f1
        self.d = d
        self.dropout_rate = dropout_rate
        self.model = None

    def get_model(self):
        if self.model is None:
            self.model = EEGNet(self.nb_classes, self.channels, self.samples, self.dropout_rate, self.kernel_length, self.f1, self.d)
        return self.model
