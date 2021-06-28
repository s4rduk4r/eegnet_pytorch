class ConfigHyperparams:
    def __init__(self, epochs_max, learning_rate, batch_size, num_workers, overfit_on_batch):
        self.epochs_max = epochs_max
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.overfit_on_batch = overfit_on_batch
