import os.path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from config.ConfigDataset import ConfigDataset
from config.ConfigHyperparams import ConfigHyperparams
from config.ConfigCheckpoints import ConfigCheckpoints
from config.ConfigEEGNet import ConfigEEGNet

from model.eegnet_pt import EEGNet

class EEGNetTrain():
    def __init__(self, config_dataset: ConfigDataset,
                 config_hyperparams: ConfigHyperparams,
                 config_checkpoints: ConfigCheckpoints,
                 config_eegnet: ConfigEEGNet) -> None:
        # Dataset config
        self.config_dataset = config_dataset
        # Hyperparams
        self.config_hyperparams = config_hyperparams
        # Checkpoints
        self.checkpoint_epoch = None
        self.checkpoint_loss = None
        self.config_checkpoints = config_checkpoints
        # EEGNet hyperparams
        self.config_eegnet = config_eegnet

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.config_eegnet.get_model().parameters(), lr=self.config_hyperparams.learning_rate)

        self.checkpoint_restart()

        self.dataset_prepare(config_hyperparams.batch_size, config_hyperparams.num_workers)

        # Set training device
        if torch.cuda.is_available():
            self.config_eegnet.get_model().cuda()


    def dataset_prepare(self, batch_size:int = 32, num_workers: int = 2) -> None:
        # Data load
        dataset = self.config_dataset.make_dataset()
        # Split into train and validation sets
        length = dataset.__len__()
        len_train = int(length * 0.8)
        len_test = int(length * 0.1)
        len_val = length - len_train - len_test
        train_data, test_data, val_data = random_split(dataset, [len_train, len_test, len_val])
        self.train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=num_workers)
        self.test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=num_workers)
        self.val_loader = DataLoader(dataset=val_data, batch_size=batch_size, num_workers=num_workers)


    def fit(self) -> None:
        print(self.config_eegnet.get_model())

        losses_train = list()

        if self.config_hyperparams.overfit_on_batch:
            batch0 = self.train_loader.__iter__().next()

        if self.config_hyperparams.overfit_on_batch:
            batches_to_train_on = 1
        else:
            batches_to_train_on = self.train_loader.__len__()


        # Restore epoch from checkpoint data
        if not self.config_checkpoints.start_fresh:
            print(f"Continue from epoch: {self.checkpoint_epoch}. Validation loss: {self.checkpoint_loss:.2f}")
            epoch_start = self.checkpoint_epoch + 1
        else:
            epoch_start = 0

        for epoch in range(epoch_start, self.config_hyperparams.epochs_max):

            batch_counter = 1
            for batch in self.train_loader:
                if self.config_hyperparams.overfit_on_batch:
                    batch = batch0

                print(f"\rEpoch: {epoch} training step 1/6: Batch: {batch_counter}/{batches_to_train_on}", end="")
                # Manipulate shape of x, y to become suitable for Conv1d()
                x, y = batch
                x = x.permute(0, 2, 1)
                y = y.max(1)[1] # For CrossEntropyLoss()
                if torch.cuda.is_available():
                    x = x.cuda()

                # 1. Forward propagation
                print(f"\rEpoch: {epoch} training step 2/6: Batch: {batch_counter}/{batches_to_train_on}", end="")
                l = self.config_eegnet.get_model()(x)

                # 2. Compute loss
                print(f"\rEpoch: {epoch} training step 3/6: Batch: {batch_counter}/{batches_to_train_on}", end="")
                if torch.cuda.is_available():
                    y = y.cuda()
                J = self.loss(l, y)

                # 3. Zero the gradients
                print(f"\rEpoch: {epoch} training step 4/6: Batch: {batch_counter}/{batches_to_train_on}", end="")
                self.config_eegnet.get_model().zero_grad()

                # 4. Backward propagation
                print(f"\rEpoch: {epoch} training step 5/6: Batch: {batch_counter}/{batches_to_train_on}", end="")
                J.backward()

                # 5. Step in the optimizer
                print(f"\rEpoch: {epoch} training step 6/6: Batch: {batch_counter}/{batches_to_train_on}", end="")
                self.optimizer.step()

                losses_train.append(J.item())

                if self.config_hyperparams.overfit_on_batch:
                    break

                batch_counter += 1

            loss_validation = self.validate()
            print(f"\tTrain loss: {torch.Tensor(losses_train).mean():.2f}", end="\t")
            print(f"Validation loss: {loss_validation:.2f}")

            # Save model state after an epoch
            if self.config_checkpoints.has_to_save_checkpoint(epoch):
                self.checkpoint_save(epoch, loss_validation)

        return

    def validate(self) -> float:
        losses_val = list()
        for batch in self.val_loader:
            x, y = batch
            x = x.permute(0, 2, 1)
            y = y.max(1)[1]
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # 1. Forward propagation
            with torch.no_grad():
                l = self.config_eegnet.get_model()(x)

            # 2. Compute loss
            J = self.loss(l, y)

            losses_val.append(J.item())

        return torch.Tensor(losses_val).mean()

    def checkpoint_save(self, epoch: int, loss_val: float) -> None:
        checkpoint_data = {
            "epoch" : epoch,
            "loss" : loss_val,
            "model" : self.config_eegnet.get_model().state_dict(),
            "optimizer" : self.optimizer.state_dict()
        }
        checkpoint_file = f"{self.config_eegnet.get_model().__class__.__name__}-epoch-{epoch}.pt.tar"
        torch.save(checkpoint_data, checkpoint_file)

    def checkpoint_load(self, checkpoint_filepath: str) -> tuple:
        # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        checkpoint_data = torch.load(checkpoint_filepath)
        self.config_eegnet.get_model().load_state_dict(checkpoint_data["model"])
        self.optimizer.load_state_dict(checkpoint_data["optimizer"])
        epoch, loss = checkpoint_data["epoch"], checkpoint_data["loss"]
        return epoch, loss

    def checkpoint_restart(self):
        if not self.config_checkpoints.start_fresh:
            print("Loading checkpoint...", end="")
            # Find last checkpoint file
            checkpoint_file = None
            for file in os.scandir():
                if file.is_file() and file.name.endswith(".pt.tar"):
                    if checkpoint_file is None:
                        checkpoint_file = file
                    elif file.stat().st_mtime > checkpoint_file.stat().st_mtime:
                        checkpoint_file = file

            if checkpoint_file is None:
                self.config_checkpoints.start_fresh = True
                print("no checkpoints found")
                return

            self.checkpoint_epoch, self.checkpoint_loss = self.checkpoint_load(checkpoint_file.path)
            self.config_eegnet.get_model().train()
            print("ok")

# Add commandline arguments
def parser_add_cmdline_arguments():
    parser = argparse.ArgumentParser(description="Train EEGNet for the 1s duration EEG sample")
    # Datasets
    datasets = parser.add_argument_group("Datasets", "Select and manipulate datasets")
    datasets.add_argument("dataset", choices=["5f", "cla", "halt", "freeform", "nomt"], default="cla",
                          help="BCI EEG dataset to work with. More info: https://doi.org/10.6084/m9.figshare.c.3917698.v1\n"
                               "5f - 5F set. 5 finger gestures\n"
                               "cla - CLA set. 3 gestures\n"
                               "halt - HaLT set. 5 gestures\n"
                               "freeform - FREEFORM set\n"
                               "nomt - NoMT set")
    datasets.add_argument("--no_download", action="store_false",
                          help="Don't download dataset files. Default: False")
    datasets_options = datasets.add_mutually_exclusive_group()
    datasets_options.add_argument("--dataset_merge", action="store_false",
                                  help="Merge data from all subjects into a single dataset. Default: True")
    datasets_options.add_argument("--dataset_subjects", nargs='+', type=int, metavar="n",
                                  help="Merge data from specific test subjects into a single dataset. "
                                       "Takes zero-based single index or a zero-based list of indices")

    # Checkpoints
    checkpoints = parser.add_argument_group("Checkpoints", "Manipulate model chekpoints behaviour")
    checkpoints.add_argument("--start_fresh", action="store_true",
                             help="Ignores checkpoints. Default: False")
    checkpoints2 = checkpoints.add_mutually_exclusive_group()
    checkpoints2.add_argument("--no_checkpoints", action="store_true",
                             help="Make no checkpoints. Default: False")
    checkpoints2.add_argument("--checkpoint_every_epoch", type=int, metavar="val", default=1,
                             help="Save model to checkpoint every 'val' epochs. Default: 1")
    # Hyperparams
    hyperparams = parser.add_argument_group("Hyperparams", "Set neuralnet training hyperparams")
    hyperparams.add_argument("--learning_rate", type=float, metavar="val", default=1e-2,
                             help="Set learning rate. Default: 1e-2", )
    hyperparams.add_argument("--epochs_max", type=int, metavar="val", default=10,
                             help="Set maximum number of training epochs. Default: 10")
    hyperparams.add_argument("--batch_size", type=int, metavar="val", default=32,
                             help="Set training batch size. Default: 32")
    hyperparams.add_argument("--num_workers", type=int, metavar="val", default=2,
                             help="Set number of CPU worker threads to prepare dataset batches. Default: 2")
    hyperparams.add_argument("--overfit_on_batch", action="store_true",
                             help="Test if model overfits on a single batch of training data")
    # EEGNet hyperparams
    eegnet_hyperparams = parser.add_argument_group("EEGNet hyperparams")
    eegnet_hyperparams.add_argument("--eegnet_nb_classes", type=int, metavar="Classes", default=4,
                                    help="Number of classification categories. Default=4")
    eegnet_hyperparams.add_argument("--eegnet_kernel_length", type=int, metavar="Krnl_Length", default=63,
                                    help="Length of temporal convolution in first layer. We found "
                                         "that setting this to be half the sampling rate worked "
                                         "well in practice. For the SMR dataset in particular "
                                         "since the data was high-passed at 4Hz we used a kernel "
                                         "length of 31. "
                                         "Must be odd number!",
                                    )
    eegnet_hyperparams.add_argument("--eegnet_channels", type=int, metavar="Chnls", default=64,
                                    help="Number of channels in the EEG data. Default: 64")
    eegnet_hyperparams.add_argument("--eegnet_samples", type=int, metavar="Freq", default=128,
                                    help="Sample frequency (Hz) in the EEG data. Default: 128Hz", )
    eegnet_hyperparams.add_argument("--eegnet_f1", type=int, metavar="F1", default=8,
                                    help="Number of temporal filters. Default: 8")
    eegnet_hyperparams.add_argument("--eegnet_d", type=int, metavar="D", default=2,
                                    help="Number of spatial filters to learn within each temporal convolution. Default: 2")
    eegnet_hyperparams.add_argument("--eegnet_dropout_rate", type=float, metavar="dr", default=0.5,
                                    help="Dropout rate in Block 1")
    return parser.parse_args()

'''
# Entry point
'''
# Commandline arguments
args = parser_add_cmdline_arguments()

# Dataset config
if args.dataset_merge:
    config_dataset = ConfigDataset(dataset_type=args.dataset, download=args.no_download, dataset_subjects=None)
else:
    config_dataset = ConfigDataset(dataset_type=args.dataset, download=args.no_download, dataset_subjects=args.dataset_subjects)

config_hyperparams = ConfigHyperparams(epochs_max=args.epochs_max, learning_rate=args.learning_rate,
                                       batch_size=args.batch_size, num_workers=args.num_workers,
                                       overfit_on_batch=args.overfit_on_batch)
config_checkpoints = ConfigCheckpoints(checkpoint_every_epoch=args.checkpoint_every_epoch,
                                       start_fresh=args.start_fresh, no_checkpoints=args.no_checkpoints)
config_eegnet = ConfigEEGNet(nb_classes=args.eegnet_nb_classes, channels=args.eegnet_channels, samples=args.eegnet_samples,
                             kernel_length=args.eegnet_kernel_length, f1=args.eegnet_f1, d=args.eegnet_d,
                             dropout_rate=args.eegnet_dropout_rate)

trainer = EEGNetTrain(config_dataset, config_hyperparams, config_checkpoints, config_eegnet)
trainer.fit()


