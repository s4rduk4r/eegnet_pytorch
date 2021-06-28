import os
import sys
import urllib3
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

class Base_EEG_BCI_Dataset(Dataset):
    def __init__(self, download: bool = False, merge_list: list = None,
                 download_dir: str = None,
                 download_uri: list = None,
                 classes: list = None,
                 samples_frequency_in_Herz: int = 200):
        self.download_path = os.path.join(os.getcwd(), "data", "download", download_dir)
        self.download_uri = download_uri
        self.classes = classes
        self.samples_frequency_in_Herz = samples_frequency_in_Herz
        self.data: torch.Tensor = None
        self.labels: torch.Tensor = None
        self.one_hot_labels: torch.Tensor = None

        if download:
            self.download_datasets()

        if merge_list is None:
            self.merge_all_datasets()
        else:
            self.merge_datasets(merge_list)

    # Download dataset files
    def download_datasets(self):
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

        root_dir = os.getcwd()
        os.chdir(self.download_path)

        print(f"Downloading dataset {self.__class__.__name__}")

        http = urllib3.PoolManager()
        for uri in self.download_uri:
            print(f"Dataset: {self.download_uri.index(uri) + 1} of {self.download_uri.__len__()} ...", end="")
            dataset_file = uri.split(sep=os.path.sep)[-1]
            if os.path.exists(dataset_file):
                print("skip")
                continue

            req = http.request("GET", uri)
            with open(dataset_file, "wb") as f:
                f.write(req.data)
                print("ok", end="")

        print("")
        os.chdir(root_dir)

    # Merge all the dataset files into a one huge dataset
    def merge_all_datasets(self):
        print("Joining all datasets into one...")
        # Get dataset files
        root_dir = os.getcwd()
        os.chdir(self.download_path)
        dataset_files = os.listdir()
        for dataset_file in dataset_files:
            data_matlab = loadmat(dataset_file)
            dataset_data = torch.Tensor(data_matlab["o"]["data"].item())
            dataset_labels = torch.Tensor(data_matlab["o"]["marker"].item())
            del data_matlab
            if self.data is None:
                self.data = dataset_data
            else:
                self.data = torch.cat((self.data, dataset_data), 0)

            del dataset_data

            if self.labels is None:
                self.labels = dataset_labels
            else:
                self.labels = torch.cat((self.labels, dataset_labels), 0)

            del dataset_labels

            print(f"Processed: {dataset_files.index(dataset_file) + 1} of {dataset_files.__len__()}")

        os.chdir(root_dir)

    def merge_datasets(self, indices: list):
        print("Joining all datasets into one...")
        # Get dataset files
        root_dir = os.getcwd()
        os.chdir(self.download_path)
        dataset_files = os.listdir()
        for index in indices:
            data_matlab = loadmat(dataset_files[index])
            dataset_data = torch.Tensor(data_matlab["o"]["data"].item())
            dataset_labels = torch.Tensor(data_matlab["o"]["marker"].item())
            del data_matlab
            if self.data is None:
                self.data = dataset_data
            else:
                self.data = torch.cat((self.data, dataset_data), 0)

            del dataset_data

            if self.labels is None:
                self.labels = dataset_labels
            else:
                self.labels = torch.cat((self.labels, dataset_labels), 0)

            del dataset_labels

            print(f"Processed: {index + 1} of {indices.__len__()}")

        os.chdir(root_dir)

    '''
    Get size in bytes, KiB, MiB of dataset without labels
    Returns tuple: (bytes, KiB, MiB)
    '''
    def get_sizeof_data_in_bytes(self) -> tuple:
        size_in_bytes = torch.flatten(self.data).shape[0] * self.data.element_size()
        kb = 1 / 1024
        mb = 1024 ** -2
        return size_in_bytes, size_in_bytes * kb, size_in_bytes * mb

    '''
    Get size in bytes, KiB, MiB of labels
    Returns tuple: (bytes, KiB, MiB)
    '''
    def get_sizeof_labels_in_bytes(self) -> tuple:
        size_in_bytes = torch.flatten(self.labels).shape[0] * self.labels.element_size()
        kb = 1 / 1024
        mb = 1024 ** -2
        return size_in_bytes, size_in_bytes * kb, size_in_bytes * mb

    '''
    Get size in bytes, KiB, MiB of whole dataset with the labels
    Returns tuple: (bytes, KiB, MiB)
    '''
    def get_sizeof_dataset_in_bytes(self) -> tuple:
        size_in_bytes = self.get_sizeof_data_in_bytes()[0] + self.get_sizeof_labels_in_bytes()[0]
        kb = 1 / 1024
        mb = 1024 ** -2
        return size_in_bytes, size_in_bytes * kb, size_in_bytes * mb

    def __str__(self):
        return f"data.shape = {self.data.shape}\n" \
               f"labels.shape = {self.labels.shape}\n" \
               f"Size of dataset: {self.get_sizeof_data_in_bytes()[2]:.2f}MiB\n" \
               f"Size of labels: {self.get_sizeof_labels_in_bytes()[2]:.2f}MiB\n" \
               f"Total size: {self.get_sizeof_dataset_in_bytes()[2]:.2f}MiB\n" \
               f"Size of dataset object: {sys.getsizeof(self)}"


    # One-hot encode labels
    def one_hot_encode(self):
        self.one_hot_labels = torch.zeros((self.__len__(), self.classes.__len__()), dtype=torch.int64)
        classes_keys = list(self.classes.keys())
        for i in range(self.__len__()):
            x, y = self.aux_get_1sec(i)
            y = int(y.max().item())
            if y == 90 or y == 99:
                y = classes_keys.index(classes_keys[-1])
            elif y == 91:
                y = classes_keys.index(classes_keys[-2])
            elif y == 92:
                y = classes_keys.index(classes_keys[-3])
            self.one_hot_labels[i, y] = 1

    def aux_get_1sec(self, index:slice):
        index_start = index * self.samples_frequency_in_Herz
        index_end = index_start + self.samples_frequency_in_Herz
        data_item = self.data[index_start:index_end]
        label_item = self.labels[index_start:index_end]
        return data_item, label_item

    # Implement torch.utils.data.Dataset interface
    def __getitem__(self, index: slice):
        index_start = index * self.samples_frequency_in_Herz
        index_end = index_start + self.samples_frequency_in_Herz
        data_item = self.data[index_start:index_end]
        return data_item, self.one_hot_labels[index]

    def __len__(self):
        if self.data is None or self.labels is None:
            return 0
        else:
            # Last second is trimmed because dataset can hold data less than a second
            return self.data.__len__() // self.samples_frequency_in_Herz - 1
