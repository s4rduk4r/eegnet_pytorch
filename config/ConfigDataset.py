import data.EEG_BCI_doi10_6084_m9_figshare_c_3917698_v1


class ConfigDataset:
    def __init__(self, dataset_type: str, download: bool, dataset_subjects: list = None) -> None:
        self.download = download
        self.dataset_type = dataset_type
        self.dataset_subjects = dataset_subjects # None = merge data for ALL subjects
        # Dataset type and download
        if self.dataset_type == "5f":
            self.dataset =\
                data.EEG_BCI_doi10_6084_m9_figshare_c_3917698_v1.EEG_BCI_5F_doi10_6084_m9_figshare_c_3917698_v1
        elif self.dataset_type == "cla":
            self.dataset = \
                data.EEG_BCI_CLA_doi10_6084_m9_figshare_c_3917698_v1
        elif self.dataset_type == "halt":
            self.dataset = \
                data.EEG_BCI_HaLT_doi10_6084_m9_figshare_c_3917698_v1
        elif self.dataset_type == "freeform":
            self.dataset = \
                data.EEG_BCI_FREEFORM_doi10_6084_m9_figshare_c_3917698_v1
        elif self.dataset_type == "nomt":
            self.dataset = \
                data.EEG_BCI_NoMT_doi10_6084_m9_figshare_c_3917698_v1
        else:
            raise ValueError("Incorrect dataset type")

    def make_dataset(self) -> data.EEG_BCI_doi10_6084_m9_figshare_c_3917698_v1.Base_EEG_BCI_Dataset:
        return  self.dataset(self.download, self.dataset_subjects)
