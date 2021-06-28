'''
    Description:
        This is an electroencephalographic brain-computer interface (EEG BCI) mental imagery dataset
        collected during development of a slow cortical potentials motor imagery EEG BCI.
        The dataset contains 60 hours of EEG BCI recordings spread across 75 experiments and 13 participants,
        featuring 60,000 mental imagery examples in 4 different BCI interaction paradigms
        with up to 6 EEG BCI interaction states. 4.8 hours of EEG recordings and 4600 mental imagery examples
        are available per participant, on average, with good longitudinal and lateral dataset span
        as well as significant EEG BCI interaction complexity.

    Reference:
        1.  Kaya, Murat; Binli, Mustafa Kemal; Ozbay, Erkan; Yanar, Hilmi; Mishchenko, Yuriy (2018):
            A large electroencephalographic motor imagery dataset for electroencephalographic brain computer interfaces.
            figshare. Collection. https://doi.org/10.6084/m9.figshare.c.3917698.v1
        2.  Kaya, M., Binli, M., Ozbay, E. et al.
            A large electroencephalographic motor imagery dataset for electroencephalographic brain computer interfaces.
            Sci Data 5, 180211 (2018). https://doi.org/10.1038/sdata.2018.211
'''
from data.Base_EEG_BCI_Dataset import Base_EEG_BCI_Dataset


class EEG_BCI_5F_doi10_6084_m9_figshare_c_3917698_v1(Base_EEG_BCI_Dataset):
    def __init__(self, download: bool = False, merge_list: list = None):
        classes = {0: "blank",
                   1: "thumb MI", 2: "index finger MI", 3: "middle finger MI",
                   4: "ring finger MI", 5: "pinkie finger MI",
                   # 90: "initial relaxation period",
                   91: "inter-session rest break period", 92: "experiment end", 99: "initial relaxation period"}
        download_uri = [
                "https://ndownloader.figshare.com/files/12400397",
                "https://ndownloader.figshare.com/files/12400394",
                "https://ndownloader.figshare.com/files/12400391",
                "https://ndownloader.figshare.com/files/12400388",
                "https://ndownloader.figshare.com/files/12400400",
                "https://ndownloader.figshare.com/files/12400385",
                "https://ndownloader.figshare.com/files/12400382",
                "https://ndownloader.figshare.com/files/12400376",
                "https://ndownloader.figshare.com/files/12400373",
                "https://ndownloader.figshare.com/files/12400370",
                "https://ndownloader.figshare.com/files/12400349",
                "https://ndownloader.figshare.com/files/12400346",
                "https://ndownloader.figshare.com/files/12400343",
                "https://ndownloader.figshare.com/files/12400271",
                "https://ndownloader.figshare.com/files/12400268",
                "https://ndownloader.figshare.com/files/12400265",
                "https://ndownloader.figshare.com/files/12400262",
                "https://ndownloader.figshare.com/files/12400259",
                "https://ndownloader.figshare.com/files/12400256"
            ]
        super().__init__(download, merge_list, "5f", download_uri, classes, samples_frequency_in_Herz=1000)

        print(f"\nSummary:\n{self}")
        super().one_hot_encode()


class EEG_BCI_CLA_doi10_6084_m9_figshare_c_3917698_v1(Base_EEG_BCI_Dataset):
    def __init__(self, download: bool = False, merge_list: list = None):
        classes = {0: "blank",
                   1: "left hand MI", 2: "right hand MI", 3: "passive state",
                   4: "left leg MI", 5: "tongue MI", 6: "right  leg MI",
                   # 90: "initial relaxation period",
                   91: "inter-session rest break period", 92: "experiment end", 99: "initial relaxation period"}
        download_uri = [
                "https://ndownloader.figshare.com/files/9636505",
                "https://ndownloader.figshare.com/files/9636502",
                "https://ndownloader.figshare.com/files/9636496",
                "https://ndownloader.figshare.com/files/9636493",
                "https://ndownloader.figshare.com/files/9636490",
                "https://ndownloader.figshare.com/files/9636487",
                "https://ndownloader.figshare.com/files/9636484",
                "https://ndownloader.figshare.com/files/9636475",
                "https://ndownloader.figshare.com/files/9636478",
                "https://ndownloader.figshare.com/files/9636481",
                "https://ndownloader.figshare.com/files/9636472",
                "https://ndownloader.figshare.com/files/9636469",
                "https://ndownloader.figshare.com/files/9636466",
                "https://ndownloader.figshare.com/files/9636463",
                "https://ndownloader.figshare.com/files/12400412",
                "https://ndownloader.figshare.com/files/12400409",
                "https://ndownloader.figshare.com/files/12400406"
            ]
        super().__init__(download, merge_list, "cla", download_uri, classes, samples_frequency_in_Herz=200)

        print(f"\nSummary:\n{self}")
        super().one_hot_encode()


class EEG_BCI_HaLT_doi10_6084_m9_figshare_c_3917698_v1(Base_EEG_BCI_Dataset):
    def __init__(self, download: bool = False, merge_list: list = None):
        classes = {0: "blank",
                   1: "left hand MI", 2: "right hand MI", 3: "passive state",
                   4: "left leg MI", 5: "tongue MI", 6: "right  leg MI",
                   # 90: "initial relaxation period",
                   91: "inter-session rest break period", 92: "experiment end", 99: "initial relaxation period"}
        download_uri = [
                "https://ndownloader.figshare.com/files/9636613",
                "https://ndownloader.figshare.com/files/9636610",
                "https://ndownloader.figshare.com/files/9636607",
                "https://ndownloader.figshare.com/files/9636604",
                "https://ndownloader.figshare.com/files/9636601",
                "https://ndownloader.figshare.com/files/9636598",
                "https://ndownloader.figshare.com/files/9636595",
                "https://ndownloader.figshare.com/files/9636592",
                "https://ndownloader.figshare.com/files/9636589",
                "https://ndownloader.figshare.com/files/9636586",
                "https://ndownloader.figshare.com/files/9636583",
                "https://ndownloader.figshare.com/files/9636574",
                "https://ndownloader.figshare.com/files/9636577",
                "https://ndownloader.figshare.com/files/9636580",
                "https://ndownloader.figshare.com/files/9636571",
                "https://ndownloader.figshare.com/files/9636568",
                "https://ndownloader.figshare.com/files/9636562",
                "https://ndownloader.figshare.com/files/9636559",
                "https://ndownloader.figshare.com/files/9636556",
                "https://ndownloader.figshare.com/files/9636553",
                "https://ndownloader.figshare.com/files/9636550",
                "https://ndownloader.figshare.com/files/9636541",
                "https://ndownloader.figshare.com/files/9636547",
                "https://ndownloader.figshare.com/files/9636544",
                "https://ndownloader.figshare.com/files/9636532",
                "https://ndownloader.figshare.com/files/9636526",
                "https://ndownloader.figshare.com/files/9636529",
                "https://ndownloader.figshare.com/files/9636523",
                "https://ndownloader.figshare.com/files/9636520"
            ]
        super().__init__(download, merge_list, "halt", download_uri, classes, samples_frequency_in_Herz=200)

        print(f"\nSummary:\n{self}")
        super().one_hot_encode()


class EEG_BCI_FREEFORM_doi10_6084_m9_figshare_c_3917698_v1(Base_EEG_BCI_Dataset):
    def __init__(self, download: bool = False, merge_list: list = None):
        classes = {0: "blank",
                   1: "left hand MI", 2: "right hand MI", 3: "passive state",
                   4: "left leg MI", 5: "tongue MI", 6: "right  leg MI",
                   # 90: "initial relaxation period",
                   91: "inter-session rest break period", 92: "experiment end", 99: "initial relaxation period"}
        download_uri = [
                "https://ndownloader.figshare.com/files/9636517",
                "https://ndownloader.figshare.com/files/9636514",
                "https://ndownloader.figshare.com/files/9636511"
            ]
        super().__init__(download, merge_list, "freeform", download_uri, classes, samples_frequency_in_Herz=200)

        print(f"\nSummary:\n{self}")
        super().one_hot_encode()


class EEG_BCI_NoMT_doi10_6084_m9_figshare_c_3917698_v1(Base_EEG_BCI_Dataset):
    def __init__(self, download: bool = False, merge_list: list = None):
        classes = {# 90: "initial relaxation period",
                   91: "inter-session rest break period", 92: "experiment end", 99: "initial relaxation period"}
        download_uri = [
                "https://ndownloader.figshare.com/files/9636634",
                "https://ndownloader.figshare.com/files/9636631",
                "https://ndownloader.figshare.com/files/9636628",
                "https://ndownloader.figshare.com/files/9636625",
                "https://ndownloader.figshare.com/files/9636622",
                "https://ndownloader.figshare.com/files/9636619",
                "https://ndownloader.figshare.com/files/9636616"
            ]
        super().__init__(download, merge_list, "nomt", download_uri, classes, samples_frequency_in_Herz=200)

        print(f"\nSummary:\n{self}")
        super().one_hot_encode()

