from networks.dcase2023t2_ae.dcase2023t2_ae import DCASE2023T2AE
from networks.autotrash_cnn_dae_coral_kl import AutoTrashCnnDaeCoralKl

class Models:
    ModelsDic = {
        "DCASE2023T2-AE":DCASE2023T2AE,
        "autotrash_cnn_dae_coral_kl":AutoTrashCnnDaeCoralKl,
    }

    def __init__(self,models_str):
        self.net = Models.ModelsDic[models_str]

    def show_list(self):
        return Models.ModelsDic.keys()
