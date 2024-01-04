import os
import pickle
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import torch
SFX_PCD = "point_cloud_"
SFX_LATENT = "latent_code_"
PFX_CE = "ce"
PFX_OG = "original"

class Batch():
    def __init__(self, batch: Dict):
        self.data = batch

    @property
    def pointclouds(self) -> Tuple[torch.Tensor]:
        return self.data[SFX_PCD + PFX_OG], self.data[SFX_PCD + PFX_CE]
    
    @property
    def latent_codes(self) -> Tuple[torch.Tensor]:
        return self.data[SFX_LATENT + PFX_OG], self.data[SFX_LATENT + PFX_CE]
    
    @property
    def sample_ids(self) -> List[int]:
        return self.data["sample_id"]

    @property
    def ce_ids(self) -> List[int]:
        return self.data["ce_id"]
    
    @property
    def flipped(self) -> List[bool]:
        flip_epoch = self.data["number_of_steps_flip"]
        max_epochs = self.data["max_number_of_steps"]
        return [(i<n) * 1 for i, n in zip(flip_epoch, max_epochs)]
    

class CounterFactualData(Dataset):
    def __init__(self, method_folder: str) -> None:
        self.files = [os.path.join(method_folder, f) for f in os.listdir(method_folder)]
        self.data = defaultdict(dict)
        self.__load_data()

    def __get_filename(self, filepath: str) -> str:
        return filepath.split("/")[-1][:-4]
    
    def __load_data(self) -> None:
        for i, f in enumerate(tqdm(self.files)):
            with open(f, 'rb') as buffer:
                filename = self.__get_filename(f)
                sample_id = int(filename.split("_")[0])
                ce_id = int(filename.split("_")[1])
                self.data[i].update(pickle.load(buffer))
                self.data[i].update({"sample_id": sample_id, "ce_id": ce_id})

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx) -> Dict:        
        return self.data[idx]
    