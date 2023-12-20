import os
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from typing import Dict 

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
    