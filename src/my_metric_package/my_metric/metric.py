import pickle
import numpy as np

class Metric:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self):
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def get_dataset_info(self):
        return self.data['dataset'], self.data['type'], self.data['split']

    # Add more methods as needed to extract information from the loaded data

    # Example method:
    def get_original_latent_code(self):
        return self.data['latent_code_original']

    def get_ce_latent_code(self):
        return self.data['latent_code_ce']

    # Add more methods as needed

    # You can also add methods for computations or analysis based on the loaded data

# Add more classes or functions as needed
