import pickle
import numpy as np
import os 
from collections import defaultdict
from abc import ABCMeta, abstractmethod
from tqdm import tqdm

from typing import Dict
import torch
import lpips

from multiprocessing import Pool

import math
from chamfer_distance import ChamferDistance as chamfer_dist

import asyncio
from PIL import Image
from torch.utils.data import DataLoader

from .projection import Renderer
from .dataset import Batch
import pdb


from .models.dgcnn.model import DGCNN

from omegaconf import OmegaConf
import yaml

RENDERING_MATPLOTLIB = "matplotlib"
RENDERING_MITSUBA = "mitsuba"

LPIPS_PROX = "proximity"
LPIPS_DIV = "diversity"



class Metric(metaclass=ABCMeta):
    """
    Abstract base class for defining metrics in experiments.

    Attributes:
        output_dir (str): The directory to store metric results.
        output_file (str): The full path to the output file for storing metric results.
        experiment (str): The name of the experiment associated with the metric.

    Methods:
        __init__(self, experiment: str, metric: str = None):
            Initializes the Metric object.

        evaluate(self, dataloader: DataLoader, verbose: bool = True) -> Dict:
            Abstract method to evaluate the metric using the provided DataLoader.

        get_results(self) -> Dict:
            Abstract method to get the results of the metric evaluation.
    """

    def __init__(self, experiment: str, metric: str = None):
        """
        Initializes a Metric object.

        Parameters:
            experiment (str): The name of the experiment associated with the metric.
            metric (str, optional): The specific metric identifier. Defaults to None.
        """
        self.output_dir = os.path.join(experiment, "metrics")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.output_dir = os.path.join(self.output_dir, metric)
        self.output_file = self.output_dir + ".pkl"
        self.experiment = experiment

    @abstractmethod
    def evaluate(self, dataloader: DataLoader, verbose: bool = True) -> Dict:
        """
        Abstract method to evaluate the metric using the provided DataLoader.

        Parameters:
            dataloader (DataLoader): DataLoader containing the data for metric evaluation.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.

        Returns:
            Dict: Results of the metric evaluation.
        """
        pass

    @abstractmethod
    def get_results(self) -> Dict:
        """
        Abstract method to get the results of the metric evaluation.

        Returns:
            Dict: Results of the metric evaluation.
        """
        pass


class SyncMetric(Metric, metaclass=ABCMeta):
    """
    Abstract subclass of Metric for synchronous metric evaluation.

    This class extends the Metric class, providing a common structure for synchronous metric
    evaluation on batches of data.

    Methods:
        __init__(self, experiment: str, metric: str = None):
            Initializes the SyncMetric object.

        evaluate(self, dataloader: DataLoader, verbose: bool = True) -> Dict:
            Evaluates the metric synchronously on batches of data from the provided DataLoader.

        measure(self, batch: Batch) -> None:
            Abstract method to measure the metric on a single batch of data.
    """

    def __init__(self, experiment: str, metric: str = None):
        """
        Parameters:
            experiment (str): The name of the experiment associated with the metric.
            metric (str, optional): The specific metric identifier. Defaults to None.
        """
        super().__init__(experiment, metric)

    def evaluate(self, dataloader: DataLoader, verbose: bool = True) -> Dict:
        """
        Evaluates the metric synchronously on batches of data from the provided DataLoader.

        Parameters:
            dataloader (DataLoader): DataLoader containing the data for metric evaluation.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.

        Returns:
            Dict: Results of the metric evaluation.
        """
        iterator = tqdm(dataloader) if verbose else dataloader
        for batch in iterator:
            self.measure(Batch(batch))
        results = self.get_results()
        with open(self.output_file, 'wb') as f:
            pickle.dump(results, f)
        return results

    @abstractmethod
    def measure(self, batch: Batch) -> None:
        """
        Abstract method to measure the metric on a single batch of data.

        Parameters:
            batch (Batch): Object representing a single batch of data.
        """
        pass

class AsyncMetric(Metric, metaclass=ABCMeta):
    """
    Abstract subclass of Metric for asynchronous metric evaluation.

    This class extends the Metric class, providing a common structure for asynchronous metric
    evaluation using asyncio and multiprocessing.

    Attributes:
        pool (Pool): Multiprocessing pool for parallel execution of asynchronous metric measures.

    Methods:
        __init__(self, experiment: str, metric: str = None):
            Initializes the AsyncMetric object.

        evaluate(self, dataloader: DataLoader, verbose: bool = True) -> Dict:
            Evaluates the metric asynchronously on batches of data from the provided DataLoader.

        async_measure(self, batch: Batch) -> None:
            Abstract asynchronous method to measure the metric on a single batch of data.
    """

    def __init__(self, experiment: str, metric: str = None):
        """
        Initializes an AsyncMetric object.

        Parameters:
            experiment (str): The name of the experiment associated with the metric.
            metric (str, optional): The specific metric identifier. Defaults to None.
        """
        super().__init__(experiment, metric)
        self.pool = Pool(60)

    def evaluate(self, dataloader: DataLoader, verbose: bool = True) -> Dict:
        """
        Evaluates the metric asynchronously on batches of data from the provided DataLoader.

        Parameters:
            dataloader (DataLoader): DataLoader containing the data for metric evaluation.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.

        Returns:
            Dict: Results of the metric evaluation.
        """
        async def async_aux(dataloader, verbose):
            batches = [self.async_measure(Batch(batch)) for batch in dataloader]
            await asyncio.gather(*batches)

        asyncio.run(async_aux(dataloader, verbose))
        print("All images accessible")
        results = self.get_results()
        with open(self.output_file, 'wb') as f:
            print("Save results")
            pickle.dump(dict(results), f)
        return results

    @abstractmethod
    async def async_measure(self, batch: Batch) -> None:
        """
        Abstract asynchronous method to measure the metric on a single batch of data.

        Parameters:
            batch (Batch): Object representing a single batch of data.
        """
        pass

class FlipRate(SyncMetric):
    def __init__(self, experiment: str):
        super().__init__(experiment, metric = "flip_rate")
        self.flips = []

    def measure(self, batch: Batch) -> None:
        self.flips += batch.flipped

    def get_results(self) -> Dict:
        return {"flip_rate" : np.mean(self.flips)}

class PNorm(SyncMetric):
    def __init__(self, experiment: str, ord):
        super().__init__(experiment, metric = "lnorm"+str(ord))
        self.diffs = []
        self.ord = ord

    def evaluate(self, dataloader: DataLoader, verbose: bool = True) -> Dict:
        iterator = tqdm(dataloader) if verbose else dataloader
        for batch in iterator:
            self.measure(Batch(batch))
        results = self.get_results()
        with open(self.output_dir, 'wb') as f:
            pickle.dump(results, f)
        return results
      
    def measure(self, batch: Batch) -> None:
        og_data, ce_data = batch.latent_codes
        result = torch.linalg.norm(og_data - ce_data, ord=self.ord, dim=1)
        assert(result.shape == (og_data.shape[0],))
        self.diffs.append(result)

    def get_results(self) -> Dict:
        return {"norm" : np.mean(torch.cat(self.diffs, dim=0).cpu().numpy())}


class ChamferDistance(SyncMetric):
    """
    Metric class for computing Chamfer Distance between two point clouds.

    This class extends the SyncMetric class and provides functionality for measuring the Chamfer
    Distance between original and reconstructed point clouds in an experiment.

    Attributes:
        dist1 (list): List to store the first set of Chamfer distances.
        dist2 (list): List to store the second set of Chamfer distances.
        distance (chamfer_dist): Object for computing Chamfer Distance.

    Methods:
        __init__(self, experiment: str):
            Initializes the ChamferDistance object.

        measure(self, batch: Batch) -> None:
            Measures the Chamfer Distance on a single batch of data.

        my_mean(self, tensor) -> float:
            Computes the mean of the tensor along two dimensions.

        get_results(self) -> Dict:
            Returns the computed Chamfer Distance sum as a dictionary.
    """

    def __init__(self, experiment: str):
        """
        Initializes a ChamferDistance object.

        Parameters:
            experiment (str): The name of the experiment associated with the metric.
        """
        super().__init__(experiment, metric="chamfer")
        self.dist1 = []
        self.dist2 = []
        self.distance = chamfer_dist()

    def measure(self, batch: Batch) -> None:
        """
        Measures the Chamfer Distance on a single batch of data.

        Parameters:
            batch (Batch): Object representing a single batch of data.
        """
        og_data, ce_data = batch.pointclouds
        dist1, dist2, _, _ = self.distance(og_data.permute(0, 2, 1).cuda(), ce_data.permute(0, 2, 1).cuda())
        self.dist1.append(dist1.cpu())
        self.dist2.append(dist2.cpu())

    def my_mean(self, tensor) -> float:
        """
        Computes the mean of the tensor along two dimensions.

        Parameters:
            tensor: Tensor for which the mean is computed.

        Returns:
            float: The computed mean.
        """
        assert len(tensor.shape) == 2
        return torch.mean(torch.mean(tensor, dim=1), dim=0).item()

    def get_results(self) -> Dict:
        """
        Returns the computed Chamfer Distance sum as a dictionary.

        Returns:
            Dict: Dictionary containing the computed Chamfer Distance sum.
        """
        return {"chamfer_dist_sum": self.my_mean(torch.cat(self.dist1, dim=0)) + self.my_mean(torch.cat(self.dist2, dim=0))}
    

def generate_image(renderer, pcd: np.ndarray, output_file: str) -> torch.Tensor:
    """
    Generate an image using a renderer and save it to a file or load from an existing file.

    Parameters:
        renderer: Renderer object used to apply the point cloud and generate the image.
        pcd (np.ndarray): Numpy array representing the point cloud.
        output_file (str): The path to the output file to save the generated image.

    Returns:
        torch.Tensor: The generated image as a torch tensor.

    Raises:
        FileNotFoundError: If the output_file is not found and the image cannot be loaded.
    """
    if not os.path.exists(output_file):
        np_image = renderer.apply(pcd)
        image = Image.fromarray(np_image.astype(np.uint8))
        image.save(output_file)
    else:
        try:
            np_image = np.array(Image.open(output_file))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not load image from {output_file}.") from e
        
    return torch.tensor(np_image).permute(2, 0, 1).unsqueeze(0).float().cpu() / 255.



class LPIPS(AsyncMetric):
    """
    Metric class for computing LPIPS (Learned Perceptual Image Patch Similarity).

    This class extends the AsyncMetric class and provides functionality for measuring LPIPS
    between original and reconstructed images in an experiment.

    Attributes:
        rotations (np.ndarray): Array containing rotation matrices for image rendering.
        renderer (Renderer): Renderer object used to apply the point cloud and generate images.
        results (defaultdict): Dictionary to store LPIPS results.
        lpips (lpips.LPIPS): LPIPS object for computing perceptual similarity.

    Methods:
        __init__(self, experiment: str, rendering_method: str):
            Initializes the LPIPS object.

        __render(self, pcd, foldernames):
            Renders images asynchronously using multiprocessing.

        async_measure(self, batch: Batch) -> None:
            Asynchronously measures LPIPS on a single batch of data.

        get_results(self) -> dict:
            Returns the computed LPIPS results in the form of : .

        __proximity(self) -> None:
            Computes LPIPS proximity between original and reconstructed point clouds.

        __diversity(self) -> None:
            Computes LPIPS diversity among a set of multiple counterfactuals from the same original point cloud.
    """

    def __init__(self, experiment: str, rendering_method: str):
        """
        Initializes an LPIPS object.

        Parameters:
            experiment (str): The name of the experiment associated with the metric.
            rendering_method (str): The method used for rendering images.
        """
        super().__init__(experiment, metric="lpips")
        os.makedirs(self.output_dir, exist_ok=True)
        
        if rendering_method == "matplotlib":
            rotations = [np.array([[math.cos(r),0,math.sin(r)],[0,1,0],[-math.sin(r),0,math.cos(r)]]) for r in np.array([0., 1., 1./2.,-1./2.]) * math.pi] #rotation around y axis
            rotations += [np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]]) for r in np.array([-1./2., 1./2.])* math.pi] #rotation around x axis
        else :
            r = math.pi / 2.
            init_rotation = np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]]) #rotation around x axis
            r = -math.pi /2.
            init_rotation = np.array([[math.cos(r),-math.sin(r),0],[math.sin(r),math.cos(r),0],[0,0,1]]) @ init_rotation #rotation around z axis
            rotations = [np.array([[math.cos(r),-math.sin(r),0],[math.sin(r),math.cos(r),0],[0,0,1]]) @ init_rotation for r in np.array([0., 1., 1./2.,-1./2.]) * math.pi] #rotation around z axis
            rotations += [np.array([[math.cos(r), 0, math.sin(r)], [0,1,0], [-math.sin(r), 0, math.cos(r)]]) @ init_rotation for r in np.array([-1./2., 1./2.])* math.pi] #rotation around y axis   
        self.rotations = np.concatenate(rotations, axis = 0)
        self.renderer = Renderer(rendering_method)
        self.results = defaultdict(lambda : defaultdict(dict))
        self.lpips = lpips.LPIPS(net='vgg').cuda()

    def __render(self, pcd, foldernames):
        pcd = pcd.permute(0,2,1).cpu().numpy()
        rotated_pcl = np.einsum('ijk, kl -> ijl', pcd, self.rotations.T) # N x 2048 x (3 x nRotation) 
        pool_params = [[self.renderer, rr, os.path.join(self.output_dir, foldernames[i]), ii] for (i, r) in enumerate(rotated_pcl) for ii, rr in enumerate(r.reshape(2048, -1, 3).transpose([1, 0, 2]))]

        final_pool_params = []
        for p in pool_params:
            os.makedirs(p[2], exist_ok=True)
            p[2]=os.path.join(p[2], str(p[3]) + ".png")
            final_pool_params.append(p[:3])
        
        async_result = self.pool.starmap_async(generate_image, final_pool_params)
        return async_result
    
    async def async_measure(self, batch: Batch) -> None:
        og_data, ce_data = batch.pointclouds
        sample_ids, ce_ids = batch.sample_ids, batch.ce_ids
        awaitable_og_images = self.__render(og_data, [f"{sample_id:04d}/" for sample_id in sample_ids])
        awaitable_ce_images = self.__render(ce_data, [f"{sample_id:04d}/{ce_id:03d}/" for sample_id, ce_id in zip(sample_ids, ce_ids)])
        
        awaitable_og_images.wait()
        awaitable_ce_images.wait()
        """
        og_images = torch.cat(awaitable_og_images.get(), dim = 0)
        ce_images = torch.cat(awaitable_ce_images.get(), dim = 0)
        _, *img_dim = og_images.shape
        nRotation = int(self.rotations.shape[0] / 3)
        og_images = og_images.reshape(-1, nRotation, *img_dim)
        ce_images = ce_images.reshape(-1, nRotation, *img_dim)

        async with self.lock:
            for (sample_id, ce_id, og, ce) in zip(sample_ids, ce_ids, og_images, ce_images):
                if sample_id not in self.results:
                    self.results[sample_id].update({"images": og})
                self.results[sample_id][ce_id].update({"images": ce, LPIPS_DIV: []})
        """

    @torch.no_grad()
    def get_results(self) -> dict:
        self.__proximity()
        self.__diversity()
        return self.results
    
    def __proximity(self):
        sample_dirs = os.listdir(self.output_dir) #dir of the metric
        for sample_dir in tqdm(sample_dirs):
            current_dir = os.path.join(self.output_dir, sample_dir)
            sample_listdir = os.listdir(current_dir)
            sample_img_dirs = [f for f in sample_listdir if f.endswith(".png")]
            sample_images = []
            for f in sample_img_dirs:
                img = torch.from_numpy(np.array(Image.open(os.path.join(current_dir, f)))).permute(2,0,1).unsqueeze(0).float().cuda() / 255.
                sample_images.append(img)
            sample_images = torch.cat(sample_images, dim=0)
            assert(sample_images.shape[0] == 6)
            ce_dirs = [f for f in sample_listdir if not(f.endswith(".png"))]
            for ce_dir in ce_dirs:
                image_dirs = [os.path.join(current_dir, ce_dir, f) for f in os.listdir(os.path.join(current_dir,ce_dir))]
                ce_images = []
                for image_dir in image_dirs:
                    image = torch.from_numpy(np.array(Image.open(image_dir))).permute(2,0,1).unsqueeze(0).float().cuda() / 255.
                    ce_images.append(image)
                ce_images = torch.cat(ce_images, dim=0)
                assert(ce_images.shape[0] == sample_images.shape[0])
                lpips_result = self.lpips(sample_images.cuda(), ce_images.cuda())
                self.results[LPIPS_PROX][sample_dir][ce_dir] = lpips_result.cpu().mean().item()

    def __diversity(self):
        sample_dirs = os.listdir(self.output_dir) #dir of the metric
        for sample_dir in tqdm(sample_dirs):
            current_dir = os.path.join(self.output_dir, sample_dir)
            ce_dirs = [f for f in os.listdir(current_dir) if not(f.endswith(".png"))]
            if len(ce_dirs) >= 2:
                all_ce_images = []
                for ce_dir in ce_dirs:
                    ce_images = []
                    image_dirs = [os.path.join(current_dir, ce_dir, f) for f in os.listdir(os.path.join(current_dir, ce_dir))]
                    for image_dir in image_dirs:
                        image = torch.from_numpy(np.array(Image.open(image_dir))).permute(2,0,1).unsqueeze(0).float().cuda() / 255.
                        ce_images.append(image)
                    all_ce_images.append(torch.cat(ce_images, dim=0))

                for i in range(len(all_ce_images)):
                    self.results[LPIPS_DIV][sample_dir][ce_dirs[i]] = {}
                    
                for i in range(len(all_ce_images)):
                    for j in range(i+1, len(all_ce_images)):
                        lpips_result = self.lpips(all_ce_images[i].cuda(), all_ce_images[j].cuda())
                        self.results[LPIPS_DIV][sample_dir][ce_dirs[i]][ce_dirs[j]] = lpips_result.cpu().mean().item()
                        self.results[LPIPS_DIV][sample_dir][ce_dirs[j]][ce_dirs[i]] = lpips_result.cpu().mean().item()

class FID(SyncMetric):
    def __init__(self, experiment: str, device: str = 'cuda:0'):
        super().__init__(experiment, metric = "fid")
        args_classifier = {
            'dropout': 0.0,
            'num_points': 2048,
            'k' : 40,
            'emb_dims': 1024,
            'use_sgd' : True,
            'eval': True,
        }
        filename = os.path.join(os.path.dirname(__file__), 'checkpoints/dgcnn_2048_backup/models/model.t7')
        self.model = DGCNN(OmegaConf.create(args_classifier), output_channels=55)
        self.model.load_state_dict(torch.load(filename))
        self.model.to(device)
        self.model = self.model.eval()
        self.og_features_cov_sum = float(0)
        self.og_features_sum = float(0)
        self.og_features_num_samples = int(0)
        self.ce_features_cov_sum = float(0)
        self.ce_features_sum = float(0)
        self.ce_features_num_samples = int(0)

    def measure(self, batch: Batch) -> None:
        og_data, ce_data = batch.pointclouds
        og_data = og_data.permute(0,2,1).cuda()
        ce_data = ce_data.permute(0,2,1).cuda()
        og_data = self.model(og_data)
        ce_data = self.model(ce_data)
        self.orig_dtype = og_data.dtype
        self.__update(og_data)
        self.__update(ce_data, real = False)


    def get_results(self) -> Dict:
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")
        mean_og = (self.og_features_sum / self.og_features_num_samples).unsqueeze(0)
        mean_ce = (self.ce_features_sum / self.ce_features_num_samples).unsqueeze(0)

        cov_og_num = self.og_features_cov_sum - self.og_features_num_samples * mean_og.t().mm(mean_og)
        cov_og = cov_og_num / (self.og_features_num_samples - 1)
        cov_ce_num = self.ce_features_cov_sum - self.ce_features_num_samples * mean_ce.t().mm(mean_ce)
        cov_ce = cov_ce_num / (self.ce_features_num_samples - 1)
        
        a = (mean_og.squeeze(0) - mean_ce.squeeze(0)).square().sum(dim=-1)
        b = cov_og.trace() + cov_ce.trace()
        c = torch.linalg.eigvals(cov_og @ cov_ce).sqrt().real.sum(dim=-1)

        return (a + b - 2 * c).to(self.orig_dtype)

    
    def __update(self, features : torch.Tensor, real = True):
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.og_features_sum += features.sum(dim=0)
            self.og_features_cov_sum += features.t().mm(features)
            self.og_features_num_samples += features.shape[0]
        else:
            self.ce_features_sum += features.sum(dim=0)
            self.ce_features_cov_sum += features.t().mm(features)
            self.ce_features_num_samples += features.shape[0]
    
"""

class LPIPS(SyncMetric):
    def __init__(self, experiment: str, rendering_method: str): 
        super().__init__(experiment, metric="lpips")
        os.makedirs(self.output_dir, exist_ok=True)
        
        if rendering_method == "matplotlib":
            rotations = [np.array([[math.cos(r),0,math.sin(r)],[0,1,0],[-math.sin(r),0,math.cos(r)]]) for r in np.array([0., 1., 1./2.,-1./2.]) * math.pi] #rotation around y axis
            rotations += [np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]]) for r in np.array([-1./2., 1./2.])* math.pi] #rotation around x axis
        else :
            r = math.pi / 2.
            init_rotation = np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]]) #rotation around x axis
            r = -math.pi /2.
            init_rotation = np.array([[math.cos(r),-math.sin(r),0],[math.sin(r),math.cos(r),0],[0,0,1]]) @ init_rotation #rotation around z axis
            rotations = [np.array([[math.cos(r),-math.sin(r),0],[math.sin(r),math.cos(r),0],[0,0,1]]) @ init_rotation for r in np.array([0., 1., 1./2.,-1./2.]) * math.pi] #rotation around z axis
            rotations += [np.array([[math.cos(r), 0, math.sin(r)], [0,1,0], [-math.sin(r), 0, math.cos(r)]]) @ init_rotation for r in np.array([-1./2., 1./2.])* math.pi] #rotation around y axis   
        self.rotations = np.concatenate(rotations, axis = 0)
        self.renderer = Renderer(rendering_method)
        self.results = defaultdict(lambda : defaultdict(list))
        self.lpips = lpips.LPIPS(net='vgg').cuda()

    def __render(self, pcd, foldernames):
        pcd = pcd.permute(0,2,1).cpu().numpy()
        rotated_pcl = np.einsum('ijk, kl -> ijl', pcd, self.rotations.T) # N x 2048 x (3 x nRotation) 
        pool_params = [[self.renderer, rr, os.path.join(self.output_dir, foldernames[i]), ii] for (i, r) in enumerate(rotated_pcl) for ii, rr in enumerate(r.reshape(2048, -1, 3).transpose([1, 0, 2]))]
        
        images = []
        for p in pool_params:
            os.makedirs(p[2], exist_ok=True)
            p[2]=os.path.join(p[2], str(p[3]) + ".png")

            images.append(generate_image(*p[:3]))
        return images

    
    def measure(self, batch: Batch) -> None:
        og_data, ce_data = batch.pointclouds
        sample_ids, ce_ids = batch.sample_ids, batch.ce_ids
        og_images = self.__render(og_data, [f"{sample_id:04d}/" for sample_id in sample_ids])
        og_images = np.concatenate(og_images, axis = 0)
        ce_images = self.__render(ce_data, [f"{sample_id:04d}/{ce_id:03d}/" for sample_id, ce_id in zip(sample_ids, ce_ids)])
        ce_images = np.concatenate(ce_images, axis = 0)
        _, *img_dim = og_images.shape
        nRotation = len(self.rotations)
        og_images = og_images.reshape(-1, nRotation, *img_dim)
        ce_images = ce_images.reshape(-1, nRotation, *img_dim)
        for (sample_id, ce_id, og, ce) in zip(sample_ids, ce_ids, og_images, ce_images):
            if sample_id not in self.results:
                self.results[sample_id].update({"images": og})
            self.results[sample_id][ce_id].update({"images": ce})

    @torch.no_grad()
    def get_results(self) -> dict:
        for sample_id in self.results.items():
            og_images = self.results[sample_id]["images"].cuda()
            ces = [ce for ce in self.results[sample_id].keys() if ce != "images"]
            for i in range(len(ces)):
                ce_images = self.results[sample_id][ces[i]]["images"].cuda()
                self.results[sample_id][ces[i]][LPIPS_PROX] = self.lpips(og_images, ce_images).cpu().mean().item()
                for j in range(i+1, len(ces)):
                    lpips_result = self.lpips(ce_images, self.results[sample_id][ces[j]]["images"].cuda()).cpu().mean().item()
                    self.results[sample_id][ces[i]][LPIPS_DIV].append(lpips_result.mean().item())
                    self.results[sample_id][ces[j]][LPIPS_DIV].append(lpips_result.mean().item())                
        return self.results

"""