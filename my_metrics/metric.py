import pickle
import numpy as np
import os 
from collections import defaultdict
from torch.utils.data import DataLoader
from abc import ABCMeta, abstractmethod
import logging
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from typing import Dict, List, Tuple
import torch
import lpips

from multiprocessing import Pool

import math
from chamfer_distance import ChamferDistance as chamfer_dist

import asyncio
from PIL import Image

from .projection import Renderer

import pdb

SFX_PCD = "point_cloud_"
SFX_LATENT = "latent_code_"
PFX_CE = "ce"
PFX_OG = "original"

RENDERING_MATPLOTLIB = "matplotlib"
RENDERING_MITSUBA = "mitsuba"

LPIPS_PROX = "proximity"
LPIPS_DIV = "diversity"

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
    

class Metric(metaclass=ABCMeta):
    def __init__(self, experiment: str, metric :str = None):
        self.output_dir = os.path.join(experiment, "metrics")
        if not(os.path.exists(self.output_dir)):
            os.makedirs(self.output_dir)
        self.output_dir = os.path.join(self.output_dir, metric)
        self.output_file = self.output_dir + ".pkl"
        self.experiment = experiment

    @abstractmethod
    def evaluate(self, dataloader: DataLoader, verbose: bool = True) -> Dict:
        pass

    @abstractmethod
    def get_results(self) -> Dict:
        pass

class SyncMetric(Metric, metaclass = ABCMeta):
    def __init__(self, experiment: str, metric :str = None):
        super().__init__(experiment, metric)

    def evaluate(self, dataloader: DataLoader, verbose: bool = True) -> Dict:
        iterator = tqdm(dataloader) if verbose else dataloader
        for batch in iterator:
            self.measure(Batch(batch))
        results = self.get_results()
        with open(self.output_file, 'wb') as f:
            pickle.dump(results, f)
        return results
      
    @abstractmethod
    def measure(self, batch : Batch) -> None:
        pass


class AsyncMetric(Metric, metaclass = ABCMeta):
    def __init__(self, experiment: str, metric :str = None):
        super().__init__(experiment, metric)
        self.pool = Pool(60)

    def evaluate(self, dataloader: DataLoader, verbose: bool = True) -> Dict:
        async def async_aux(dataloader, verbose):
            batches = [self.async_measure(Batch(batch)) for batch in dataloader]
            await asyncio.gather(*batches)
            
        asyncio.run(async_aux(dataloader, verbose))
        print("all images accessible")
        results = self.get_results()
        with open(self.output_file, 'wb') as f:
            print("save results")
            pickle.dump(dict(results), f)
        return results
      
    @abstractmethod
    async def async_measure(self, batch : Batch) -> None:
        pass

class FlipRate(SyncMetric):
    def __init__(self, experiment: str):
        super().__init__(experiment, metric = "flip_rate")
        self.flips = []

    def measure(self, batch: Batch) -> None:
        self.flips += batch.flipped

    def get_results(self) -> Dict:
        return {"flip_rate" : np.mean(self.flips)}

class LNorm(SyncMetric):
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
    def __init__(self, experiment: str):
        super().__init__(experiment, metric = "chamfer")
        self.dist1 = []
        self.dist2 = []
        self.distance = chamfer_dist()
    
    def measure(self, batch: Batch) -> None:
        og_data, ce_data = batch.pointclouds
        dist1, dist2, _, _ = self.distance(og_data.permute(0,2,1).cuda(), ce_data.permute(0,2,1).cuda())
        self.dist1.append(dist1.cpu())
        self.dist2.append(dist2.cpu())

    def my_mean(self, tensor) :
        assert(len(tensor.shape) == 2)
        return torch.mean(torch.mean(tensor, dim=1), dim=0).item()
    
    def get_results(self) -> Dict:
        return {"dist1" : self.my_mean(torch.cat(self.dist1, dim = 0)), "dist2": self.my_mean(torch.cat(self.dist2, dim = 0))}
        
def generate_image(renderer, pcd: np.array, output_file: str) -> None:
    if not(os.path.exists(output_file)) :
        np_image = renderer.apply(pcd)
        image = Image.fromarray(np_image.astype(np.uint8))
        image.save(output_file)
    else :
        np_image = np.array(Image.open(output_file))
    return torch.tensor(np_image).permute(2,0,1).unsqueeze(0).float().cpu() / 255.

class LPIPS(AsyncMetric):
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