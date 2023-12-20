import pickle
import numpy as np
import os 
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from abc import ABCMeta, abstractmethod
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple
import torch
import lpips

import mitsuba as mi
mi.set_variant("scalar_rgb")

from multiprocessing import Pool

import math

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from chamfer_distance import ChamferDistance as chamfer_dist

from PIL import Image

SFX_PCD = "point_cloud_"
SFX_LATENT = "latent_code_"
PFX_CE = "ce"
PFX_OG = "original"

RENDERING_MATPLOTLIB = "matplotlib"
RENDERING_MITSUBA = "mitsuba"

LPIPS_PROX = "proximity"
LPIPS_DIV = "diversity"

class CounterFactualData(Dataset):
    def __init__(self, method_folder: str) :
        self.files = [os.path.join(method_folder, f) for f in os.listdir(method_folder)]
        self.data = defaultdict(dict)
        self.__load_data()

    def get_filename(filepath):
        return filepath.split("/")[-1][:-4]
    
    def __load_data(self):
        for i, f in enumerate(tqdm(self.files)):
            with open(f, 'rb') as f:
                filename = self.get_filename(f)
                sample_id = int(filename.split("_")[0])
                ce_id = int(filename.split("_")[1])
                self.data[i].update(pickle.load(f))
                self.data[i].update({"sample_id": sample_id, "ce_id": ce_id})

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):        
        return idx, self.data[idx]
    
class Metric(metaclass=ABCMeta):
    def __init__(self, experiment: str, dataloader: DataLoader):
        self.output_dir = os.path.join(experiment, "metrics")
        self.experiment = experiment
        self.dataloader = dataloader

    @abstractmethod
    def __measure(self, og_data, ce_data, sample_ids, ce_ids) -> None:
        pass

    @abstractmethod
    def get_results(self) -> dict:
        pass
    

    def __evaluate(self, datatype: str, verbose: bool = True) -> None:
        iterator = tqdm(self.dataloader) if verbose else self.dataloader
        for batch in iterator:
            og_data = batch[PFX_OG + datatype]
            ce_data = batch[PFX_CE + datatype]
            self.__measure(og_data, ce_data, batch["sample_id"], batch["ce_id"])

    def evaluate_pointclouds(self) -> None:
        self.__evaluate(SFX_PCD)
    def evaluate_latent_codes(self):
        self.__evaluate(SFX_LATENT)


class LNorm(Metric):
    def __init__(self, experiment: str, ord):
        super().__init__(experiment)
        self.diffs = []
        self.ord = ord
        self.output_dir = os.path.join(self.output_dir, "lnorm_" + str(ord))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    def norm(self, og_data, ce_data) -> torch.Tensor:
        return torch.linalg.norm(og_data - ce_data, ord=self.ord, dim=1)

    def __measure(self, og_data, ce_data, sample_ids, ce_ids) -> None:
        self.diffs.append(self.norm(og_data, ce_data))

    def get_results(self) -> Dict:
        return {"norm" : np.mean(torch.cat(self.diffs, dim=0).cpu().numpy())}
    
class ChamferDistance(Metric):
    def __init__(self, experiment: str):
        super().__init__(experiment)
        self.dist1 = []
        self.distance = chamfer_dist()
        self.output_dir = os.path.join(self.output_dir, "chamfer")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def __measure(self, og_data, ce_data, sample_ids, ce_ids) -> None:
        dist1, dist2, _, _ = self.distance(og_data.cuda(), ce_data.cuda())
        self.dist1.append(dist1.cpu())
        self.dist2.append(dist2.cpu())

    def get_results(self) -> Dict:
        return {"dist1" : np.mean(self.dist1), "dist2": np.mean(self.dist2)}

class Renderer():
    def __init__(self, rendering :str):
        self.rendering = rendering
        self.xml_head = \
        """
        <scene version="0.6.0">
            <integrator type="path">
                <integer name="maxDepth" value="-1"/>
            </integrator>
            <sensor type="perspective">
                <float name="farClip" value="100"/>
                <float name="nearClip" value="0.1"/>
                <transform name="toWorld">
                    <lookat origin="3,0,0" target="0,0,0" up="0,0,0.5"/>
                </transform>
                <float name="fov" value="25"/>
                
                <sampler type="ldsampler">
                    <integer name="sampleCount" value="256"/>
                </sampler>
                <film type="hdrfilm">
                    <integer name="width" value="1024"/>
                    <integer name="height" value="1024"/>
                    <rfilter type="gaussian"/>
                    <boolean name="banner" value="false"/>
                </film>
            </sensor>
            
            <bsdf type="roughplastic" id="surfaceMaterial">
                <string name="distribution" value="ggx"/>
                <float name="alpha" value="0.05"/>
                <float name="intIOR" value="1.46"/>
                <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
            </bsdf>
            
        """

        self.xml_ball_segment = \
        """
            <shape type="sphere">
                <float name="radius" value="0.025"/>
                <transform name="toWorld">
                    <translate x="{}" y="{}" z="{}"/>
                </transform>
                <bsdf type="diffuse">
                    <rgb name="reflectance" value="{},{},{}"/>
                </bsdf>
            </shape>
        """

        self.xml_tail = \
        """
            <shape type="rectangle">
                <ref name="bsdf" id="surfaceMaterial"/>
                <transform name="toWorld">
                    <scale x="10" y="10" z="1"/>
                    <translate x="0" y="0" z="-0.5"/>
                </transform>
            </shape>
            
            <shape type="rectangle">
                <transform name="toWorld">
                    <scale x="10" y="10" z="1"/>
                    <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
                </transform>
                <emitter type="area">
                    <rgb name="radiance" value="1,1,1"/>
                </emitter>
            </shape>

        </scene>
        """

    def __standardize_bbox(pcd : np.ndarray):
        mins = np.amin(pcd, axis=0)
        maxs = np.amax(pcd, axis=0)
        center = ( mins + maxs ) / 2.
        scale = np.amax(maxs-mins)
        result = ((pcd - center)/scale).astype(np.float32) # [-0.5, 0.5]
        return result
    
    def __colormap(x,y,z):
        vec = np.array([x,y,z])
        vec = np.clip(vec, 0.001,1.0)
        norm = np.sqrt(np.sum(vec**2))
        vec /= norm
        return [vec[0], vec[1], vec[2]]

    @staticmethod
    def __render_mitsuba(self, pcd : np.array) -> np.array: #8UINT RGB
        xml_segments = [self.xml_head]
        pcd = self.__standardize_bbox(pcd)
        #pcd = pcd[:,[1,0,2]]

        for i in range(pcd.shape[0]):
            color = self.__colormap(pcd[i,0]+0.5,pcd[i,1]+0.5,pcd[i,2]+0.5-0.0125)
            xml_segments.append(self.xml_ball_segment.format(pcd[i,0],pcd[i,1],pcd[i,2], *color))
        xml_segments.append(self.xml_tail)

        xml_content = str.join('', xml_segments)
        # Load the scene
        scene = mi.load_string(xml_content)

        # Create an ImageBlock to store the rendered image
        image = mi.render(scene, spp=256)

        mybitmap = mi.Bitmap(image).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)
        image_np = np.array(mybitmap).clip(0,255) #.reshape((image.height(), image.width(), 3)).numpy()

        # Convert to PyTorch tensor
        return image_np
    
    @staticmethod
    def __render_matplotlib(self, pcd : np.array, elevation_angle= 90, azimuthal_angle = -90) -> np.array:
        # Load and standardize point cloud
        pcd = self.__standardize_bbox(pcd)

        # Create a 3D plot
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elevation_angle, azim=azimuthal_angle)

        # Plot each point as a sphere with color mapped to depth
        for i in range(pcd.shape[0]):
            color = self.__colormap(pcd[i, 0] + 0.5, pcd[i, 1] + 0.5, pcd[i, 2] + 0.5 - 0.0125)
            ax.scatter(pcd[i, 0], pcd[i, 1], pcd[i, 2], c=[color], marker='o', s=20)

        ax.axis("off")

        # Create a canvas to render the figure
        canvas = FigureCanvas(fig)
        canvas.draw()
        # Convert the figure to a tensor
        np_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return np_image
    
    def apply(self, pcd : np.array):
        locals()("__render_" + self.rendering)(pcd)

class LPIPS(Metric):
    def __init__(self, experiment: str, rendering_method: str, compare_method: str): 
        super().__init__(experiment)

        self.output_dir = os.path.join(self.output_dir, "lpips")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

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
        self.results = defaultdict(defaultdict(list))
        self.lpips = lpips.LPIPS(net='vgg').cuda()
        self.compare_method = compare_method

    def __render(self, pcd, foldernames):
        pcd = pcd.permute(0,2,1).cpu().numpy()
        rotated_pcl = np.einsum('ijk, kl -> ijl', pcd, self.rotations.T) # N x 2048 x (3 x nRotation) 
        pool_params = [(rr, os.path(self.output_dir, foldernames[i] + str(ii) + ".png")) for (i, r) in enumerate(rotated_pcl) for ii, rr in enumerate(r.reshape(2048, -1, 3).transpose([1, 0, 2]))]
        pool_params = [x for x in pool_params if not(os.path.exists(x[1]))]

        def generate_image(pcd: np.array, output_file: str) -> None:
            assert(not(os.path.exists(output_file)))
            np_image = self.renderer(pcd)
            image = Image.fromarray(np_image.astype(np.uint8))
            image.save(output_file)

        with Pool(60) as p:
            p.starmap(generate_image, pool_params)

    def get_results(self) -> dict:
        if self.compare_method == LPIPS_PROX:
            self.__proximity()
        elif self.compare_method == LPIPS_DIV:
            self.__diversity()
        else:
            raise NotImplementedError
        return self.results

    def __measure(self, og_data, ce_data, sample_ids: List[int], ce_ids: List[int]) -> None:
        self.__render(og_data, [f"{sample_id}/" for sample_id in sample_ids])
        self.__render(ce_data, [f"{sample_id}/{ce_id}/" for sample_id, ce_id in zip(sample_ids, ce_ids)])
        
    def __proximity(self):
        sample_dirs = os.listdir(self.output_dir) #dir of the metric
        for sample_dir in tqdm(sample_dirs):
            current_dir = os.path.join(self.output_dir, current_dir)
            sample_listdir = os.listdir(current_dir)
            sample_img_dirs = [f for f in sample_listdir if f.endswith(".png")]
            sample_images = []
            for f in sample_img_dirs:
                img = torch.from_numpy(np.array(Image.open(os.path.join(sample_dir, f)))).permute(2,0,1).unsqueeze(0).float().cuda() / 255.
                sample_images.append(img)
            sample_images = torch.cat(sample_images, dim=0)

            ce_dirs = [os.path.join(current_dir, f) for f in sample_listdir if not(f.endswith(".png"))]
            for ce_dir in ce_dirs:
                image_dirs = [os.path.join(ce_dir, f) for f in os.listdir(ce_dir)]
                ce_images = []
                for image_dir in image_dirs:
                    image = torch.from_numpy(np.array(Image.open(image_dir))).permute(2,0,1).unsqueeze(0).float().cuda() / 255.
                    ce_images.append(image)
                ce_images = torch.cat(ce_images, dim=0)
                lpips_result = self.lpips(sample_images, ce_images)
                self.results["proximity"][sample_dir].append(lpips_result.mean().item())

    def __diversity(self):
        sample_dirs = os.listdir(self.output_dir) #dir of the metric
        for sample_dir in tqdm(sample_dirs):
            current_dir = os.path.join(self.output_dir, sample_dir)
            ce_dirs = [os.path.join(sample_dir, f) for f in os.listdir(current_dir) if not(f.endswith(".png"))]
            all_ce_images = []
            if len(ce_dirs) >= 2:
                ce_images = []
                for ce_dir in ce_dirs:
                    image_dirs = [os.path.join(ce_dir, f) for f in os.listdir(ce_dir)]
                    for image_dir in image_dirs:
                        image = torch.from_numpy(np.array(Image.open(image_dir))).permute(2,0,1).unsqueeze(0).float().cuda() / 255.
                        ce_images.append(image)
                    all_ce_images.append(torch.cat(ce_images, dim=0))

                for i in range(len(all_ce_images)):
                    for j in range(i+1, len(all_ce_images)):
                        lpips_result = self.lpips(all_ce_images[i], all_ce_images[j])
                        self.results["diversity"][sample_dir].append(lpips_result.mean().item())
        
class MetricManager:
    def __init__(self, experiment: str, dataloader: DataLoader, metrics: List[Metric]):
        self.experiment = experiment
        self.dataloader = dataloader
        self.metrics = metrics

    def evaluate(self, datatype: str = SFX_PCD) -> None:
        for metric in self.metrics:
            metric.evaluate(datatype)
    
    def get_results(self) -> Dict:
        results = {}
        for metric in self.metrics:
            results.update(metric.get_results())
        return results

    def save_results(self, path: str) -> None:
        results = self.get_results()
        with open(os.path.join(path, self.experiment + ".pkl"), 'wb') as f:
            pickle.dump(results, f)
    
    def __repr__(self):
        return f"MetricManager(experiment={self.experiment}, metrics={self.metrics})"    
    
    