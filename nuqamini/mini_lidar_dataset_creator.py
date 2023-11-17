#import rangevit_extract
#from .utils_extract import RangeProjection

##
# Create files under /range_projection_outputs/

from nuscenes.nuscenes import NuScenes

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import datasets
import matplotlib.pyplot as plt
import pandas as pd
import json
import io
import os
import base64
import random
import torch
import gc


import numpy as np

class RangeProjection(object):
    '''
    Project the 3D point cloud to 2D data with range projection

    Adapted from Z. Zhuang et al. https://github.com/ICEORY/PMF
    
    Comment from boyuan: This comes from https://github.com/valeoai/rangevit/blob/main/dataset/preprocess/projection.py
    '''

    def __init__(self, fov_up, fov_down, proj_w, proj_h, fov_left=-180, fov_right=180):

        # check params
        assert fov_up >= 0 and fov_down <= 0, 'require fov_up >= 0 and fov_down <= 0, while fov_up/fov_down is {}/{}'.format(
            fov_up, fov_down)
        assert fov_right >= 0 and fov_left <= 0, 'require fov_right >= 0 and fov_left <= 0, while fov_right/fov_left is {}/{}'.format(
            fov_right, fov_left)

        # params of fov angeles
        self.fov_up = fov_up / 180.0 * np.pi
        self.fov_down = fov_down / 180.0 * np.pi
        self.fov_v = abs(self.fov_up) + abs(self.fov_down)

        self.fov_left = fov_left / 180.0 * np.pi
        self.fov_right = fov_right / 180.0 * np.pi
        self.fov_h = abs(self.fov_left) + abs(self.fov_right)

        # params of proj img size
        self.proj_w = proj_w
        self.proj_h = proj_h

        self.cached_data = {}

    def doProjection(self, pointcloud: np.ndarray):
        self.cached_data = {}
        # get depth of all points
        depth = np.linalg.norm(pointcloud[:, :3], 2, axis=1)
        # get point cloud components
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]

        # get angles of all points
        yaw = -np.arctan2(y, x)
        pitch = np.arcsin(z / depth)

        # get projection in image coords
        proj_x = (yaw + abs(self.fov_left)) / self.fov_h
        # proj_x = 0.5 * (yaw / np.pi + 1.0) # normalized in [0, 1]
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov_v  # normalized in [0, 1]

        # scale to image size using angular resolution
        proj_x *= self.proj_w
        proj_y *= self.proj_h

        px = np.maximum(np.minimum(self.proj_w, proj_x), 0) # or proj_x.copy()
        py = np.maximum(np.minimum(self.proj_h, proj_y), 0) # or proj_y.copy()

        # round and clamp for use as index
        proj_x = np.maximum(np.minimum(
            self.proj_w - 1, np.floor(proj_x)), 0).astype(np.int32)

        proj_y = np.maximum(np.minimum(
            self.proj_h - 1, np.floor(proj_y)), 0).astype(np.int32)

        self.cached_data['uproj_x_idx'] = proj_x.copy()
        self.cached_data['uproj_y_idx'] = proj_y.copy()
        self.cached_data['uproj_depth'] = depth.copy()
        self.cached_data['px'] = px
        self.cached_data['py'] = py

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        pointcloud = pointcloud[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # get projection result
        proj_range = np.full((self.proj_h, self.proj_w), -1, dtype=np.float32)
        proj_range[proj_y, proj_x] = depth

        proj_pointcloud = np.full(
            (self.proj_h, self.proj_w, pointcloud.shape[1]), -1, dtype=np.float32)
        proj_pointcloud[proj_y, proj_x] = pointcloud

        proj_idx = np.full((self.proj_h, self.proj_w), -1, dtype=np.int32)
        proj_idx[proj_y, proj_x] = indices

        proj_mask = (proj_idx > 0).astype(np.int32)

        return proj_pointcloud, proj_range, proj_idx, proj_mask



## Load and convert existing dataset (projection)
#
# See also https://www.nuscenes.org/nuscenes?tutorial=nuscenes

class NuScenesMiniLidarDatasetCreator():
    def __init__(self, version='v1.0-mini', datapath='data', outputdir='range_projection_outputs'):
        self.nusc = NuScenes(version=version, dataroot=datapath)
        self.dataroot = datapath
        self.cam_directions = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT',
        ]
        self.projection = RangeProjection(
            fov_up=10, fov_down=-30,
            fov_left=-180, fov_right=180,
            proj_h=32, proj_w=2048,
        )
        self.lidar_mapping = dict()
        self.outputdir = outputdir
        pass
    
    def save_projected_data(self, lidar_top_token, lidar_iter_origfilename, numpy_data, outputdir):
        # First, check the mapping to see whether has been recorded
        if lidar_top_token in self.lidar_mapping:
            #print("Found previous token {} in lidar mapping!\nOld path: {}, New path: {}".format(
            #    lidar_top_token, self.lidar_mapping[lidar_top_token], lidar_iter_origfilename,
            #))
            pass
        else:
            self.lidar_mapping[lidar_top_token] = lidar_iter_origfilename
        # Actually save the file
        #np.save(Path(outputdir) / (str(lidar_top_token) + '.npy'), numpy_data)
    
    def process_all_and_save(self):
        '''Process all lidar files and save to outputdir as .npy format.
        
        File name shall be: ${outputdir}/${sample_token}.npy
        '''
        outputdir = self.outputdir
        for scene in self.nusc.scene:
            first_sample_token = scene['first_sample_token']
            last_sample_token = scene['last_sample_token']
            iter_token = first_sample_token
            while iter_token != last_sample_token:
                sample = self.nusc.get('sample', iter_token)
                lidar_iter_token = sample['data']['LIDAR_TOP']
                while lidar_iter_token != "":
                    lidar_iter = self.nusc.get('sample_data', lidar_iter_token)
                    lidar_iter_origfilename = lidar_iter['filename']
                    lidar_iter_filepath = Path('.') / self.dataroot / lidar_iter_origfilename
                    lidar_iter_projected = self.project_lidar_data_from_filepath(str(lidar_iter_filepath))
                    self.save_projected_data(lidar_iter_token, lidar_iter_origfilename, lidar_iter_projected, outputdir)
                    lidar_iter_token = lidar_iter['next']
                    print('*', end='')
                iter_token = sample['next']
                print('.', end='')
            print('\nNext scene!\n')
        
        # Save the mappings file
        with open(Path(self.outputdir) / 'mapping.txt', 'w') as f:
            f.write(str(self.lidar_mapping))
        
        # Close all
        print('\nAll Done!!!\n\n')
        return
    
    @staticmethod
    def project_lidar_data_from_filepath(filepath) -> np.ndarray:
        '''
        Take a lidar data filepath as input, give a (5, 32, 2048) np.ndarray as projected lidar image.
        
        Use RangeProjection()
        '''
        proj_img_mean = torch.tensor([12.12, 10.88, 0.23, -1.04, 0.21], dtype=torch.float)
        proj_img_stds = torch.tensor([12.32, 11.47, 6.91, 0.86, 0.16], dtype=torch.float)
        # read data
        raw_data = np.fromfile(filepath, dtype=np.float32).reshape((-1, 5))
        pointcloud = raw_data[:, :4] # (34688, 4)
        # Do projection below
        my_projection = RangeProjection(fov_up=10, fov_down=-30, fov_left=-180, fov_right=180, proj_h=32, proj_w=2048)
        proj_pointcloud, proj_range, proj_idx, proj_mask = my_projection.doProjection(pointcloud)
        # Conversion to tensor, from range_view_loader.py
        proj_mask_tensor = torch.from_numpy(proj_mask)
        proj_range_tensor = torch.from_numpy(proj_range)
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])
        proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3])
        proj_intensity_tensor = proj_intensity_tensor.ne(-1).float() * proj_intensity_tensor
        proj_feature_tensor = torch.cat(
            [proj_range_tensor.unsqueeze(0), proj_xyz_tensor.permute(2, 0, 1), proj_intensity_tensor.unsqueeze(0)], 0)
        proj_feature_tensor = (proj_feature_tensor - proj_img_mean[:, None, None]) / proj_img_stds[:, None, None]
        proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0).float()
        #lidar = proj_feature_tensor.numpy().tolist()
        # size of lidar: (list, len:5), each sublist has size of 32, each subsublist has size of 2048
        ## HOWEVER: We do not use tolist()
        lidar = proj_feature_tensor.numpy()
        # this time shape: (5, 32, 2048)
        return lidar
    
    
### Test our class
if __name__ == '__main__':
    creator = NuScenesMiniLidarDatasetCreator(
        version='v1.0-mini',
        datapath='dataset/v1.0-mini/data/sets/nuscenes/',
        outputdir='dataset/v1.0-mini/data/sets/range_projection_outputs/')
    creator.process_all_and_save()
