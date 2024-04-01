'''
    MUJOCO_GL=egl && python test_loader.py
'''
from block_towers.datasets import settings1, generate_blocktower_dataset
from block_towers.cubes import gen_start_positions_cubes
from block_towers.render import show_tower_grid
from datasets import Dataset, DatasetDict, concatenate_datasets
from fastprogress import progress_bar
import os
import numpy as np
from block_towers.world_models import generate_xml_model_from_start_positions
from PIL import Image
from threading import Lock
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import logging
from dm_control import mujoco

transform = transforms.Compose([
    transforms.ToTensor()
])

# Initialize a Lock
render_lock = Lock()

logging.basicConfig(filename='data_loader_errors.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

class TowerRenderDataset(object):
    def __init__(self, start_positions_list, xml_fun, scale_factor=1.0, render_opts=dict(height=360,width=480,camera_id="closeup"),
                 transform=None):
        
        self.samples = start_positions_list
        self.scale_factor = scale_factor
        self.render_opts = render_opts
        self.transform = transform
        
        # initialize the physics simulator
        scaled_positions = [{k:v/self.scale_factor if isinstance(v,(int,float)) else v for k,v in pos.items()} for pos in self.samples[0]]
        self.world_model = xml_fun(scaled_positions)                                   
        self.physics = mujoco.Physics.from_xml_string(self.world_model)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        start_positions = self.samples[index]
        scaled_positions = [{k:v/self.scale_factor if isinstance(v,(int,float)) else v for k,v in pos.items()} for pos in start_positions]
        
        # manually set the block positions to align with stored positions
        for idx,pos in enumerate(scaled_positions):
            name = f"box{idx}"
            self.physics.data.geom(name).xpos = np.array([pos['x'],pos['y'],pos['z']])
        with render_lock:
            img = self.physics.render(**self.render_opts)
        #img = torch.rand(360,480,3)
        if self.transform is not None:
            img = self.transform(np.ascontiguousarray(img))
        return img, self.physics.data.time
    
class TowerRenderDataset2(object):
    def __init__(self, start_positions_list, xml_fun, scale_factor=1.0, render_opts=dict(height=360,width=480,camera_id="closeup"),
                 transform=None):
        
        self.samples = start_positions_list
        self.xml_fun = xml_fun
        self.scale_factor = scale_factor
        self.render_opts = render_opts
        self.transform = transform                
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):        
        start_positions = self.samples[index]
        scaled_positions = [{k:v/self.scale_factor if isinstance(v,(int,float)) else v for k,v in pos.items()} for pos in start_positions]
        
        # initialize the physics simulator
        scaled_positions = [{k:v/self.scale_factor if isinstance(v,(int,float)) else v for k,v in pos.items()} for pos in self.samples[0]]
        world_model = self.xml_fun(scaled_positions)                                   
        physics = mujoco.Physics.from_xml_string(world_model)
        img = physics.render(**self.render_opts)
        if self.transform is not None:
            img = self.transform(np.ascontiguousarray(img))
        return img
    
class TowerRenderDataset3(object):
    def __init__(self, start_positions_list, xml_fun, scale_factor=1.0, render_opts=dict(height=360,width=480,camera_id="closeup"),
                 transform=None):
        
        self.samples = start_positions_list
        self.xml_fun = xml_fun
        self.scale_factor = scale_factor
        self.render_opts = render_opts
        self.transform = transform                
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        from dm_control import mujoco
        
        start_positions = self.samples[index]
        scaled_positions = [{k:v/self.scale_factor if isinstance(v,(int,float)) else v for k,v in pos.items()} for pos in start_positions]
        
        # initialize the physics simulator
        scaled_positions = [{k:v/self.scale_factor if isinstance(v,(int,float)) else v for k,v in pos.items()} for pos in self.samples[0]]
        world_model = self.xml_fun(scaled_positions)                                   
        physics = mujoco.Physics.from_xml_string(world_model)     
        
        try:
            #img = physics.render(**self.render_opts)
            img = torch.rand(360,480,3)
            if self.transform is not None:
                img = self.transform(np.ascontiguousarray(img))
            return img
        except Exception as e:
            # Log the exception with a stack trace
            logging.exception(f"Error processing item {index}: {e}")
            # Return a fallback value or re-raise the exception
            raise e             
    
if __name__ == "__main__":  # It's important to guard the entry point for multiprocessing
    
    print("==> creating dataset")
    settings = {6: {'num_blocks': 6, 'side_length': 0.4, 'std': 0.13, 'truncate': 0.65}}
    dataset = generate_blocktower_dataset(settings, gen_start_positions_cubes, 
                                      num_samples=10000*2, pct_fall=.50, test_size=.20)
    
    px_dataset3 = TowerRenderDataset(dataset['stack6_unstable']['train']['data'], generate_xml_model_from_start_positions,
                                  transform=transform)
    
    print("==> testing no workers")
    train_loader = DataLoader(px_dataset3, batch_size=10, shuffle=True, num_workers=0, pin_memory=True, drop_last=False,
                          prefetch_factor=None, persistent_workers=False,)
    batch = next(iter(train_loader))
    print(batch[0].shape, batch[1])
    
    print("==> testing with workers")
    train_loader = DataLoader(px_dataset3, batch_size=10, shuffle=True, num_workers=2, pin_memory=True, drop_last=False,
                          prefetch_factor=None, persistent_workers=True)
    batch = next(iter(train_loader))
    print(batch[0].shape, batch[1])
    