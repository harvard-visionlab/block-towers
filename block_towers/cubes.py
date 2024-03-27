from torch.nn.modules.utils import _triple
from collections import OrderedDict
import numpy as np

import matplotlib.pyplot as plt
from fastprogress import progress_bar
import numpy as np
import math
from pdb import set_trace

from .towerstats import compute_will_fall

def init_block(x,y,z,lx,ly,lz,rx=0,ry=0,rz=0,mass=None,density=None):
    return dict([('x', x), ('y', y), ('z', z),       # x,y,z coordinate
                 ('lx', lx), ('ly', ly), ('lz', lz), # length of x,y,z side
                 ('rx', rx), ('ry', ry), ('rz', rz), # orientation (default 0,0,0)
                 ('mass', mass),
                 ('density', density)
                ]);

def coords(x,y,z):
    return dict([('x', x), ('y', y), ('z', z)]);

def bounded_random_normal(mu, std=1, lower=-0.98, upper=0.98, size=None):
    ''' Get random sample, with lower and upper bounds.

        If a value is out of bounds, replace it with another random sample.

        These bounds are needed to make sure no block is ever set "off of" the one below it.
    '''

    x = np.random.normal(mu, std, size)
    idx = np.logical_or(x < (mu+lower), x > (mu+upper))

    while np.any(idx):

        if size is None:
            x = np.random.normal(mu, std)
        else:
            size = np.sum(idx)
            x[idx] = np.random.normal(mu, std, size)

        idx = np.logical_or(x < (mu+lower), x > (mu+upper))

    return x

def gen_start_positions_cubes(numBlocks, side_length, std, truncate=.90, jitter_y=False):
    ''' Generate random initial cube positions, varying only the x position

        numBlocks (int): number of blocks
        sideLength (float): length of cube side (meters)
        std (float): standard deviation used when generating random normal

        x-position is randomly jittered (+/- std centered on xpos of the block below)
        y-position is fixed (zero) by default, otherwise randomly jittered (+/- std centered on ypos of the block below)
        z-position is based on the height of the blocks, so they will be stacked
    '''

    positions = []
    side_lengths = _triple(side_length)
    lx,ly,lz = side_lengths
    lower_x, lower_y, lower_z = [side_length * -truncate for side_length in side_lengths]
    upper_x, upper_y, upper_z = [side_length * truncate for side_length in side_lengths]

    # always start at origin:
    positions.append(init_block(0, 0, lz/2,
                                lx, ly, lz,
                                0, 0, 0))
    
    # stack cubes with jitter
    for block in range(1,numBlocks):
        x = bounded_random_normal(positions[block-1]['x'], std, lower_x, upper_x)
        y = 0 if jitter_y==False else bounded_random_normal(positions[block-1]['y'], std, lower_y, upper_y)
        z = positions[block-1]['z'] + ly
        positions.append(init_block(x, y, z,
                                    lx, ly, lz,
                                    0, 0, 0))
    
    # compute stability at each block, add to record
    _, isUnstable = compute_will_fall(positions)
    for idx in range(len(positions)):
        positions[idx]['unstable'] = int(isUnstable[idx])
    
    return positions

