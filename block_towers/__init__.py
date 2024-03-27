import os

os.environ['MUJOCO_GL']='egl'

from .cubes import *
from .render import *
from .simulation import *
from .utils import *