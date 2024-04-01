import os

if not os.environ.get('MUJOCO_GL', None):
    os.environ['MUJOCO_GL']='egl'

from .cubes import *
from .render import *
from .simulation import *
from .utils import *