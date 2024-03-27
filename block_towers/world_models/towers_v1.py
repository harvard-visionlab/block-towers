'''
  Helper functions for Generating Mujoco XML "World Models"
'''
default_colors = [
    [1, 0, 0, 1],
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 1],
    [0, 0, 1, 1],
    [0, 1, 0, 1],
]

def print_world_model(world_model):
    from lxml import etree
    root = etree.fromstring(world_model)
    print(etree.tostring(root, pretty_print=True).decode())

def add_static_cube(idx, x, y, z, lx, ly, lz, r, g, b, a):
  return f'''
    <geom name="box{idx}" type="box" pos="{x:3.6f} {y:3.6f} {z:3.6f}" size="{lx/2:3.6f} {ly/2:3.6f} {lz/2:3.6f}" rgba="{r:3.3f} {g:3.3f} {b:3.3f} {a:3.3f}" />
  '''

def add_dynamic_cube(idx, x, y, z, lx, ly, lz, r, g, b, a):
  return f'''
    <body name="box{idx}" pos="{x:3.6f} {y:3.6f} {z:3.6f}">
      <joint type="free"/>
      <geom name="box{idx}" type="box" size="{lx/2:3.6f} {ly/2:3.6f} {lz/2:3.6f}" rgba="{r:3.3f} {g:3.3f} {b:3.3f} {a:3.3f}" />
    </body>
  '''

def generate_xml_model_from_start_positions(positions, cam_pos=None, colors=default_colors):
    '''
        This function checks whether the tower is stable or unstable, then
        returns an xml Mujoco world model. For unstable towers we use
        a dynamic world model (where objects can/will move). For the unstable
        tower we use a static world model (nothing can move), because it
        turns out the hardest thing to do in physics simulations is to create
        a stable block tower.
    '''
    anyFall = any([p['unstable'] for p in positions])
    if anyFall:
        xml = generate_dynamic_world_model(positions, cam_pos=cam_pos, colors=colors)
    else:
        xml = generate_static_world_model(positions, cam_pos=cam_pos, colors=colors)

    return xml

def generate_dynamic_world_model(positions, cam_pos=None, colors=default_colors):
  '''

    Inputs:
    `positions` is a list of dictionaries, one per block in the tower.
    Each dictionary describes the initial starting locations of the blocks
    (x,y,z), and the side lengths of the blocks (lx, ly, lz).


    Outputs:
    The function outputs a `world_model` in xml format, to be 
    consumed by mujoco physics simulator.

    The boxes are added as a geom type="box" (see Mujoco docs)

  '''
  if cam_pos is None:
    pos = positions[0]
    max_side = max([pos['lx'], pos['ly'], pos['lz']])
    cam_pos = (0, -max_side*10, max_side)

  cam_x, cam_y, cam_z = cam_pos

  world_model = f"""
  <mujoco model="tippe top">

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="600" height="600"/>
    <material name="grid" texture="grid" texrepeat="14 14" reflectance="0"/>
  </asset>

  <worldbody>
    <geom name="floor" size="1 1 .01" type="plane" material="grid"/>
    <light pos="0 0 1" castshadow="false" diffuse="1 1 1"/>
    <camera name="closeup" fovy="45" mode="targetbody" target="lookhere"
      pos="{cam_x} {cam_y} {cam_z}" xyaxes="1 0 0 0 1 2"/>
  """

  for idx,p in enumerate(positions):
    r, g, b, a = colors[idx]
    xml = add_dynamic_cube(idx, p['x'], p['y'], p['z'], 
                           p['lx'], p['ly'], p['lz'], 
                           r, g, b, a)    

    world_model += f"{xml}\n"

  world_model += f"""
    <body name="lookhere" pos="0 0 {max_side*2:3.6f}" />
    
  </worldbody>
</mujoco>

  """

  return world_model

def generate_static_world_model(positions, cam_pos=None, colors=default_colors):
  '''

    Inputs:
    `positions` is a list of dictionaries, one per block in the tower.
    Each dictionary describes the initial starting locations of the blocks
    (x,y,z), and the side lengths of the blocks (lx, ly, lz).

    `cam_pos` (optional) can be the x,y,z position of the camera.

    `colors` (optional): list of rgba tuples (length must be >= #blocks).

    Outputs:
    The function outputs a `world_model` in xml format, to be 
    consumed by mujoco physics simulator.

    The boxes are added as a geom type="box" (see Mujoco docs)

  '''
  if cam_pos is None:
    pos = positions[0]
    max_side = max([pos['lx'], pos['ly'], pos['lz']])
    cam_pos = (0, -max_side*10, max_side)

  cam_x, cam_y, cam_z = cam_pos

  world_model = f"""
  <mujoco model="tippe top">

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="600" height="600"/>
    <material name="grid" texture="grid" texrepeat="14 14" reflectance="0"/>
  </asset>

  <worldbody>
    <geom name="floor" size="1 1 .01" type="plane" material="grid"/>
    <light pos="0 0 1" castshadow="false" diffuse="1 1 1"/>
    <camera name="closeup" fovy="45" mode="targetbody" target="lookhere"
      pos="{cam_x} {cam_y} {cam_z}" xyaxes="1 0 0 0 1 2"/>        
  """

  # add each cube
  world_model += '''  
    <body name="tower">
  '''
  for idx,p in enumerate(positions):
    r, g, b, a = colors[idx]
    cube = add_static_cube(idx, p['x'], p['y'], p['z'], 
                   p['lx'], p['ly'], p['lz'], 
                   r, g, b, a)

    world_model += f'''\t\t{cube}'''

  world_model += '''  
    </body>
  '''

  world_model += f"""
    <body name="lookhere" pos="0 0 {max_side*2:3.6f}" />
  </worldbody>
</mujoco>

  """

  return world_model  