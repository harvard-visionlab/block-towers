'''
    Code for generating BlockTowers and running physics simulations
    while storing the trajectories / video frames.
'''
from dm_control import mujoco
from fastprogress import progress_bar
from pdb import set_trace

from .towerstats import compute_will_fall

def get_num_boxes(physics):
    box_type_index = mujoco.mjtGeom.mjGEOM_BOX.value
    is_box = physics.model.geom_type == box_type_index
    num_boxes = sum(is_box)
    return num_boxes

def get_geom_names(physics):
    return physics.named.data.geom_xpos.axes.row.names

def get_geom_types(physics):
    return physics.named.model.geom_type

def get_geom_data(physics, name):
    return physics.data.geom(name)

def get_box_positions(physics):
    num_boxes = get_num_boxes(physics)
    box_positions = {}
    for box_idx in range(num_boxes):
        box_name = f'box{box_idx}'
        x = physics.named.data.geom_xpos[box_name,'x']
        y = physics.named.data.geom_xpos[box_name,'y']
        z = physics.named.data.geom_xpos[box_name,'z']
        box_positions[box_name] = [x,y,z]
  
    return box_positions

def get_box_data(physics):
    num_boxes = get_num_boxes(physics)
    box_data = []
    for box_idx in range(num_boxes):
        box_name = f'box{box_idx}'
        data = physics.data.geom(box_name)
        box_data.append(dict(
            id=data.id,
            name=data.name,
            xmat=data.xmat.flatten().tolist(),
            xyz=data.xpos.flatten().tolist()
        ))
    return box_data 

def run_simulation(physics, duration, framerate, timestep=.001, render_frames=False, render_opts={}):
    physics.model.opt.timestep = timestep
    physics.reset()  # Reset state and time
    trajectory = []  
    frames = []
    step_num = 0
    frame_num = 0
    while physics.data.time < duration:  
        if len(trajectory) <= physics.data.time * framerate:
            if render_frames:
                pixels = physics.render(**render_opts)
                frames.append(pixels)
            curr_data = get_box_data(physics)
            trajectory.append(dict(
                physics_step=step_num,
                t=physics.data.time,
                video_frame=frame_num,
                video_t=frame_num*(1/framerate),        
                data=curr_data,
            ))
            frame_num += 1
        physics.step() 
        step_num+=1 
  
    return trajectory, frames

def generate_trajectory(start_positions, xml_fun, duration=3, framerate=60, timestep=.001, scale_factor=1.0,
                        render_frames=False, render_opts=dict(height=360,width=480,camera_id="closeup")):
    # scale the item locations and sizes by scale_factor
    scaled_positions = [{k:v/scale_factor if isinstance(v,(int,float)) else v for k,v in pos.items()} for pos in start_positions]

    # setup the xml world model for the physics engine
    world_model = xml_fun(scaled_positions)

    # initialize the physics engine
    physics = mujoco.Physics.from_xml_string(world_model)  

    # run the simulation
    trajectory, frames = run_simulation(physics, duration, framerate, timestep=timestep, 
                                        render_frames=render_frames, render_opts=render_opts)
    
    # get the final positions
    final_positions = []
    data = trajectory[-1]['data']
    for box in trajectory[-1]['data']:
        x,y,z = box['xyz']
        final_positions.append(dict(x=x, y=y, z=z))
    
    simulation = dict(
        params=dict(duration=duration,framerate=framerate,timestep=timestep,scale_factor=scale_factor),             
        start_positions=scaled_positions,
        final_positions=final_positions,
        trajectory=trajectory,      
    )

    return simulation, frames

def generate_batch_initial_positions(gen_fun, num_blocks=3, side_length=.40, std=.350, truncate=.60, num_samples=1000, pct_fall=.50, mb=None):
    num_unstable = int(num_samples*pct_fall)
    num_stable = num_samples - num_unstable
    stable = []
    unstable = []
    pbar = progress_bar(range(num_samples), parent=mb)
    pbar.comment = 'Initializing'
    pbar.update(0)
    while (len(stable) < num_stable) or (len(unstable) < num_unstable):
        # positions = gen_start_positions_cubes(num_blocks, sideLength=side_length, std=std, truncate=truncate)
        positions = gen_fun(num_blocks, side_length=side_length, std=std, truncate=truncate)
        anyFall, isUnstable = compute_will_fall(positions)
        if (len(stable) < num_stable) and (anyFall==False):
            stable.append(positions)
            pbar.update(len(stable)+len(unstable))
        if (len(unstable) < num_unstable) and (anyFall==True):
            unstable.append(positions)
            pbar.update(len(stable)+len(unstable))
    return stable, unstable