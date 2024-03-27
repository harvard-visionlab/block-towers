import PIL
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from dm_control import mujoco
from IPython.display import HTML
from math import ceil

# from .towerstats import compute_will_fall

default_render_opts = dict(height=360,width=480,camera_id="closeup")

def render_first_frame(physics, render_opts=default_render_opts):
    physics.reset()
    pixels = physics.render(**render_opts)
    image = PIL.Image.fromarray(pixels)
    return image

def render_image(physics, render_opts=default_render_opts):
    pixels = physics.render(**render_opts)
    image = PIL.Image.fromarray(pixels)
    return image

def get_physics_engine(simulation, xml_fun):
    start_positions = simulation['start_positions']

    # setup the xml world model for the physics engine
    world_model = xml_fun(start_positions)

    # initialize the physics engine
    physics = mujoco.Physics.from_xml_string(world_model)  

    # set params / init variables
    physics.model.opt.timestep = simulation['params']['timestep']

    return physics

def render_from_simulation(simulation, xml_fun, render_opts=default_render_opts):
    start_positions = simulation['start_positions']

    # setup the xml world model for the physics engine
    world_model = xml_fun(start_positions)

    # initialize the physics engine
    physics = mujoco.Physics.from_xml_string(world_model)  

    # set params / init variables
    physics.model.opt.timestep = simulation['params']['timestep']
    step = 0
    frames = []

    # render frames
    # resolution of stored frames (framerate) can be coarser than the physics timestep
    # (smaller timesteps = more stable physics; but storing/rendering every tiny step
    #  might not be necessary; e.g., timestep .001 results in good physics, but 
    # videos render only at 30 or 60Hz, i.e., framerate = .033 or .0167)
    for frame in simulation['trajectory']:
        # iterate the physics engine until we reach the next stored frame
        while step < frame['physics_step']:
            physics.step()
            step += 1
    
        # make sure physics engine is in sync with stored trajectory
        assert step==frame['physics_step'], "oops, frames out of sync"
        assert physics.data.time==frame['t'], "oops, time out of sync"

        # manually set the block positions to align with stored positions
        # (they should be identical, but by forcing it we gurantee it)
        for data in frame['data']:
            name = data['name']
            physics.data.geom(name).xpos = np.array(data['xyz'])
            physics.data.geom(name).xmat = np.array(data['xmat'])
        pixels = physics.render(**render_opts)
        frames.append(pixels)
    
    assert len(frames) == len(simulation['trajectory'])

    return frames  

def show_tower(positions):
    ''' Visualize tower positions using matplotlib.
    '''
    if not isinstance(positions[0], DotDict): positions = [DotDict(p) for p in positions]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')

    for idx, pos in enumerate(positions):
        height = positions[idx]['lz']
        width = positions[idx]['lx']
        if (positions[idx]['unstable']):
            color = 'red'
        else:
            color = 'blue'

        ax1.add_patch(
            patches.Rectangle(
                (pos['x']-width/2, pos['z']-height/2),   # (x,y)
                width,           # width
                height,          # height
                color=color
            )
        )
    plt.xlim([-5*width, 5*width])
    plt.ylim([0, 10*height])
    if anyFall:
        plt.title('Unstable')
    else:
        plt.title('Stable')
        
def show_tower_grid(list_of_positions, max_cols=4):
    ''' Visualize multiple towers positions using matplotlib in a grid format. '''
    
    sideLength = list_of_positions[0][0]['lx']

    # Calculate the number of rows needed based on the number of towers and max columns
    num_towers = len(list_of_positions)
    num_rows = ceil(num_towers / max_cols)
    
    # Create a figure with subplots in a grid
    fig, axs = plt.subplots(num_rows, max_cols, figsize=(max_cols * 3, num_rows * 3))
    if num_rows == 1: axs = [axs]
    
    # Iterate over each tower and its position in the grid    
    for i, positions in enumerate(list_of_positions):
        anyFall = any([p['unstable'] for p in positions])
        
        # Calculate the current row and column in the grid
        row, col = divmod(i, max_cols)
        
        # Get the appropriate axes for this subplot
        ax = axs[row][col]
        ax.set_aspect('equal')
        
        for idx, pos in enumerate(positions):
            height = positions[idx]['lz']
            width = positions[idx]['lx']
            color = 'red' if positions[idx]['unstable'] else 'blue'

            ax.add_patch(
                patches.Rectangle(
                    (pos['x'] - width / 2, pos['z'] - height / 2),   # (x,y)
                    width,           # width
                    height,          # height
                    color=color
                )
            )
        
        ax.set_title('Unstable' if anyFall else 'Stable')
        ax.axis('square')
        ax.set_xlim([-5 * sideLength, 5 * sideLength])
        ax.set_ylim([0, 10*sideLength])

    # Turn off unused subplots
    for j in range(i + 1, num_rows * max_cols):
        row, col = divmod(j, max_cols)
        axs[row][col].axis('off')

    plt.tight_layout()
    plt.show()

def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
        im.set_data(frame)
        return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    return HTML(anim.to_html5_video())