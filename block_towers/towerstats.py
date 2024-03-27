import numpy as np

# --------------------------------------------------------
#  Check whether a tower will fall (ground truth can be 
#  computed from centroids assuming uniform density)n
# --------------------------------------------------------

def compute_will_fall(pos):
    ''' Compute whether any blocks in the tower will fall.

        A block will fall if and only if the center of mass of the
        block + all blocks above it falls over the edge of the block below, 
        considering both x and y directions.
    '''    
    numBlocks = len(pos)
    willFall = [False]  # The base block can't "fall" as it's on the ground
    
    for block in range(1, numBlocks):
        block_below = block - 1
        below_x = pos[block_below]['x']
        below_y = pos[block_below]['y']
        below_lx = pos[block_below]['lx']
        below_ly = pos[block_below]['ly']
        bounds_x = [below_x - below_lx / 2, below_x + below_lx / 2]
        bounds_y = [below_y - below_ly / 2, below_y + below_ly / 2]

        # Centers of mass for blocks above, including current block
        rest_x = np.array([p['x'] for p in pos[block:]])
        rest_y = np.array([p['y'] for p in pos[block:]])
        center_x = rest_x.mean()
        center_y = rest_y.mean()

        # A block will fall if center of mass is out of bounds in either x or y direction
        fall = bool(center_x < bounds_x[0] or center_x > bounds_x[1] or center_y < bounds_y[0] or center_y > bounds_y[1])
        willFall.append(fall)

    anyFall = bool(np.any(willFall))

    return anyFall, willFall

def compute_ground_truth(pos, sideLength):
    ''''''

    numBlocks = len(pos)

    xs = np.array([p.x for p in pos])

    distances = []

    for block in range(1,numBlocks):
        below = pos[block-1].x
        bounds = [below - sideLength/2, below + sideLength/2]
        rest = np.array([p.x for p in pos[block:]])
        center = rest.mean()

        if center < bounds[0]:
            d = bounds[0] - center
        elif center > bounds[1]:
            d = center - bounds[1]
        else:
            d = max([center - bounds[1], bounds[0] - center])

        distances.append(d)

    return distances, np.max(distances), np.mean(distances)

def compute_stability_metrics(pos, sideLength):
    ''' Compute metrics of tower stability.

        These metrics are computed separately for each block in the tower. Only distanceFromTowerCenter
        applies to the bottom block (array length N), but the remaining measures ignore the bottom block (N-1).

        Later you can square, mean, max these values to see which is the best predictor of human behavior
        under different conditions.

        distanceFromTowerCenter: distance of each block from the tower's center-of-mass
        distanceFromBottomBlock: distance of each block from the bottom block
        percentSupported: percentage of a block that rests on the block below
        centroidEdgeDistance: position of centroid of block+above relative to the edge of the block below

        distanceFromTowerCenter and distanceFromBottomBlock are measures of the overall "wonkiness" of
        the tower, and they will deviate from each other if you square the distances (to emphasize large distances).

        percentSupported is a "local" measurement that indicates how stable a block is if you ignore
        the blocks above it.

        centroidEdgeDistance is a continuous measure of the physics rule "a block will fall IFF the
        centroid of it and the blocks above are over the edge of the block below". So here we just provide a
        continous measure of "how far over the edge" the centroid is for each block.

    '''



    numBlocks = len(pos)

    if isinstance(pos[0], (list,)): pos = [DotDict({"x": p[0], "y": p[1], "z": p[2]}) for p in pos]
    xs = np.array([p.x for p in pos])
    towerCenter = xs.mean()

    metrics = DotDict()

    metrics.distanceFromTowerCenter = [p.x-towerCenter for p in pos]
    metrics.distanceFromBottomBlock = [p.x-pos[0].x for p in pos[1:]]

    percentSupported = []
    centroidEdgeDistance = []

    for block in range(1,numBlocks):
        below = pos[block-1].x
        bounds = [below - sideLength/2, below + sideLength/2]
        rest = np.array([p.x for p in pos[block:]])
        center = rest.mean()

        percentSupported.append(compute_percent_supported(pos[block].x, below, sideLength, sideLength))
        centroidEdgeDistance.append(compute_centroid_edge_dist(center, bounds))

    metrics.percentSupported = percentSupported
    metrics.centroidEdgeDistance = centroidEdgeDistance

    metrics.unstable, metrics.isUnstable = compute_will_fall(pos, sideLength=sideLength)
    metrics.numUnstable = np.sum(metrics.isUnstable)
    metrics.pctUnstable = metrics.numUnstable/(numBlocks-1)

    # summary stats
    metrics.maxDistanceFromTowerCenter = np.max(np.abs(metrics['distanceFromTowerCenter']))
    metrics.meanDistanceFromTowerCenter = np.mean(np.abs(metrics['distanceFromTowerCenter']))

    metrics.maxDistanceFromBottomBlock = np.max(np.abs(metrics['distanceFromBottomBlock']))
    metrics.meanDistanceFromBottomBlock = np.mean(np.abs(metrics['distanceFromBottomBlock']))

    # taking min here because "less supported" means "more likely to fall"
    metrics.minPercentSupported = np.min(metrics['percentSupported'])
    metrics.meanPercentSupported = np.mean(metrics['percentSupported'])

    # not taking abs here because sign matters: neg values mean "wont fall", pos mean "will fall"
    metrics.maxCentroidEdgeDistance = np.max(metrics['centroidEdgeDistance'])
    metrics.meanCentroidEdgeDistance = np.mean(metrics['centroidEdgeDistance'])

    metrics.gt_class = 'unstable' if metrics.unstable else 'stable'
    metrics.correctRequiresMax = 1 if ((metrics.maxCentroidEdgeDistance > 0) and (metrics.meanCentroidEdgeDistance <= 0)) else 0

    return metrics

def compute_percent_supported(topX,bottomX,topLength,bottomLength):
    ''' Compute percent of top block that rests on top of the bottom block.

        Works for blocks of different sizes.

    '''
    topLeft = (topX - topLength/2.0)
    topRight = (topX + topLength/2.0)
    bottomLeft = (bottomX - bottomLength/2.0)
    bottomRight = (bottomX + bottomLength/2.0)

    left_over = np.min([0, (topLeft-bottomLeft)])
    right_over = np.max([0, (topRight-bottomRight)])

    overlap = topLength - np.abs(left_over) - np.abs(right_over)
    percent_supported = np.max([0,overlap]) / topLength

    return percent_supported

def compute_centroid_edge_dist(center, bounds):
    ''' Compute how far the centroid position is from the edge of the bottom block.

        @params
        center: center-of-mass of block+blocks above
        bounds: array, coordinates of the [left, right] edge of the block below

        returns distance

        Values can be interpreted as "distance over the edge". Negative values indicate
        the block is stable (centroid within the bottom block). Positive values indicate
        unstable blocks (centroid over the edge of the bottom block).
    '''
    if center < bounds[0]:
        d = bounds[0] - center
    elif center > bounds[1]:
        d = center - bounds[1]
    else:
        edge_distances = np.abs([center-bounds[0], center-bounds[1]])
        d = -np.min(edge_distances) # negative signifies "stable" (distance over the edge is negagive)

    return d

def format_trajectory(b):
    return (b.getPosition(), b.getRotation())

def format_trajectory_new(frame_no, frame_time, b):
    position = b.getPosition()
    lx,ly,lz = b.boxsize
    vx,vy,vz = b.getLinearVel()
    # vx,vy,vz = b.getPointVel(position)
    rotation = b.getRotation()

    # looks like my rx,ry,rz matches b.getQuaternion * 2
    # rotation2 = [a*2 for a in b.getQuaternion()][1:]
    # print(rotation2)
    density = b.getMass().mass / lx*ly*lz

    rx = math.atan2(rotation[7], rotation[8])
    ry = math.atan2(-rotation[6], math.sqrt(rotation[7]*rotation[7] + rotation[8]*rotation[8]));
    rz = math.atan2(rotation[3], rotation[0]);

    return DotDict(
            ('frame_no', frame_no), ('frame_time', frame_time),
            ('x', position[0]), ('y', position[1]), ('z', position[2]),
            ('lx', lx), ('ly', ly), ('lz', lz),
            ('vx', vx), ('vy', vy), ('vz', vz),
            ('rx', rx), ('ry', ry), ('rz', rz),
            ('rotation', rotation), ('quaternion', b.getQuaternion()),
            ('density', density)
           );


# --------------------------------------------------------
# Cube Tower Shape Helpers
# --------------------------------------------------------

def classify_tower_shape(positions, bins, shift=0):
  '''classify tower shape

      A coarse classification of the overall shape of a tower
      (for stacks of cubes).

      Given N blocks (block0, block1, ...blockN), for each block 1-N,
      check whether the block is left of center (-1), directly above (0),
      or right of center (+1). The shape of a tower is then the sequence of
      bins from bottom to top,
      e.g., 0,1,1,1 might look like:

        1    []
        1   []
        1  []
        0 []
          []

     You can adjust the bin sizes to give more precision in the shape
     description, but this is the basic idea.

     If you just have 3 bins, then the set of possible shapes is basically
     a tree structure, and the number of possible shape categories is
     3 ^ (N-1), e.g., for 5 blocks there are 81 possible tower shapes.
  '''
  num_positions = len(positions)
  offsets = []
  for idx in range(1,num_positions):
    p1 = positions[idx-1]
    p2 = positions[idx]
    dx = p2.x - p1.x
    offsets.append(dx)
  return np.digitize(np.array(offsets), bins) - shift

def classify_tower_shape_coarse(positions, sideLength=.2):
  '''
    preset that treats the middle 1/4 as "0", left -1, right +1
  '''
  bins = classify_tower_shape(positions, [-sideLength, -sideLength/8, sideLength/8, sideLength], shift=2)
  return "_".join([str(i) for i in bins])

def classify_tower_shape_fine(positions, sideLength=.2):
  '''
    5 bins (-2, -1, 0, 1, 2), with middle bin taking 1/4,
    +/-1 bins taking the next 1/4, and the +/-2 bins taking
    the final 1/8 per side.
  '''
  bins = classify_tower_shape(positions, [-sideLength, -sideLength/8*3, -sideLength/8, sideLength/8, sideLength/8*3, sideLength], shift=3)
  return "_".join([str(i) for i in bins])

def get_all_possible_shapes(stack_height, bins=['-1', '0', '1']):
  shapes = []
  if stack_height == 3:
    for b1 in bins:
      for b2 in bins:
        shapes.append(f'{b1}_{b2}')
  elif stack_height == 4:
    for b1 in bins:
      for b2 in bins:
        for b3 in bins:
          shapes.append(f'{b1}_{b2}_{b3}')
  elif stack_height == 5:
    for b1 in bins:
      for b2 in bins:
        for b3 in bins:
          for b4 in bins:
            shapes.append(f'{b1}_{b2}_{b3}_{b4}')
  elif stack_height == 6:
    for b1 in bins:
      for b2 in bins:
        for b3 in bins:
          for b4 in bins:
            for b5 in bins:
              shapes.append(f'{b1}_{b2}_{b3}_{b4}_{b5}')
  elif stack_height == 7:
    for b1 in bins:
      for b2 in bins:
        for b3 in bins:
          for b4 in bins:
            for b5 in bins:
              for b6 in bins:
                shapes.append(f'{b1}_{b2}_{b3}_{b4}_{b5}_{b6}')

  return shapes