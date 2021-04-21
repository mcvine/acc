from numba import cuda
import numpy as np
import math
import numba



@cuda.jit(device=True)
def oneNeutronINtersectRectangle(rx, ry, rz,vx, vy, vz,X, Y):
    t_pos = (0 - rz) / vz
    r1x = rx + vx * t_pos
    r1y = ry + vy * t_pos
    bad = abs(r1x) > X / 2 or abs(r1y) > Y / 2
    if bad:
        t_pos= np.nan
    return t_pos



@cuda.jit()
def intersectBox(arrow_array_position, arrow_array_direction, box, t):
    # neutron_times = []
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw

    # calculate for each pos  (index of the neutrons) intersection which of the phases of one box
    t[pos, 0] = np.nan
    t[pos,1] = np.nan
    if pos>arrow_array_direction.shape[0]:
        return
    start = arrow_array_position[pos]
    direction = arrow_array_direction[pos]
    x = start[0]
    y = start[1]
    z = start[2]
    vx = direction[0]
    vy = direction[1]
    vz = direction[2]


    X = box[0]
    Y = box[1]
    Z = box[2]
    # t_store = np.empty(shape=len(t)+3)
    t_cal = cuda.local.array(6, numba.float64)
    for i in range(6):
        t_cal[i] = np.nan
    if vz != 0:
        t_cal[0] = oneNeutronINtersectRectangle(x, y, z-Z / 2, vx, vy, vz, X, Y)
        t_cal[1] = oneNeutronINtersectRectangle(x, y, z+Z / 2, vx, vy, vz, X, Y)



    if vx != 0:
        t_cal[2] = oneNeutronINtersectRectangle(y, z, x-X / 2, vy, vz, vx, Y, Z)
        t_cal[3] = oneNeutronINtersectRectangle(y, z, x+X / 2, vy, vz, vx, Y, Z)


    if vy != 0:
        t_cal[4] =oneNeutronINtersectRectangle(z, x, y-Y / 2, vz, vx, vy, Z, X)
        t_cal[5] =oneNeutronINtersectRectangle(z, x, y+Y / 2, vz, vx, vy, Z, X)

    for i in range(6):
        if not math.isnan(t_cal[i]):
            if math.isnan (t[pos,0]):
                t[pos,0] = t_cal[i]
            else:
                t[pos,1] = t_cal[i]

    if t[pos,0] > t[pos,1]:
        t[pos,0], t[pos,1] = t[pos,1], t[pos,0]



def ArrowIntersector(arrow_array_position, arrow_array_direction, box, threadsperblock = 64):
    '''

    :param box: NamedTuple :
    :param m_arrow:
    :param threadsperblock:
    :return:
    '''

    arrow_array_position_gpu = cuda.to_device(arrow_array_position)

    arrow_array_direction_gpu = cuda.to_device(arrow_array_direction)

    box_gpu = cuda.to_device(box)
    t_gpu = cuda.device_array((len(arrow_array_position),2))

    blockspergrid = (t_gpu.size + (threadsperblock - 1)) // threadsperblock
    intersectBox[blockspergrid, threadsperblock](arrow_array_position_gpu, arrow_array_direction_gpu,
                                                 box_gpu, t_gpu)
    t = t_gpu.copy_to_host()
    return t

