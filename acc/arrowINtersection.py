from numba import cuda
import numpy as np

@cuda.jit
def intersectRectangle(rx, ry, rz,vx, vy, vz,X, Y, t):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < t.size:  # Check array boundaries
        t_pos = (0-rz[pos])/vz[pos]
        r1x = rx[pos] + vx[pos] * t_pos
        r1y = ry[pos] + vy[pos] * t_pos
        # print (pos, t_pos, r1x, r1y, t.size)
        if abs(r1x)<X[pos]/2 and abs(r1y)<Y[pos]/2:
            t[pos]=t_pos
        else:
            t[pos] = np.nan




# rx= np.array([0.,0.,0.])
# ry= np.array([0.,0,0])
# rz= np.array([1.,1,1])
# vx= np.array([1.,0,-5])
# vy= np.array([0.,0,0])
# vz= np.array([-1.,-1,-1])
# X=np.array([3.,0.5,3])
# Y=np.array([1.,1,1])
# t=np.zeros(rx.shape)

array_size =100000000
rx= np.zeros((array_size))
ry= np.zeros((array_size))
rz= np.ones((array_size))
vx= np.ones((array_size))
vy= np.zeros((array_size))
vz= -np.ones((array_size))
X=np.ones((array_size))*3
Y=np.ones((array_size))
t=np.zeros(rx.shape)

threadsperblock = 32
blockspergrid = (t.size + (threadsperblock - 1)) // threadsperblock
intersectRectangle[blockspergrid, threadsperblock](rx, ry, rz,vx, vy, vz,X, Y,t)

# print (t)
