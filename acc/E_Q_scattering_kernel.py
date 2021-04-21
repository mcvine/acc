import math
import importlib
import numba
from numba import cuda
import numpy as np
from math import sqrt
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32


PI = 3.14159265
neutron_mass = 1.6749286e-27
mN = neutron_mass

electron_charge = 1.60217733e-19
e = electron_charge

boltzman_constant = 1.3806504e-23
kB = boltzman_constant

hbar = 1.054571628e-34
try:
    import user_input
except (ImportError, NameError):
    with open('user_input.py', 'w+') as f:
        f.write('def dis(Q):\n'+
                '   return None')
    import user_input

K2V = hbar/mN*1e10
V2K = 1./K2V
#V2K = 1.58801E-3 # Convert v[m/s] to k[1/AA]
SE2V = sqrt(2e-3*e/mN)
# SE2V = 437.3949	   #/* Convert sqrt(E)[meV] to v[m/s] */
VS2E = mN/(2e-3*e)
#VS2E = 5.227e-6	   #/* Convert (v[m/s])**2 to E[meV] */
RV2W = 2*PI*K2V            # Converts reverse v[m/s] to wavelength[AA]; w = RV2W*1/v
#RV2W = 3.95664E+3

@cuda.jit(device=True)
def v2k(vel):
    return V2K * vel
@cuda.jit(device=True)
def e2v(energy):
    return sqrt(energy)*SE2V
@cuda.jit(device=True)
def e2k(energy):
    return v2k( e2v( energy) )
@cuda.jit(device=True)
def k2v(k):
    return K2V * k
@cuda.jit(device=True)
def v2e(v):
    return v*v*VS2E
@cuda.jit(device=True)
def k2e(k):
    return v2e( k2v( k) )


def write_user_input(E_Q, S_Q, scattering_cross_section, absorption_cross_section, loc='.'):
    with open(f'{loc}/user_input.py', 'w+') as f:
        f.write('from numba import cuda\n\n')
        f.write('@cuda.jit(device=True)\ndef dis(Q):\n'+
                f'   return {E_Q.strip()}')
        f.write('\n\n')
        f.write('@cuda.jit(device=True)\ndef S_Q(Q):\n'+
                f'   return {S_Q}')
        f.write('\n\n')
        f.write('@cuda.jit(device=True)\ndef scattering_coefficient(neutron_velocity, neutron_spin):\n'+
                f'   return {scattering_cross_section}')
        f.write('\n\n')
        f.write('@cuda.jit(device=True)\ndef absorption_coefficient(neutron_velocity, neutron_spin):\n'+
                f'   return {absorption_cross_section}')
    importlib.reload(user_input)
@cuda.jit(device=True)
def my_disp(Q):
    return (user_input.dis(Q))
@cuda.jit(device=True)
def my_S(Q):
    return (user_input.S_Q(Q))

def _Q_check(Qmin, Qmax):
    if Qmin<0:
        raise ValueError("Qmin must not be negative")
    if Qmin>=Qmax:
        raise ValueError("Qmin must be smaller than Qmax")


@cuda.jit(device=True)
def random_gpu ( rng_states, min,max, thread_id):
    Q = (xoroshiro128p_uniform_float32(rng_states,thread_id)) * (max - min) + min
    return Q


@cuda.jit()
def E_Q_kernel (rng_states, Qmin, Qmax, neutron_velocity, neutron_probability, scattered_neutron_probability,
                scattered_neutron_velocity):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw

    if pos >= neutron_velocity.shape[0]:
        return
    m_epsilon = 1e-4
    vi = sqrt(neutron_velocity[pos,0] ** 2 + neutron_velocity[pos,1] ** 2 +
              neutron_velocity[pos,2] ** 2)
    Ei = v2e (vi)


    while (1):


        Q = random_gpu(rng_states,Qmin, Qmax, pos)
        E = my_disp(Q)
        Ef = Ei-E

        if Ef<0:
            continue

        vf = e2v(Ef)

        ki = v2k(vi)

        kf = v2k(vf)

        cost = (ki*ki + kf*kf -Q*Q)/(2*ki*kf)
        cost2 = cost*cost

        if cost2>1:
            continue
        break

    sint = math.sqrt(1-cost2)

    phi_uuper_range = math.pi*2

    phi = random_gpu(rng_states,0,phi_uuper_range, pos)

    e1= neutron_velocity[pos]

    # e1 = normalize(e1)
    e1_normalize = cuda.local.array(3, numba.float64)
    for i in range(3):
        e1_normalize[i] = np.nan


    for i in range(3):
        e1_normalize[i] = e1[i]*(1./ (sqrt(e1[0] ** 2 + e1[1] ** 2 +e1[2] ** 2)))


    e2_vector = cuda.local.array(3, numba.float64)
    for i in range(3):
        e2_vector[i] = np.nan

    e2_normalize = cuda.local.array(3, numba.float64)
    for i in range(3):
        e2_normalize[i] = np.nan

    unit_z_vector = [0.,0.,1.]

    if abs(neutron_velocity[0,pos])>m_epsilon or abs (neutron_velocity[1,pos])>m_epsilon:


        e2_vector[0] = (0. * e1_normalize[2]) -(1.*e1_normalize[1])
        e2_vector[1] = (1. * e1_normalize[0]) - (0.*e1_normalize[2])
        e2_vector[2] = (0. * e1_normalize[1]) - (0.*e1_normalize[0])

        for i in range(3):
            e2_normalize[i] = e2_vector[i] * (1. / (sqrt(e2_vector[0] ** 2 + e2_vector[1] ** 2 + e2_vector[2] ** 2)))


    else:
        e2_normalize[0] = 1.
        e2_normalize[1] =0.
        e2_normalize[2] = 0.

    e3 = cuda.local.array(3, numba.float64)
    for i in range(3):
        e3[i] = np.nan


    e3[0] = (e1_normalize[1]*e2_normalize[2]) - (e1_normalize[2]*e2_normalize[1])
    e3[1] = (e1_normalize[2]*e2_normalize[0])-(e1_normalize[0]*e2_normalize[2])
    e3[2] = (e1_normalize[0]*e2_normalize[1])-(e1_normalize[1]*e2_normalize[0])

    v_f_array = cuda.local.array(3, numba.float64)
    for i in range(3):
        v_f_array[i] = np.nan

    for i in range(3):
        v_f_array[i] = (e2_normalize[i]*sint*math.cos(phi)+e3[i]*sint*math.sin(phi)+
                        e1_normalize[i]*cost)*vf

    scattered_neutron_probability[pos] = neutron_probability[pos] * my_S(Q) * (vf/vi) * \
                                         Q*(Qmax-Qmin) /(kf*ki)/2

    scattered_neutron_velocity[pos,0] = v_f_array[0]
    scattered_neutron_velocity[pos, 1] = v_f_array[1]
    scattered_neutron_velocity[pos, 2] = v_f_array[2]

def E_Q_scattering_kernel_call(Qmin, Qmax, neutron_velocity, neutron_probability, E_Q, S_Q, scattering_coefficient,
                               absorption_cross_section,  threadsperblock = 64 ):
    neutron_velocity_gpu = cuda.to_device(neutron_velocity)
    neutron_probability_gpu = cuda.to_device(neutron_probability)
    scattered_neutron_velocity =cuda.device_array ((len(neutron_velocity),3)) #cuda.device_array_like(d_a)
    scattered_neutron_probability = cuda.device_array(len(neutron_probability))

    write_user_input(E_Q, S_Q, scattering_coefficient, absorption_cross_section)
    rng_states = create_xoroshiro128p_states(15000, seed=1)

    blockspergrid = (neutron_velocity_gpu.size + (threadsperblock - 1)) // threadsperblock
    E_Q_kernel[blockspergrid, threadsperblock] (rng_states,Qmin,Qmax, neutron_velocity_gpu, neutron_probability_gpu,
                    scattered_neutron_probability,
                    scattered_neutron_velocity)


    scattered_neutron_velocity_cpu = scattered_neutron_velocity.copy_to_host()
    scattered_neutron_probability_cpu = scattered_neutron_probability.copy_to_host()

    return scattered_neutron_probability_cpu, scattered_neutron_velocity_cpu





