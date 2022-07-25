import numpy as np
from numba import cuda, void, int64
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_type
from math import sqrt, pi, sin, cos, asin, radians, isnan
from mcni.utils.conversion import V2K, K2V
from .. import vec3, neutron as nu
DEG2RAD = radians(1)
epsilon = 1e-7

@cuda.jit(device=True)
def scattering_coefficient(neutron, unitcell_volume, Npeaks, q_v, my_s_v2):
    xs = scattering_xs(neutron, Npeaks, q_v, my_s_v2)
    return xs/unitcell_volume

@cuda.jit(device=True)
def scattering_xs(neutron, Npeaks, q_v, my_s_v2):
    v = neutron[3:6]
    v_l = vec3.length(v)
    xs_v2 = 0.0;
    for i in range(Npeaks):
        # find out the one to be diffracted
        if v_l >= q_v[i]/2:
            xs_v2 += my_s_v2[i];
    return xs_v2 / (v_l*v_l) # devided by v**2 at this step

@cuda.jit(device=True)
def scatter(
        threadindex, rng_states, neutron,
        w_v, q_v, my_s_v2, Npeaks,
        d_phi, n, vtmp, vout, unitcell_volume,
):
    v = neutron[3:6]
    vl = vec3.length(v)
    if vl<q_v[0]/2:
        nu.absorb(neutron)
        return
    for peakindex in range(1, Npeaks):
        if vl<q_v[peakindex]/2: break
    Npeaks = peakindex
    if Npeaks > 1:
        # randomly pick peak
        peakindex = int(xoroshiro128p_uniform_float32(rng_states, threadindex)*Npeaks)
    else:
        peakindex = 0
    #
    # random in [-1, 1)
    rt = xoroshiro128p_uniform_float32(rng_states, threadindex)*2-1
    arg = q_v[peakindex]*(1+w_v[peakindex]*rt)/(2.0*vl)

    # be very careful here
    scatter_xs = my_s_v2[peakindex]/(vl*vl); # this is the cross section for this peak
    scatter_xs *=Npeaks; # from randomly choosing one peak
    theta = asin(arg);

    # Choose point on Debye-Scherrer cone
    if d_phi:
        #  relate height of detector to the height on DS cone
        arg = sin(d_phi*DEG2RAD/2)/sin(2*theta);
        # If full Debye-Scherrer cone is within d_phi, don't focus
        if (arg < -1 or arg > 1):
            d_phi = 0
        else:  # Otherwise, determine alpha to rotate from scattering plane into d_phi focusing area
            alpha = 2*asin(arg)
    if d_phi:
        # Focusing
        alpha = abs(alpha);
        # Trick to get scattering for pos/neg theta's 
        rt = xoroshiro128p_uniform_float32(rng_states, threadindex)
        alpha0= 2*rt*alpha
        if (alpha0 > alpha):
            alpha0=pi+(alpha0-1.5*alpha);
        else:
            alpha0=alpha0-0.5*alpha;
    else:
        rt = xoroshiro128p_uniform_float32(rng_states, threadindex)*2-1
        alpha0 = pi*rt

    #  now find a nearly vertical rotation axis:
    #  Either
    #   (v along Z) x (X axis) -> nearly Y axis
    #  Or
    #   (v along X) x (Z axis) -> nearly Y axis
    #
    # ex dot v = vx; ez dot v = vz
    if ( abs( v[0] ) < abs( v[2] ) ):
        n[0] = 1; n[1] = 0; n[2] = 0;
    else:
        n[0] = 0; n[1] = 0; n[2] = 1;
    vec3.cross(v, n, vtmp)

    # vout is incident v rotated by 2*theta around tmp_v
    vec3.copy(v, vout)
    vec3.rotate(vout, vtmp, 2*theta, epsilon)

    # rotate vout by alpha0 around incident direction (Debye-Scherrer cone)
    vec3.rotate(vout, v, alpha0, epsilon)

    # V_t vtest(vout);
    # printf("(Vx, Vy, Vz) = (%f, %f, %f)\n", vtest.x, vtest.y, vtest.z);

    # change event
    vec3.copy(vout, v)

    # XXX: Hack, in case the engine is not working rationally, for now let us ignore them
    if (isnan(vout[0]) or isnan(vout[1]) or isnan(vout[2])):
        nu.absorb(neutron)
        return
    neutron[-1] *= scatter_xs/unitcell_volume
    return


class PowderDiffraction:

    def __init__(
            self, peaks, unitcell_volume,
            absorption_cross_section, incoherent_cross_section, coherent_cross_section,
            pack=1.,
            # XsectionFactor = 1, if cross-section in fm^2, or XsectionFactor = 100, if cross-section in barns
            XsectionFactor=100.,
            d_phi = 0,
    ):
        Npeaks = len(peaks)
        q_v = np.zeros(Npeaks)
        w_v = np.zeros(Npeaks)
        my_s_v2 = np.zeros(Npeaks)
        for i in range(Npeaks):
            dw = peaks[i].DebyeWaller_factor
            dw = dw if dw else 1.0
            my_s_v2[i] = 4*pi*pi*pi*pack*dw/unitcell_volume/(V2K*V2K)
            my_s_v2[i] *= (peaks[i].multiplicity * peaks[i].F_squared / peaks[i].q) *XsectionFactor; 
            # unit is cross_section * v**2

            # Squires [3.103] 
            q_v[i] = peaks[i].q*K2V
            # to be updated for size broadening
            w_v[i] = peaks[i].intrinsic_line_width
            #  make sure the list is sorted
            if i>0: assert(q_v[i-1] <= q_v[i])
            continue

        # coherent cross section is calculated under scattering_coefficient function, as it depends on the incident neutron velocity.
        # total_scattering_cross_section = pack*( coherent_cross_section + data.incoherent_cross_section); // barn
        # total_scattering_coeff = total_scattering_cross_section/unitcell_volume * 100; // converted to 1/meter
        self.absorption_cross_section = pack * absorption_cross_section # barn
        self.absorption_coeff = absorption_cross_section/unitcell_volume * 100 # converted to 1/meter
        self.incoherent_cross_section = incoherent_cross_section;
        self.coherent_cross_section = coherent_cross_section;
        self.Npeaks = Npeaks
        self.q_v = q_v
        self.w_v = w_v
        self.my_s_v2 = my_s_v2
        self.d_phi = d_phi
        self.unitcell_volume = unitcell_volume
        return
