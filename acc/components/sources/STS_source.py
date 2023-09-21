#!/usr/bin/env python

"""
Converted from Garrett's STS_source mcstas component
"""

import math, numpy as np, numba as nb
import warnings
from numba import cuda, void, int32, int64, boolean
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_type

from .SourceBase import SourceBase
from ...config import get_numba_floattype
NB_FLOAT = get_numba_floattype()
from mcni.utils import conversion

Pt_tmp_arr_size = 256
SE2V = conversion.SE2V

# # Neutron generation function
@cuda.jit(device=True)
def generate(
        EPmin, EPmax, Es, Eidx_start, Eidx_stop,
        Pvec, ts, tidx_start, tidx_stop, Ptmat,
        randE, randt
):
    """
    float64, float64, float64[:], int32, int32,
    float64[:], float64[:], int32, int32, float64[:,:]
    randE, randt: two random numbers between 0,1 generated outside this method
    """
    # /*First generate E random value */
    randp=EPmin+randE*(EPmax-EPmin)
    # /* find E value corresponding to random probability */
    Eval=zero_find(Es, Pvec,randp,Es[Eidx_start],Es[Eidx_stop],1e-7)
    # /* from a known E value generate an emission time value */
    # /* find the index of the E values that bracket the random E value */
    idx1=0
    nEvals = len(Es)
    while((idx1<nEvals)and(Es[idx1]<Eval)):
        idx1+=1
    Eidxh=idx1;
    Eidxl=idx1-1;
    # interpolate along E to get P(t) curve
    wl = (Es[Eidxh]-Eval)/(Es[Eidxh]-Es[Eidxl])
    nt = len(ts)
    # Pt = Ptmat[Eidxl]*wl + Ptmat[Eidxh]*(1-wl)
    Pt = cuda.local.array(shape=Pt_tmp_arr_size, dtype=nb.float64)
    for it in range(nt):
        Pt[it] = Ptmat[Eidxl, it]*wl + Ptmat[Eidxh, it]*(1-wl)
    # generate t prob
    randp=Pt[tidx_start]+randt*(Pt[tidx_stop-1]-Pt[tidx_start])
    if (randp>0.0):
        tval=zero_find(ts, Pt, randp, ts[tidx_start],ts[tidx_stop-1],1e-5)
    else:
        tval=0
    Eout = Eval*1000.0 # /* Convert Energy from Ev to meV */
    tout = tval*1e-6   #   /* Convert time from mus to S */
    return Eout, tout


# ## Interpolation
#  1d quadratic interpolation
#  given 2 points on the low side of xdes and one on the high side, return
#  a quadratically interpolated result
@cuda.jit(device=True, inline=True)
def quadint(xdes, x1,  x2, x3, y1, y2, y3):
    t1=((xdes-x2)*(xdes-x3)*y1)/((x1-x2)*(x1-x3))
    t2=((xdes-x1)*(xdes-x3)*y2)/((x2-x1)*(x2-x3))
    t3=((xdes-x1)*(xdes-x2)*y3)/((x3-x1)*(x3-x2))
    return t1+t2+t3

@cuda.jit(device=True, inline=True)
def quadfuncint(xdes,  xylen, vecx, vecy):
    idx=1
    while (vecx[idx]<xdes) and idx<xylen:
        idx+=1

    if (vecx[idx]==xdes):
        return vecy[idx];
    else:
        return quadint(
            xdes,vecx[idx-2],vecx[idx-1],vecx[idx],vecy[idx-2],vecy[idx-1],vecy[idx])

# ## From cumulative probability find x (E or t)
@cuda.jit(device=True)
def zero_find(vecx, vecy, yval, xmin, xmax, tol):
    func = lambda x,y: quadfuncint(x, len(vecx), vecx, vecy) - y
    xl=xmin
    xh=math.pow(10,(math.log10(xmin)+yval*(math.log10(xmax)-math.log10(xmin))))
    f=func(xl,yval)
    fmid=func(xh,yval)
    while (fmid*f>=0.0):
        xh=xh+(xh-xl)*2.0
        fmid=func(xh,yval)
    dx=xh-xl
    rtb=xl
    while(math.fabs(func(rtb,yval))>tol):
        dx=dx*0.5
        xmid=rtb+dx
        fmid=func(xmid,yval)
        if (fmid<0):
            rtb=xmid
    return rtb

class STS_source(SourceBase):

    def __init__(
            self, name, filename, Emin, Emax, xwidth, yheight,
            dist, focus_xw, focus_yh,
            PPower = 0.0, # [kW] facility proton beam power
            frequency = 0.0, # [Hz] repetition rate
            MOffangle = -1000.0, # [deg] flight angle with regard to emission port location
            dmode = 1.0, # 0 = flat-area distribution, 1 = flat-projection distribution (relevant only for cylindrical moderator)
            radius = 0.0 # optional radius for disk shaped moderator port
    ):
        self.name = name
        self.Emin, self.Emax = Emin, Emax
        from ._SNS_source_utils import init
        self.INorm2, Es, Pvec, ts, Ptmat, EPmin, EPmax, Eidx_range, tidx_range, STS_params = init(
            Emin, Emax, filename, sts_load=True)
        assert len(ts) < Pt_tmp_arr_size
        Eidx_start, Eidx_stop = Eidx_range
        tidx_start, tidx_stop = tidx_range
        self._Et_generation_args = (
            EPmin, EPmax, Es, Eidx_start, Eidx_stop, Pvec, ts, tidx_start, tidx_stop, Ptmat
            )
        self.xwidth, self.yheight, self.radius = xwidth, yheight, radius
        self.PPower, self.frequency = PPower, frequency

        # Params = Power0, Freq0, MRadius0, MOffangle0, PHeight, PWidth, TRadius
        self.PPower0, self.Freq0, self.MRadius0, MOffangle0, PHeight, PWidth, TRadius = STS_params
        self.MOffangle = MOffangle0
        
        if MOffangle != MOffangle0 and MOffangle > -1000.0:
            self.MOffangle = MOffangle0
            warnings.warn("Requested MOffangle different from value in source file")

        self.MOffangle_rad = math.pi / 180.0 * self.MOffangle

        Anorm = 1.0
        self.tanMOa = 1.0
        
        if TRadius > 0.0:
            Anorm = np.pi * TRadius * TRadius
            if self.radius > TRadius or self.radius <= 0.0:
                self.radius = TRadius
                warnings.warn("radius not set or requested is larger than TRadius. Setting radius=TRadius.")

            self.square = False
            srcArea = np.pi * self.radius * self.radius

            print(" Anorm = {}".format(Anorm))
            print(" srcArea = {}".format(srcArea))

            self.tanMOa = math.tan(self.MOffangle_rad)
            if self.xwidth > 0.0 or self.yheight > 0.0:
                warnings.warn("Requested square source for cylindrical moderator: set disk source shape!")
        else:
            Anorm = PWidth * PHeight
            if self.xwidth > PWidth or self.xwidth <= 0.0:
                self.xwidth = PWidth
                warnings.warn("xwidth not set or requested larger than PWidth. Setting xwidth=PWidth")
            if self.yheight > PHeight or self.yheight <= 0.0:
                self.yheight = PHeight
                warnings.warn("yheight not set or requested larger than PHeight. Setting yheight=PHeight")
            
            if self.radius > 0.0:
                warnings.warn("Requested disk source for cylindrical moderator: set square source shape!")

            self.square = True
            srcArea = self.xwidth * self.yheight

            self.cylz = -self.MRadius0 * math.cos(self.MOffangle_rad)
            self.cylx = -self.MRadius0 * math.sin(self.MOffangle_rad)

            print(" Anorm = {}".format(Anorm))
            print(" srcArea = {}".format(srcArea))
            print(" cylz = {}".format(self.cylz))
            print(" cylx = {}".format(self.cylx))

            arg = math.sqrt(self.MRadius0 * self.MRadius0 - (self.xwidth / 2.0 - self.cylx)**2) + self.cylz
            self.cylphi1 = 2.0 * math.asin( math.sqrt(arg*arg + 0.25 * self.xwidth * self.xwidth) / 2.0 / self.MRadius0)
            arg = math.sqrt(self.MRadius0 * self.MRadius0 - (-self.xwidth / 2.0 - self.cylx)**2) + self.cylz
            self.cylphi2 = 2.0 * math.asin( math.sqrt(arg*arg + 0.25 * self.xwidth * self.xwidth) / 2.0 / self.MRadius0)

            print(" cylphi1 = {}".format(self.cylphi1))
            print(" cylphi2 = {}".format(self.cylphi2))


        # Calculate solid angle
        self.p_in = focus_xw * focus_yh / (dist * dist)
        self.p_in *= srcArea / Anorm
        
        if self.PPower > 0.0:
            self.p_in *= self.PPower / self.PPower0
        if self.frequency > 0.0:
            self.p_in *= self.Freq0 / self.frequency

        self.dmode = True if dmode == 1.0 else False
        
        
        self._prob = self.p_in * self.INorm2
        self.dist, self.focus_xw, self.focus_yh = dist, focus_xw, focus_yh
        self.Anorm = Anorm
        self.propagate_params = (
            self.square, xwidth, yheight, radius,
            focus_xw, focus_yh, dist,
            EPmin, EPmax, Es, Eidx_start, Eidx_stop,
            Pvec, ts, tidx_start, tidx_stop, Ptmat,
            self._prob, self.tanMOa, self.cylx, self.cylz, 
            self.cylphi1, self.cylphi2, self.MRadius0, 
            self.MOffangle_rad, self.dmode
        )
        import mcni
        neutrons = mcni.neutron_buffer(1)
        self.process(neutrons)
        return

    @cuda.jit(void(
        int32, xoroshiro128p_type[:],
        NB_FLOAT[:],
        boolean, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT,
        NB_FLOAT, NB_FLOAT, NB_FLOAT[:], int32, int32,
        NB_FLOAT[:], NB_FLOAT[:], int32, int32, NB_FLOAT[:, :],
        NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, boolean
    ), device=True)
    def propagate(
            threadindex, rng_states,
            in_neutron,
            square, xwidth, yheight, radius, focus_xw, focus_yh, dist,
            EPmin, EPmax, Es, Eidx_start, Eidx_stop,
            Pvec, ts, tidx_start, tidx_stop, Ptmat,
            prob, tanMOa, cylx, cylz, cylphi1, cylphi2, MRadius0, MOffangle_rad, dmode
    ):
        randx = xoroshiro128p_uniform_float32(rng_states, threadindex)
        randy = xoroshiro128p_uniform_float32(rng_states, threadindex)
        rand_theta = xoroshiro128p_uniform_float32(rng_states, threadindex)
        rand_phi = xoroshiro128p_uniform_float32(rng_states, threadindex)
        randE = xoroshiro128p_uniform_float32(rng_states, threadindex)
        randt = xoroshiro128p_uniform_float32(rng_states, threadindex)
        
        z = 0.
        if square:
            y = yheight * (randy - 0.5)
            if dmode:
                # constant projection areal density
                x = xwidth * (randx - 0.5)
                z = math.sqrt( MRadius0 * MRadius0 - (x - cylx)**2) + cylz
            else:
                # constant areal density on port face
                ang = -cylphi1 + randx*(cylphi1+cylphi2) + MOffangle_rad
                z = cylz + MRadius0 * math.cos(ang)
                x = cylx + MRadius0 * math.sin(ang)
        else:
            chi = 2.0 * math.pi * randx            # Choose point on disk source
            r = math.sqrt(randy) * radius          # with uniform distribution.
            x = r * math.cos(chi)
            y = r * math.sin(chi)
            z = tanMOa * x

        hdivmax=math.atan((focus_xw/2.0-x)/dist);
        hdivmin=math.atan(-(focus_xw/2.0+x)/dist);
        vdivmax=math.atan((focus_yh/2.0-y)/dist);
        vdivmin=math.atan(-(focus_yh/2.0+y)/dist);

        theta = hdivmin + (hdivmax-hdivmin)*rand_theta # /* Small angle approx. */
        phi = vdivmin + (vdivmax-vdivmin)*rand_phi

        E, t = generate(
            EPmin, EPmax, Es, Eidx_start, Eidx_stop,
            Pvec, ts, tidx_start, tidx_stop, Ptmat,
            randE, randt)
        v = SE2V*math.sqrt(E)
        # /* Calculate components of velocity vector such that the neutron is within the focusing rectangle */
        cos = math.cos; sin = math.sin
        vz = v*cos(phi)*cos(theta);   #/* Small angle approx. */
        vy = v*sin(phi);
        vx = v*cos(phi)*sin(theta);
        in_neutron[:6] = x,y,z, vx,vy,vz
        in_neutron[-2] = t
        in_neutron[-1] = prob
        return

