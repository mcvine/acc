#!/usr/bin/env python

"""
Converted from Garrett's SNS_source mcstas component
"""

import os
import math, numpy as np, numba as nb
from numba import jit
import scipy.integrate as si
import scipy.interpolate as sinterp

import mcvine, mcvine.components
from mcni.AbstractComponent import AbstractComponent
from mcni.utils import conversion
from mcni import neutron_buffer, neutron

class SNS_source(AbstractComponent):

    def __init__(
            self, name, datapath, Emin, Emax, xwidth, yheight,
            dist, focus_xw, focus_yh,
            Anorm = 0.01,
            radius=None
    ):
        self.name = name
        self.Emin, self.Emax = Emin, Emax
        from ._SNS_source_utils import init
        self.INorm2, Es, Pvec, ts, Ptmat, EPmin, EPmax, Eidx_range, tidx_range = init(
            Emin, Emax, datapath)
        Eidx_start, Eidx_stop = Eidx_range
        tidx_start, tidx_stop = tidx_range
        self._Et_generation_args = (
            EPmin, EPmax, Es, Eidx_start, Eidx_stop, Pvec, ts, tidx_start, tidx_stop, Ptmat
            )
        self.xwidth, self.yheight, self.radius = xwidth, yheight, radius
        if xwidth <= 0 or yheight <= 0:
            warnings.warn("Either xwidth or yheight is invalid. will assume a circle")
            self.square = False
            assert radius > 0, "radius is invalid: {}".format(radius)
            srcArea = np.pi*radius*radius
        else:
            self.square = True
            srcArea = xwidth * yheight
        self.p_in = focus_xw*focus_yh/dist/dist * srcArea/Anorm
        self.dist, self.focus_xw, self.focus_yh = dist, focus_xw, focus_yh
        self.Anorm = Anorm
        return

    def process(self, neutrons):
        if not len(neutrons):
            return
        from mcni.neutron_storage import neutrons_from_npyarr, ndblsperneutron # number of doubles per neutrons thats means each neutron is represented by x, y, z, vx, vy, vz, s1, s2, t, t0, p (10 double variables)
        N = len(neutrons)
        arr = np.zeros((N, ndblsperneutron))
        x,y,z,vx,vy,vz,s1,s2,t,p = arr.T
        if (self.square == 1) :
            x[:] = self.xwidth * (np.random.rand(N) - 0.5);
            y[:] = self.yheight * (np.random.rand(N) - 0.5);
        else :
            chi=2*np.pi*np.random.rand(N)               #/* Choose point on source */
            r=sqrt(np.random.rand(N))*radius            # /* with uniform distribution. */
            x[:]=r*np.cos(chi)
            y[:]=r*np.sin(chi)
        z[:] = 0
        atan = np.arctan
        dist = self.dist
        hdivmax=atan((self.focus_xw/2.0-x)/dist);
        hdivmin=atan(-(self.focus_xw/2.0+x)/dist);
        vdivmax=atan((self.focus_yh/2.0-y)/dist);
        vdivmin=atan(-(self.focus_yh/2.0+y)/dist);

        theta = hdivmin + (hdivmax-hdivmin)*np.random.rand(N) # /* Small angle approx. */
        phi = vdivmin + (vdivmax-vdivmin)*np.random.rand(N)
        hdiv=theta
        vdiv=phi

        E, t[:] = generate(N, *self._Et_generation_args)
        v = conversion.SE2V*np.sqrt(E)
        # /* Calculate components of velocity vector such that the neutron is within the focusing rectangle */
        cos = np.cos; sin = np.sin
        vz[:] = v*cos(phi)*cos(theta);   #/* Small angle approx. */
        vy[:] = v*sin(phi);
        vx[:] = v*cos(phi)*sin(theta);
        p[:] = self.p_in*self.INorm2
        neutrons.from_npyarr(arr)
        return neutrons

# ## Interpolation
#  1d quadratic interpolation 
#  given 2 points on the low side of xdes and one on the high side, return
#  a quadratically interpolated result 
@jit("double(double, double, double, double, double, double, double)", nopython=True)
def quadint(xdes, x1,  x2, x3, y1, y2, y3):
    t1=((xdes-x2)*(xdes-x3)*y1)/((x1-x2)*(x1-x3))
    t2=((xdes-x1)*(xdes-x3)*y2)/((x2-x1)*(x2-x3))
    t3=((xdes-x1)*(xdes-x2)*y3)/((x3-x1)*(x3-x2))
    return t1+t2+t3

@jit("double(double, double, double[:], double[:])", nopython=True)
def quadfuncint(xdes,  xylen, vecx, vecy):
    idx=1
    while (vecx[idx]<xdes) and idx<xylen:
        idx+=1
    if (idx>xylen):
        print("error exceeded vector length")

    if (vecx[idx]==xdes):
        return vecy[idx];
    else:
        return quadint(xdes,vecx[idx-2],vecx[idx-1],vecx[idx],vecy[idx-2],vecy[idx-1],vecy[idx])

# ## From cumulative probability find x (E or t)
@jit("double(double[:], double[:], double, double, double, double)", nopython=True)
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


def test_zero_find():
    vecx = np.arange(0.02, 1., 0.02)
    vecy = vecx*vecx
    quadfuncint(0.51, len(vecx), vecx, vecy)
    zero_find(vecx, vecy, 0.04, xmin=0.02, xmax=1., tol=1e-6)
    return


# # Neutron generation function
@jit('''Tuple((float64[:], float64[:]))(
int32, float64, float64, float64[:], int32, int32, float64[:], float64[:],
int32, int32, float64[:,:])''', nopython=True)
def generate(N, EPmin, EPmax, Es, Eidx_start, Eidx_stop, Pvec, ts, tidx_start, tidx_stop, Ptmat):
    Eout = np.zeros(N, dtype=np.dtype('float64'))
    tout = np.zeros(N, dtype=np.dtype('float64'))
    for i in range(N):
        # /*First generate E random value */
        randp=EPmin+np.random.rand()*(EPmax-EPmin)
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
        Pt = Ptmat[Eidxl]*wl + Ptmat[Eidxh]*(1-wl)
        # generate t prob
        randp=Pt[tidx_start]+np.random.rand()*(Pt[tidx_stop-1]-Pt[tidx_start])
        if (randp>0.0):
            tval=zero_find(ts, Pt, randp, ts[tidx_start],ts[tidx_stop-1],1e-5)
        else:
            tval=0
        Eout[i] = Eval*1000.0; # /* Convert Energy from Ev to meV */
        tout[i] = tval*1e-6;   #   /* Convert time from mus to S */
    return Eout,tout
