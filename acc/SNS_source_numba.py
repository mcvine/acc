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

class SNS_source_numba(AbstractComponent):

    def __init__(
            self, name, datapath, Emin, Emax, xwidth, yheight,
            dist, focus_xw, focus_yh,
            Anorm = 0.01,
            radius=None
    ):
        self.name = name
        self.Emin, self.Emax = Emin, Emax
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

# ## Load data
def sns_source_load(filename, xcol=0, ycol=2):
    with open(filename, 'rt') as stream:
        first_block = []
        while True:
            line = stream.readline()
            if line.startswith('#'): 
                continue
            if len(line.strip())!=0:
                d = list(map(float, line.split()))
                first_block.append(d)
            else:
                break
        first_block = np.array(first_block)
        xvec = first_block[:, xcol]
        yvec = first_block[:, ycol]
        idx1 = first_block.shape[0]
        idx2 = idx1//2
        while((idx2<idx1) and (yvec[idx2]>0)):
            idx2+=1
        if(idx2<idx1):
            veclen=idx2
        else:
            veclen=idx1-2
        # skip over to section2
        while True:
            line = stream.readline()
            if not (line.startswith('#') or len(line.strip())==0):
                break
        mat = []
        while len(line):
            # print(len(mat))
            s2block = []
            while len(line.strip())!=0:
                d = [float(t) for t in line.split()]
                s2block.append(d)
                line = stream.readline()
            mat.append(s2block)
            # skip to next subsection
            while len(line) and len(line.strip())==0:
                line = stream.readline()
                continue
        mat = np.array(mat)
    nE, nt, ncols = mat.shape
    assert nE == len(xvec)
    for i in range(nt):
        assert np.allclose(mat[:, i, 1], xvec)
    tvec = mat[0, :, 0]
    for i in range(nE):
        assert np.allclose(mat[i, :, 0], tvec)
    return xvec, yvec, veclen, tvec, mat[:, :, 2]


# ## Compute integrated probability
def Pcalc(func, llim,  hlim, xvec, Prob, veclen):
    """ calculate integrated probability

    Parameters
    ----------
    func: function
        I(x) function
    llim, hlim: floats
        lower and higher limits of x
    xvec: array
        x array
    Prob: array
        output cumulative probability
    veclen: int
        length of xvec. should be same as len(xvec) and len(Prob). remove?
    """
    idx1=0
    while(xvec[idx1]<=llim):
        Prob[idx1]=0
        idx1+=1
    if (idx1<1):
        raise RuntimeError("Error: lower energy limit is out of bounds\n")
    idxstart=idx1
    Prob[idx1], _=si.quad(func,llim,xvec[idx1])
    relerr = _/Prob[idx1]
    if np.abs(relerr)>1e-6:
        print("integration error:", relerr)
    idx1+=1
    while(xvec[idx1]<=hlim):
        junk, _=si.quad(func,xvec[idx1-1],xvec[idx1])
        if junk==0:
            if _!=0:
                print("integration is zero. integration error:", _)
        else:
            relerr = _/junk
            if np.abs(relerr)>1e-6:
                print("relative integration error:", _/junk)
        Prob[idx1]=Prob[idx1-1]+junk   
        idx1+=1
    idxstop=idx1
    while(idx1<veclen):
        Prob[idx1]=1
        idx1+=1
    # /*Normalize all Probability values*/
    Norm=Prob[idxstop-1]
    if (Norm>0):
        for idx2 in range(idxstart, idxstop):
            Prob[idx2]/=Norm
    return idxstart, idxstop


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


def init(Emin, Emax, datafile):
    'Emin, Emax: meV'
    inxvec, inyvec, xylength, tvec, mat = sns_source_load(datafile, 0, 2)
    nE, nt = mat.shape
    Es = inxvec
    nEvals = len(Es)
    assert(nEvals==nE)
    llim=inxvec[1];hlim=inxvec[xylength];
    print("Start calculating probability distribution")
    # /* calculate total number of neutrons in specified energy window */
    # I_E = lambda E: np.interp(E, inxvec, inyvec)
    I_E = sinterp.interp1d(inxvec, inyvec, kind='quadratic')
    INorm2, _=si.quad(I_E, Emin/1000.0, Emax/1000.0)
    Pvec = np.zeros(nE)
    idxstart,idxstop = Pcalc(I_E, llim,  hlim, inxvec, Pvec, xylength)
    print(idxstart, idxstop)
    Ptmat = np.zeros((nE, nt))
    ltlim=0.1
    htlim=1.8e3
    for iE in range(nEvals):
        # I_t = lambda t: np.interp(t, tvec, mat[iE])
        I_t = sinterp.interp1d(tvec, mat[iE], kind='quadratic')
        tidxstart, tidxstop = Pcalc(I_t, ltlim, htlim, tvec, Ptmat[iE], nt)
        continue
    EPmax=quadfuncint(Emax/1000.0,xylength,inxvec,Pvec)
    EPmin=quadfuncint(Emin/1000.0,xylength,inxvec,Pvec)
    return INorm2, Es, Pvec, tvec, Ptmat, EPmin, EPmax, (idxstart, idxstop), (tidxstart, tidxstop)


def test_init():
    dat = '/home/97n/dv/mcvine/resources/instruments/ARCS/moderator/source_sct521_bu_17_1.dat'
    INorm2, Es, Pvec, ts, Ptmat, EPmin, EPmax, Eidx_range, tidx_range = init(3, 1500., dat)
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


def test_generate():
    dat = 'source_rot2_cdr_cyl_3x3_20190417.dat'
    INorm2, Es, Pvec, ts, Ptmat, EPmin, EPmax, Eidx_range, tidx_range = init(3, 1500., dat)
    Eidx_start, Eidx_stop = Eidx_range
    tidx_start, tidx_stop = tidx_range
    E, t = generate(10, EPmin, EPmax, Es, Eidx_start, Eidx_stop, Pvec, ts, tidx_start, tidx_stop, Ptmat)
    print(E,t*1e6)
    return

def test_component():
    dat = 'source_rot2_cdr_cyl_3x3_20190417.dat'
    src = SNS_source_numba(
        'src', dat, 5, 20, 0.03, 0.03,
        5, .03, .03,
    )
    neutrons = src.process(neutron_buffer(10))
    for n in neutrons:
        print(n)
    return

def test_component_n1e6():
    dat = 'source_rot2_cdr_cyl_3x3_20190417.dat'
    src = SNS_source_numba(
        'src', dat, 5, 20, 0.03, 0.03,
        5, .03, .03,
    )
    neutrons = src.process(neutron_buffer(int(1e6)))
    return

def main():
    # test_generate()
    # test_component()
    test_component_n1e6()
    return

if __name__ == '__main__': main()
