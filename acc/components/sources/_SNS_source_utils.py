#!/usr/bin/env python

"""
Converted from Garrett's SNS_source mcstas component
"""

import os
import numpy as np
import scipy.integrate as si
import scipy.interpolate as sinterp

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
    f = sinterp.interp1d(inxvec, Pvec, kind='quadratic')
    EPmax = float(f(Emax/1000.0))
    EPmin = float(f(Emin/1000.0))
    # EPmax=quadfuncint(Emax/1000.0,xylength,inxvec,Pvec)
    # EPmin=quadfuncint(Emin/1000.0,xylength,inxvec,Pvec)
    return INorm2, Es, Pvec, tvec, Ptmat, EPmin, EPmax, (idxstart, idxstop), (tidxstart, tidxstop)

def test_init():
    dat = '/home/97n/dv/mcvine/resources/instruments/ARCS/moderator/source_sct521_bu_17_1.dat'
    INorm2, Es, Pvec, ts, Ptmat, EPmin, EPmax, Eidx_range, tidx_range = init(3, 1500., dat)
    return
