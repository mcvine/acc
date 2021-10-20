
"""
Test Pcal
"""
import os
import math, numpy as np, numba as nb

from numba import guvectorize, float64
import scipy.integrate as si
import scipy.interpolate as sinterp
from timeit import default_timer as timer

import matplotlib.pyplot as plt


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


def quadint(xdes, x1,  x2, x3, y1, y2, y3):
    t1=((xdes-x2)*(xdes-x3)*y1)/((x1-x2)*(x1-x3))
    t2=((xdes-x1)*(xdes-x3)*y2)/((x2-x1)*(x2-x3))
    t3=((xdes-x1)*(xdes-x2)*y3)/((x3-x1)*(x3-x2))
    return t1+t2+t3

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





def Pcalc_loop(Ptmat, mat, tvec, ltlim, htlim, nEvals, nt):
    
    for iE in range(nEvals):
        I_t = sinterp.interp1d(tvec, mat[iE], kind='quadratic')
        tidxstart, tidxstop = Pcalc(I_t, ltlim, htlim, tvec, Ptmat[iE], nt)

        
        
@guvectorize(['void(float64[:],float64[:],float64[:],float64[:],int64[:],int64[:],int64[:],float64[:])'],
               '(nm),(nv),(mp),(mp),(mp),(mp),(n2)->(nm)', target='cuda')
def Pcalc_gpu(mat_gpu,tvec_gpu,qweights,qnodes,base_stencile,shift_ind,NxNy,out):
    
    Nx = NxNy[0]
    Ny = NxNy[1]
    
    shift_stencile = 0
    lagrange_basis = 0
    mp = qweights.shape[0]
    mq = mp
    
    for jr in range(Nx):
        for jc in range(Ny-1):
            dx = tvec_gpu[jc+1]-tvec_gpu[jc]
            for jq in range(mq):
                shift_ind[jq] = jc+base_stencile[jq]-shift_stencile
                
            shift_l =  min(shift_ind[0],0)
            shift_r = max(shift_ind[mq-1]-Ny+1,0)
            for jq in range(mq):
                shift_ind[jq] = shift_ind[jq]-shift_l- shift_r 
                

            
            qnodes[0] = tvec_gpu[jc]
            qnodes[1] = (tvec_gpu[jc+1]+tvec_gpu[jc])/2.0
            qnodes[2] = tvec_gpu[jc+1]
            
            quad_sum = 0
            
            quad_sum = 0
            for jq in range(mq):
                for jp in range(mp):
                    lagrange_basis = 1.0
                    cind = -1
                    for jps in range(mp-1):
                        cind = cind+1+(jps==jp)
                        lagrange_basis = lagrange_basis*(qnodes[jq]-tvec_gpu[shift_ind[cind]])/(tvec_gpu[shift_ind[jp]]-tvec_gpu[shift_ind[cind]])
                    
                    cind = jr+Nx*shift_ind[jp]
                    quad_sum = quad_sum + qweights[jq]*mat_gpu[cind]*lagrange_basis    
                        
            out[jr+Nx*(jc+1)] = out[jr+jc*Nx]+max(quad_sum,0)*dx/2.0
            
        for jc in range(Ny):
            out[jr+jc*Nx] = out[jr+jc*Nx]/out[jr+Nx*(Ny-1)]


datafile = 'source_rot2_cdr_cyl_3x3_20190417.dat'
Emin = 5
Emax = 20
inxvec, inyvec, xylength, tvec, mat = sns_source_load(datafile, 0, 2)

nE, nt = mat.shape
Es = inxvec
nEvals = len(Es)
assert(nEvals==nE)
llim=inxvec[1];hlim=inxvec[xylength];
print("Start calculating probability distribution")

I_E = sinterp.interp1d(inxvec, inyvec, kind='quadratic')
INorm2, _=si.quad(I_E, Emin/1000.0, Emax/1000.0)
Pvec = np.zeros(nE)
idxstart,idxstop = Pcalc(I_E, llim,  hlim, inxvec, Pvec, xylength)
print(idxstart, idxstop)
Ptmat = np.zeros((nE, nt))
ltlim=0.1
htlim=1.8e3

print('start cpu')
s = timer()


# N = 4
# M = 3
# pfloat = 3.5
# pint = 1
# arr = np.zeros([N,M])
# arr2 = np.zeros([N,M])
# vec = np.zeros(N)
# print(arr)
# test_gpu(arr, arr2, vec, pfloat, pint)
# print(arr)



Pcalc_loop(Ptmat, mat, tvec, ltlim, htlim, nEvals, nt)


e = timer()
print('cpu final time: ')
print(e - s)



##Set up quad points and weights ##


mp = 3 # Order of spline interpolation
mq = 3 # Order of quadure integration
qnodes = np.zeros(mq,dtype=np.float64)
qweights = np.zeros(mq,dtype=np.float64)
    
qweights[0] = 1.0/3.0
qweights[1] = 4.0/3.0
qweights[2] = 1.0/3.0
base_stencile = np.arange(mq)
shift_ind = np.arange(mq) 
NxNy = np.zeros(2,dtype=int)
NxNy[0] = mat.shape[0]
NxNy[1] = mat.shape[1]
mat_gpu = mat.reshape(np.prod(mat.shape), order='F')
Ptmat_gpu = np.zeros(np.prod(mat.shape))
tvec_gpu = np.zeros(tvec.shape[0])
tvec_gpu[:] = tvec[:]

print('start gpu')
s = timer()

for i in range(10):
    Pcalc_gpu(mat_gpu,tvec_gpu,qweights,qnodes,base_stencile,shift_ind,NxNy,Ptmat_gpu)

e = timer()
print('gpu final time: ')
print(e - s)

Ptmat_gpu  = Ptmat_gpu.reshape(mat.shape, order='F') 

print('Error')
print(np.max(np.abs(Ptmat-Ptmat_gpu)))



plt.plot(tvec[0:85],Ptmat[0,0:85],tvec[0:85],Ptmat_gpu[0,0:85])
plt.show()
plt.plot(tvec[0:85],mat[0,0:85])
plt.show()

