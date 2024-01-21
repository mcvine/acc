import numpy as np, math, numba
from numba import cuda

@cuda.jit(device=True)
def insert_into_sorted_list(d, l, N):
    'insert data "d" into existing sorted array (low to high) of length N'
    if N>=len(l): return N
    if N==0:
        l[0] = d
        return 1
    minidx = 0
    maxidx = N-1
    idx = minidx
    while minidx < maxidx-1:
        mididx = (minidx+maxidx)//2
        mid = l[mididx]
        if d>mid:
            minidx = mididx
        else:
            maxidx = mididx
    if minidx == maxidx:
        idx = minidx
        if d>l[idx]: idx = idx+1
    else: # minidx == maxidx-1
        if d>=l[maxidx]: idx = maxidx+1
        elif d<=l[minidx]: idx = minidx
        else: idx = minidx+1
    # shift
    for i in range(N, idx, -1):
        l[i] = l[i-1]
    l[idx] = d
    return N+1

@cuda.jit(device=True)
def insert_into_sorted_list_with_indexes(index, d, index_list, l, N):
    'insert data "d" into existing sorted array (low to high) of length N and also update the index list'
    if N>=len(l): return N
    if N==0:
        l[0] = d
        index_list[0] = index
        return 1
    minidx = 0
    maxidx = N-1
    idx = minidx
    while minidx < maxidx-1:
        mididx = (minidx+maxidx)//2
        mid = l[mididx]
        if d>mid:
            minidx = mididx
        else:
            maxidx = mididx
    if minidx == maxidx:
        idx = minidx
        if d>l[idx]: idx = idx+1
    else: # minidx == maxidx-1
        if d>=l[maxidx]: idx = maxidx+1
        elif d<=l[minidx]: idx = minidx
        else: idx = minidx+1
    # shift
    for i in range(N, idx, -1):
        l[i] = l[i-1]
        index_list[i] = index_list[i-1]
    l[idx] = d
    index_list[idx] = index
    return N+1

