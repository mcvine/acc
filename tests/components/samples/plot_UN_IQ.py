#!/usr/bin/env python

import numpy as np, histogram.hdf as hh

def plot(path):
    IQE = hh.load(path)
    try:
        IQE.energy
        eaxis = 'energy'
    except:
        eaxis = 'E'
    Es = np.arange(0., 360., 50.)
    I_Q_list = []
    for E in Es:
        I_Q_list.append( IQE[(), (E-10, E+10)].sum(eaxis) )
    from matplotlib import pyplot as plt
    plt.figure()
    for E, IQ in zip(Es, I_Q_list):
        plt.plot(IQ.Q, IQ.I, 'k', label=f"{E}")
    plt.show()
    return

def main():
    import sys
    path = sys.argv[1]
    plot(path)
    return

if __name__ == '__main__': main()
