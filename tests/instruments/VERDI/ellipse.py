#!/usr/bin/env python
# coding: utf-8

import numpy as np

class SingleWall:

    def __init__(
            self, a, b,
            z_src, src_size,
            z_guide_start, z_guide_end,
            z_target, target_size):
        """Single wall elliptical supper mirror. 2D

        Parameters
        ----------
        a, b : floats
          half of minor and major axes. unit: meter
        z_src : float
          z of source. unit: meter
        src_size : float
          src size. unit: meter
        z_guide_start : float
          distance from source to start of guide
        z_guide_end : float
          distance from source to end of guide
        z_target : float
          z of target. unit: meter
        target_size : float
          target size. unit: meter
        """
        self.a, self.b = a, b
        self.z_src, self.src_size = z_src, src_size
        self.z_guide_start, self.z_guide_end = z_guide_start, z_guide_end
        self.z_target, self.target_size = z_target, target_size
        return

    def calc_shield_xz(self):
        "calculate x and z of the shield vertex"
        # line from src bottom to top of guide_end
        x_guide_end = ellipse_x(self.z_guide_end-self.z_src, b=self.b, a=self.a)
        x_src = -self.src_size/2.
        k1 = (x_guide_end-x_src)/(self.z_guide_end-self.z_src)
        b1 = x_src - k1*self.z_src
        # line from guide start top to bottom of target
        x_guide_start = ellipse_x(self.z_guide_start-self.z_src, b=self.b, a=self.a)
        x_target = -self.target_size/2.
        k2 = (x_target-x_guide_start)/(self.z_target-self.z_guide_start)
        b2 = x_target - k2*self.z_target
        # intersection
        z_middle = -(b1-b2)/(k1-k2)
        x_middle = k1*z_middle + b1
        # x_middle2 = k2*z_middle + b2
        return x_middle, z_middle

    def acceptance_diagram(self):
        return single_reflection_ellipse_acceptance_diagram(
            self.src_size, self.a, self.b, self.z_guide_start, self.z_guide_end)

def single_reflection_ellipse_acceptance_diagram(src_height, a, b, z_guide_start, z_guide_end):
    """create curves for the single-reflection acceptance diagram of an elliptical mirror

    Parameters
    ----------
    src_height : float
      source height
    a, b : floats
      half of minor and major axes
    z_guide_start : float
      distance from source to start of guide
    z_guide_end : float
      distance from source to end of guide
    """
    z1, z2 = z_guide_start, z_guide_end
    z3 = 2*b
    y_of_z = lambda z: ellipse_x(z, a, b)
    y1, y2 = y_of_z(np.array([z1, z2]))
    y0_max = src_height/2.
    # y of target at single reflection at z
    y_of_z_refl1 = lambda z: y0_max/z*(z3-z)
    y2 = y_of_z(z2)
    # from end of guide
    y3_max = y_of_z_refl1(z2)
    y3_ge = np.linspace(-y3_max, y3_max, 100)
    div_ge = (y3_ge-y2)/(z3-z2)
    # from start of guide
    y3_max = y_of_z_refl1(z1)
    y3_gs = np.linspace(-y3_max, y3_max, 100)
    div_gs = (y3_gs-y1)/(z3-z1)
    # from top of src
    z = np.linspace(z1, z2, 100)
    y = y_of_z(z)
    y3_mt = y_of_z_refl1(z)
    div_mt = (y3_mt-y)/(z3-z)
    # from bottom of src
    y3_mb = -y3_mt
    div_mb = (y3_mb-y)/(z3-z)
    return (y3_ge, div_ge), (y3_gs, div_gs), (y3_mt, div_mt), (y3_mb, div_mb)

def create_guide_tapering_data(
    path, z0, z_start, z_exit, segments, major_axis_len, minor_axis_len):
    """
    Parameters
    ----------
    path : str
      output path
    z0 : float
      z of the center of the ellipse
    z_start : float
      z of the start opening of the guide
    z_exit : float
      z of the exiting opening of the guide
    segments : int
      number of segments
    major_axis_len : float
      length of major axis
    minor_axis_len : float
      length of minor axis
    """
    lines = []
    p = lambda s: lines.append(s)
    p('c use with Guide_tapering.comp')
    p('c start position: {} from focal point, length: {}, number of segments: {}'.format(
        z_start,z_exit-z_start,segments))
    p('c h1(i)     h2(i)   w1(i)    w2(i)')

    b = major_axis_len/2.; a = minor_axis_len/2.

    zs, interval =  np.linspace(
        z_start, z_exit, segments,retstep=True,endpoint=False)
    zs -= z0-b

    for z in zs:
        h1 = 2*ellipse_x(z, b = b, a = a)
        h2 = 2*ellipse_x(z+interval, b = b, a = a)
        w1 = 2*ellipse_x(z, b = b, a = a)
        w2 = 2*ellipse_x(z+interval, b = b, a = a)
        #print(z, h1, w1)
        p('{} {} {} {}'.format(h1,h2,w1,w2))
        continue
    # endPosition = z+interval
    # print('')
    # print("summary (delete for calculations):")
    # print("last point :" + str(endPosition), str(h2), str(w2))
    # print("length of each segment: "+str(interval))
    # print("end position : "+str(endPosition))
    with open(path, 'wt') as stream:
        for l in lines: stream.write(l+'\n')
    return

def ellipse_x(z,a,b):
    '''
    calculates x (or y) of elliptic guide segment at point along z axis (neutron path)
    center of ellipse is at (0, 0, b)

    Parameters
    ----------
    z : float / float array
      z values at which x (or y) of guide profile
    a : float
      half of minor axis length
    b : float
      half of major axis length
    '''
    return a*np.sqrt(z/b*(2-z/b))


def test_verdi():
    # VERDI
    guide11_z_start =  6.35
    guide11_z_exit = guide11_z_start + 10.99
    guide12_z_start =  guide11_z_exit
    guide12_z_exit = guide12_z_start + 9.56
    # guide12_z_exit = 31.75
    segments = 500
    majorAx = 31.75
    minorAx = 0.21
    create_guide_tapering_data(
        'guide11.dat', majorAx/2., guide11_z_start, guide11_z_exit, segments, majorAx, minorAx)
    create_guide_tapering_data(
        'guide12.dat', majorAx/2., guide12_z_start, guide12_z_exit, segments, majorAx, minorAx)

    guide21_z_start =  35.05
    guide21_z_exit = guide21_z_start + 2.44
    guide22_z_start =  guide21_z_exit
    guide22_z_exit = guide22_z_start + 1.01
    # guide12_z_exit = 31.75
    segments = 500
    majorAx = 40.0-31.75
    minorAx = 0.1
    z0 = (31.75+40.0)/2
    create_guide_tapering_data(
        'guide21.dat', z0, guide21_z_start, guide21_z_exit, segments, majorAx, minorAx)
    create_guide_tapering_data(
        'guide22.dat', z0, guide22_z_start, guide22_z_exit, segments, majorAx, minorAx)
    #
    wall = SingleWall(
        a=0.21/2, b=31.75/2,
        z_src=0., z_guide_start=6.35, z_guide_end=6.35+10.99+9.56, z_target=31.75,
        src_size=0.03, target_size=0.03
    )
    print(wall.calc_shield_xz())
    print(wall.acceptance_diagram())
    return

def main():
    test_verdi()
    return

if __name__ == '__main__': main()
