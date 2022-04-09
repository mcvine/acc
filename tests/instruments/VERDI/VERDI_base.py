import os, sys
thisdir = os.path.dirname(__file__)
import mcvine
from ellipse import create_guide_tapering_data, SingleWall, ellipse_x
from mcvine.acc.components.optics.arm import Arm
from mcvine.acc.components.optics.slit import Slit
from mcvine.acc.components.optics.beamstop import Beamstop
from mcvine.acc.components.optics.guide_tapering import Guide as Guide_tapering
from mcvine.acc.components.sources.SNS_source import SNS_source
from mcvine.acc.components.monitors.wavelength_monitor import Wavelength_monitor
from mcvine.acc.components.monitors.posdiv_monitor import PosDiv_monitor
from mcvine.acc.components.monitors.psd_monitor import PSD_monitor

def instrument(
        guide_mvalue = 6.,
        sample_width = 0.005, sample_height = 0.005,
        moderator_datafile=os.path.join(thisdir, "source_rot2_cdr_cyl_3x3_20190417.dat"),
        source_width = 0.03, source_height = 0.03,
        secsource_size = 0.02,
        xDivergence = 2, yDivergence = 2,
        guide1_start = 6.35, guide1_len = 10.99 + 9.56,
        ellipse1_major_axis_len = 31.75, # second source z position
        ellipse1_minor_axis_len = 0.21,
        guide11_dat = '_autogen_guide11.dat', guide12_dat = '_autogen_guide12.dat',
        guide2_start = 35.05, guide2_len = 2.44 + 1.01,
        ellipse2_major_axis_len = 40.0-31.75,
        ellipse2_minor_axis_len = 0.1,
        guide21_dat = '_autogen_guide21.dat', guide22_dat = '_autogen_guide22.dat',
        guide_n_segments = 500,
        Emin = 0.1, Emax = 1000.,
):
    # calc shield position
    # 1st mirror
    wall1 = SingleWall(
        a=ellipse1_minor_axis_len/2, b=ellipse1_major_axis_len/2,
        z_src=0., z_guide_start=guide1_start, z_guide_end=guide1_start+guide1_len,
        z_target=ellipse1_major_axis_len,
        src_size=source_width, target_size=secsource_size,
    )
    shield1_x, shield1_z = wall1.calc_shield_xz()
    guide11_len = shield1_z - guide1_start
    guide12_len = guide1_len - guide11_len
    print('guide11, guide12:', guide11_len, guide12_len)
    print('shield1 stick out length: ', shield1_x)
    moderator_focusheight = moderator_focuswidth = ellipse_x(
        guide1_start, ellipse1_minor_axis_len/2, ellipse1_major_axis_len/2)*2
    moderator_focusdistance = guide1_start*.99
    print('moderator focus distance and dims: ',
          moderator_focusdistance, moderator_focuswidth, moderator_focusheight)
    # 2nd mirror backward!
    z_guide_start=ellipse1_major_axis_len+ellipse2_major_axis_len-guide2_start-guide2_len
    wall2 = SingleWall(
        a=ellipse2_minor_axis_len/2, b=ellipse2_major_axis_len/2,
        z_src=0.,
        z_guide_start=z_guide_start,
        z_guide_end=z_guide_start+guide2_len,
        z_target=ellipse2_major_axis_len,
        src_size=sample_width, target_size=secsource_size,
    )
    shield2_x, shield2_z = wall2.calc_shield_xz()
    guide22_len = shield2_z - z_guide_start
    guide21_len = guide2_len - guide22_len
    guide21_start = ellipse1_major_axis_len + (ellipse2_major_axis_len - z_guide_start - guide2_len)
    print('guide21, guide22:', guide21_len, guide22_len)
    print('shield2 stick out length: ', shield2_x)

    # create guide data
    ellipse1_center = ellipse1_major_axis_len/2.
    guide11_start = guide1_start
    guide12_start = guide11_end = guide11_start + guide11_len
    guide12_end = guide12_start + guide12_len
    create_guide_tapering_data(
        guide11_dat, ellipse1_center,
        guide11_start, guide11_end, guide_n_segments,
        ellipse1_major_axis_len, ellipse1_minor_axis_len)
    create_guide_tapering_data(
        guide12_dat, ellipse1_center,
        guide12_start, guide12_end, guide_n_segments,
        ellipse1_major_axis_len, ellipse1_minor_axis_len)
    # ellipse2 first focus overlaps with ellipse1 second focus
    ellipse2_center = ellipse1_major_axis_len + ellipse2_major_axis_len/2
    guide22_start = guide21_end = guide21_start + guide21_len
    guide22_end = guide22_start + guide22_len
    create_guide_tapering_data(
        guide21_dat, ellipse2_center,
        guide21_start, guide21_end, guide_n_segments,
        ellipse2_major_axis_len, ellipse2_minor_axis_len)
    create_guide_tapering_data(
        guide22_dat, ellipse2_center,
        guide22_start, guide22_end, guide_n_segments,
        ellipse2_major_axis_len, ellipse2_minor_axis_len)

    instrument = mcvine.instrument()

    origin = Arm(name='origin')
    instrument.append(origin, position=(0.0, 0.0, 0.0), orientation=(0, 0, 0))

    moderator = SNS_source(
        'moderator', moderator_datafile,
        Anorm=0.0009,
        Emin=Emin, Emax=Emax,
        dist=moderator_focusdistance,
        focus_xw=moderator_focuswidth, focus_yh=moderator_focusheight,
        xwidth=source_width, yheight=source_height)
    instrument.append(
        moderator, position=(0.0, 0.0, 0.0), orientation=(0, 0, 0),
        relativeTo=origin)

    ModeratorSpectrumL = Wavelength_monitor(
        name='ModeratorSpectrumL', Lmax=20, Lmin=0,
        filename="moderator_L.dat", nchan=100,
        # restore_neutron=1,
	    xwidth = source_width*1.1, yheight=source_height*1.1,
        #xwidth=sample_width, yheight=sample_height,
    )
    instrument.append(
        ModeratorSpectrumL, position=(0.0, 0.0, 1e-05), orientation=(0, 0, 0),
        relativeTo=moderator)

    ModeratorDivergence_xpos = PosDiv_monitor(
        name='ModeratorDivergence_xpos', filename="moderator_divXpos.dat",
        maxdiv=xDivergence, ndiv=1000, npos=100,
	    xwidth = source_width*1.1, yheight=source_height*1.1,
        #xwidth=sample_width, yheight=sample_height,
    )
    instrument.append(
        ModeratorDivergence_xpos,
        position=(0.0, 0.0, 2e-05), orientation=(0, 0, 0),
        relativeTo=moderator)

    ModeratorDivergence_ypos = PosDiv_monitor(
        name='ModeratorDivergence_ypos', filename="moderator_divYpos.dat",
        maxdiv=yDivergence, ndiv=1000, npos=100,
	    xwidth = source_width*1.1, yheight=source_height*1.1,
        #xwidth=sample_width, yheight=sample_height,
    )
    instrument.append(
        ModeratorDivergence_ypos,
        position=(0.0, 0.0, 3e-05), orientation=(0.0, 0.0, 90.0),
        relativeTo=moderator)

    Guide1_1 = Guide_tapering(
        name='Guide1_1',
        option="file={}".format(guide11_dat),
        l=guide11_len, mx=guide_mvalue, my=guide_mvalue)
    instrument.append(
        Guide1_1, position=(0.0, 0.0, guide11_start), orientation=(0, 0, 0),
        relativeTo=moderator)

    Shield1_1 = Beamstop(
        name='Shield1_1', xmax=shield1_x, xmin=-0.5, ymax=0.5, ymin=-0.5)
    instrument.append(
        Shield1_1, position=(0.0, 0.0, guide11_len+0.000001), orientation=(0, 0, 0),
        relativeTo=Guide1_1)

    Shield1_2 = Beamstop(
        name='Shield1_2', xmax=0.5, xmin=-0.5, ymax=shield1_x, ymin=-0.5)
    instrument.append(
        Shield1_2, position=(0.0, 0.0, guide11_len+0.000002), orientation=(0, 0, 0),
        relativeTo=Guide1_1)

    Guide1_2 = Guide_tapering(
	    option="file={}".format(guide12_dat),
        name='Guide1_2', l=guide12_len, mx=guide_mvalue, my=guide_mvalue)
    instrument.append(
        Guide1_2, position=(0.0, 0.0, guide11_len+4e-6), orientation=(0, 0, 0),
        relativeTo=Guide1_1)

    SecSource = Arm(name='SecSource')
    instrument.append(
        SecSource, position=(0.0, 0.0, ellipse1_major_axis_len), orientation=(0, 0, 0),
        relativeTo=origin)

    secsrc_divxpos = PosDiv_monitor(
        name='secsrc_divxpos', filename="secsrc_divXpos.dat",
        maxdiv=xDivergence, ndiv=1000, npos=100,
        xwidth=0.15, yheight=0.15)
    instrument.append(
        secsrc_divxpos, position=(0.0, 0.0, 1e-07), orientation=(0, 0, 0),
        relativeTo=SecSource)

    secsrc_divypos = PosDiv_monitor(
        name='secsrc_divypos', filename="secsrc_divYpos.dat",
        maxdiv=yDivergence, ndiv=1000, npos=100,
        xwidth=0.15, yheight=0.15)
    instrument.append(
        secsrc_divypos,
        position=(0.0, 0.0, 2e-07), orientation=(0.0, 0.0, 90.0),
        relativeTo=SecSource)

    SecSourceSlit = Slit(
        name='SecSourceSlit', xmax=0.013, xmin=-0.01, ymax=0.013, ymin=-0.01)
    instrument.append(
        SecSourceSlit, position=(0.0, 0.0, 1e-05), orientation=(0, 0, 0),
        relativeTo=SecSource)

    secsrc_IL = Wavelength_monitor(
        name='secsrc_IL', Lmax=20, Lmin=0, filename="secsrc_L",
        nchan=100,
        # restore_neutron=1,
        xwidth=0.026, yheight=0.026)
    instrument.append(
        secsrc_IL, position=(0.0, 0.0, 1e-6), orientation=(0, 0, 0),
        relativeTo=SecSourceSlit)

    Guide2_1 = Guide_tapering(
	    option="file={}".format(guide21_dat),
        name='Guide2_1', l=guide21_len, mx=guide_mvalue, my=guide_mvalue)
    instrument.append(
        Guide2_1, position=(0.0016, 0.0016, guide21_start), orientation=(0, 0, 0),
        relativeTo=moderator)

    Shield2_1 = Beamstop(
        name='Shield2_1', xmax=0.5, xmin=-shield2_x, ymax=0.5, ymin=-0.5)
    instrument.append(
        Shield2_1, position=(0.0, 0.0, guide21_len + 1e-6), orientation=(0, 0, 0),
        relativeTo=Guide2_1)

    Shield2_2 = Beamstop(
        name='Shield2_2', xmax=0.5, xmin=-0.5, ymax=0.5, ymin=-shield2_x)
    instrument.append(
        Shield2_2, position=(0.0, 0.0, guide21_len + 2e-6), orientation=(0, 0, 0),
        relativeTo=Guide2_1)

    Guide2_2 = Guide_tapering(
	    option="file={}".format(guide22_dat),
        name='Guide2_2', l=guide22_len, mx=guide_mvalue, my=guide_mvalue)
    instrument.append(
        Guide2_2, position=(0.0, 0.0, guide21_len+4e-6), orientation=(0, 0, 0),
        relativeTo=Guide2_1)

    sampleMantid = Arm(name='sampleMantid')
    instrument.append(
        sampleMantid, position=(0.001, 0.001, 40.0), orientation=(0, 0, 0),
        relativeTo=origin)

    sample0_IL = Wavelength_monitor(
        name='sample0_IL', Lmax=20, Lmin=0, filename="sample0_L",
        nchan=100,
        # restore_neutron=1,
        xwidth=sample_width, yheight=sample_height)
    instrument.append(
        sample0_IL, position=(0.0, 0.0, 0.), orientation=(0, 0, 0),
        relativeTo=sampleMantid)

    brilliancePack = Arm(name='brilliancePack')
    instrument.append(
        brilliancePack, position=(0.0, 0.0, 0.0), orientation=(0, 0, 0),
        relativeTo=sampleMantid)

    Mask = Slit(
        name='Mask', width=sample_width, height=sample_height)
    instrument.append(
        Mask, position=(0.0, 0.0, 0.0), orientation=(0, 0, 0),
        relativeTo=brilliancePack)

    imagePlate = PSD_monitor(
        name='imagePlate', filename="sample_xy",
        nx=500, ny=500,
        # restore_neutron=1,
        xwidth=3*sample_width,
        yheight=3*sample_height)
    instrument.append(
        imagePlate, position=(0.0, 0.0, 1e-05), orientation=(0, 0, 0),
        relativeTo=brilliancePack)

    sampleSpectrumL = Wavelength_monitor(
        name='sampleSpectrumL', Lmax=20, Lmin=0, filename="sample_L",
        nchan=100,
        # restore_neutron=1,
        xwidth=sample_width, yheight=sample_height)
    instrument.append(
        sampleSpectrumL, position=(0.0, 0.0, 2e-05), orientation=(0, 0, 0),
        relativeTo=brilliancePack)

    Divergence_xpos = PosDiv_monitor(
        name='Divergence_xpos', filename="sample_divXpos.dat",
        maxdiv=xDivergence, ndiv=1000, npos=100,
        xwidth=sample_width, yheight=sample_height)
    instrument.append(
        Divergence_xpos, position=(0.0, 0.0, 3e-05), orientation=(0, 0, 0),
        relativeTo=brilliancePack)

    Divergence_ypos = PosDiv_monitor(
        name='Divergence_ypos', filename="sample_divYpos.dat",
        maxdiv=yDivergence, ndiv=1000, npos=100,
        xwidth=sample_width, yheight=sample_height)
    instrument.append(
        Divergence_ypos,
        position=(0.0, 0.0, 4e-05), orientation=(0.0, 0.0, 90.0),
        relativeTo=brilliancePack)
    return instrument

if __name__ == '__main__':
    ins = instrument()