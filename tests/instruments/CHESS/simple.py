import os
thisdir = os.path.abspath(os.path.dirname(__file__))
import mcvine, mcvine.components

def instrument(
        use_acc=True,
        Minwavelength = 0.2, # [AA] Minum Wavelength emitted from source
        Maxwavelength = 20, # [AA] Maximum Wavelength emitted from source
        Zguide1 = 14.2, # [m] Total Guide1 Length
        Yguide1 = 0.05559, # [m] Total Guide1 Width
        Rslit1 = 0.015, # [m] Guide 1 Slit Radius
        Zguide2 = 14.9, # [m] Total Guide2 Length
        Yguide2 = 0.074018, # [m] Total Guide2 Width
        Rslit2 = 0.021891, # [m] Guide 2 Slit Radius
        Zguide3 = 0.95, # [m] Total Guide3 Length
        Yguide3 = 0.032639, # [m] Total Guide3 Width
        Rslit3 = 0.015077, # [m] Guide 3 Slit Radius
        polariserIn=0, 
        SFlipper=0,
    ):
    instrument = mcvine.instrument()
    if use_acc:
        import mcvine.acc.components as mc
    else:
        import mcvine.components as mc
    origin = mc.optics.Arm(name='origin')
    instrument.append(origin, position=(0,0,0))

    source_simple = mc.sources.Source_simple(
        name="source",
        yheight=0.03,
        xwidth=0.03, 
        dist=0.751, 
        focus_xw=0.032, 
        focus_yh=0.032, 
        lambda0=10.1, 
        dlambda=9.9, 
        flux=1e14,
    )
    instrument.append(source_simple, position=(0,0,0), relativeTo=origin)
    bypassSourceCHESS = mc.optics.Slit(name='bypassSourceCHESS', radius = 0.015)
    instrument.append(bypassSourceCHESS, position=(0,0,0.071), relativeTo=origin)

    Slitguide1 = mc.optics.Slit(name='Slitguide1', radius = Rslit1)
    instrument.append(Slitguide1, position=(0,0,0.749), relativeTo=origin)

    Guide_anyshape = mc.optics.Guide_anyshape_gravity
    Zeppelin1 = Guide_anyshape(
        name = "Zepplin1",
        geometry = os.path.join(thisdir, "Guide1_3tmp.off"),
        xwidth = Yguide1, yheight = Yguide1, zdepth = Zguide1,
        center = 1,
        m=6.0, R0=0.99, Qc=0.0219, alpha=3.044, W=0.0025,
    )
    instrument.append(Zeppelin1, position=(0,0,7.85), orientation=(0, 0, 22.5), relativeTo=origin)

    post_zeppedlin1_Ixdivx = mc.monitors.PosDiv_monitor(
        name = "post_zeppedlin1_Ixdivx",
        xwidth=0.025, yheight=0.025,
        maxdiv=2.,
        npos=50, ndiv=50,
        filename = "post_zeppedlin1_Ixdivx.h5",
    )
    instrument.append(post_zeppedlin1_Ixdivx, position=(0,0,15.048), orientation=(0, 0, 0), relativeTo=origin)

    Slitguide2 = mc.optics.Slit(name = "Slitguide2", radius = Rslit2)
    instrument.append(Slitguide2, position=(0,0,15.049), relativeTo=origin)

    Zeppelin2 = Guide_anyshape(
        name = "Zepplin2",
        geometry = os.path.join(thisdir, "Guide2_3tmp.off"),
        xwidth = Yguide2,
        yheight = Yguide2, zdepth = Zguide2,
        center = 1,
        m=6.0, R0=0.99, Qc=0.0219, alpha=3.044, W=0.0025,
    )
    instrument.append(Zeppelin2, position=(0,0,22.5), orientation=(0,0,22.5), relativeTo=origin)

    post_zeppedlin2_Ixdivx = mc.monitors.PosDiv_monitor(
        name = "post_zeppedlin2_Ixdivx",
        xwidth=0.025, yheight=0.025,
        maxdiv=2.,
        npos=50, ndiv=50,
        filename = "post_zeppedlin2_Ixdivx.h5",
    )
    instrument.append(post_zeppedlin2_Ixdivx, position=(0,0,30), orientation=(0, 0, 0), relativeTo=origin)

    return instrument