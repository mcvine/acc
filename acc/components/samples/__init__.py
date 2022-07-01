import os
import mccomponents.sample

def loadScattererComposite(sampleassembly_xml):
    "load scatterer composite from sample assembly xml file"
    from mccomposite.extensions import HollowCylinder, SphereShell
    filename = os.path.realpath( sampleassembly_xml )
    dir, filename = os.path.split( os.path.abspath( filename ) )
    save = os.path.abspath( os.curdir )
    os.chdir( dir )

    from sampleassembly.saxml import parse_file
    sa = parse_file( filename )

    from mccomponents.sample.sampleassembly_support import sampleassembly2compositescatterer, \
         findkernelsfromxmls

    scatterercomposite = findkernelsfromxmls(
        sampleassembly2compositescatterer( sa ) )

    return scatterercomposite

def loadFirstHomogeneousScatterer(sampleassembly_xml):
    sc = loadScattererComposite(sampleassembly_xml)
    return sc.elements()[0]
