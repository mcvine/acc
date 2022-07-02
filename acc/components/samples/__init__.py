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

def getAbsScttCoeffs(kernel):
    abs = kernel.absorption_coefficient
    sctt = kernel.scattering_coefficient

    if abs is None or sctt is None:
        #need to get cross section from sample assembly representation
        # svn://danse.us/inelastic/sample/.../sampleassembly
        #origin is a node in the sample assembly representation
        #
        #scatterer_origin is assigned to kernel when a kernel is
        #constructed from kernel xml.
        #see sampleassembly_support.SampleAssembly2CompositeScatterer for details.
        origin = kernel.scatterer_origin
        from sampleassembly import cross_sections
        abs, inc, coh = cross_sections( origin )
        sctt = inc + coh
        pass

    abs, sctt = _units_remover.remove_unit( (abs, sctt), 1./units.length.meter )
    return abs, sctt

from mccomposite.units_utils import UnitsRemover
from mccomposite import units
_units_remover = UnitsRemover(
    length_unit = units.length.meter, angle_unit = units.angle.degree)
