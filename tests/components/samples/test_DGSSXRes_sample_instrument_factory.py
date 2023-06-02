# -*- python -*-
from mcvine.acc.test.instrument_factory import InstrumentBuilder as base

class Builder(base):

    @classmethod
    def get_source(cls, neutron):
        """
        Construct a source.
        """
        import mcvine.components as mc
        return mc.sources.MonochromaticSource(
            name='source',
            neutron=neutron
        )
