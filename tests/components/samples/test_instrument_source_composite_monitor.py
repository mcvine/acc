#!/usr/bin/env python

import os
thisdir = os.path.dirname(__file__)

from test_sample_instrument_factory import construct

def instrument(samplexml, factory):
    tokens = factory.split('.')
    module = '.'.join(tokens[:-1])
    method = tokens[-1]
    import importlib
    module = importlib.import_module(module)
    factory = getattr(module, method)
    target = factory(name="sample", samplexml=samplexml)
    return construct(target, size=0.)
