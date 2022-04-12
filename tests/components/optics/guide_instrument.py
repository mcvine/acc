#!/usr/bin/env python

from instrument_factory import construct


def instrument(
        guide_mod=None, guide_factory=None, is_acc=False):
    if guide_factory:
        import mcvine.components
        factory = eval(guide_factory)
    elif guide_mod:
        import importlib
        guide_module = importlib.import_module(guide_mod)
        factory = guide_module.Guide
    elif is_acc:
        from mcvine.acc.components.optics.guide import Guide
        factory = Guide
    else:
        import mcvine.components
        factory = mcvine.components.optics.Guide

    guide_length = 10.
    target = factory(
        name='guide',
        w1=0.035, h1=0.035, w2=0.035, h2=0.035, l=guide_length,
        R0=0.99, Qc=0.0219, alpha=6.07, m=3, W=0.003
    )
    return construct(target, guide_length)
