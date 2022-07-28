#!/usr/bin/env python

from mcni import neutron_buffer, neutron
from mcni.neutron_storage import storage

b = neutron_buffer(1)

# use this to check neutronfromstorage is reading and then emitting the right neutron
# by comparing the arrays directly
# see test_neutronfromstorage.test_component
b[0] = neutron(r=(0,1,2), v=(3,4,5), s=(6,7), time=8, prob=9)
st = storage("singletestneutron.mcv", mode='w')
st.write(b)
st.close()

# use this so that there is intensity at the center of the psdmonitor.
# see test_neutronfromstorage.test_component_with_monitors
b[0] = neutron(r=(0,0,0), v=(0,0,1000), s=(0,0), time=0, prob=1)
st = storage("singletestneutron2.mcv", mode='w')
st.write(b)
st.close()
