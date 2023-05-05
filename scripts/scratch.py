from minigrad.tensor import BadTensor
from minigrad.graph import draw_dot
from numpy import r_

r_[1]

a = BadTensor(r_[2.], label='a')
b = BadTensor(r_[-3.], label='b')
c = BadTensor(r_[10.], label='c')
f = BadTensor(r_[-2.], label='f')
e = (a * b).label_('e')
d = (c + e).label_('d')
L = (d * f).label_('L')

L.backward()
pass