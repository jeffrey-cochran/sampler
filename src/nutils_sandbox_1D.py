from nutils import function, export, mesh, solver, testing, cli
from nutils.expression_v2 import Namespace
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

ns = function.Namespace()

topo, geom = mesh.rectilinear([np.linspace(0, 1, 5)])

ns.x = geom
ns.basis = topo.basis('spline', degree=3)
bezier = topo.sample('bezier', 120)
# ================================================================================

#
# Sample with the constraint that c[0] = c[1] ... c[-1] = c[-2]
Q = np.identity(7)
Q[0,:] = [1,-1,0,0,0,0,0]
Q[-1,:] = [0,0,0,0,0,1,-1]

samples = np.random.randn(7)
samples[0] = 0
samples[-1] = 0

temp = sp.sparse.linalg.spsolve(Q, samples)


# ================================================================================
ns.f = np.dot(ns.basis, temp)
ns.fgrad = 'f_,0'

xvals, vals, grad_vals = bezier.eval([ns.x, ns.f, ns.fgrad])

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.set_title('Function Value')
ax1.plot(xvals, vals)

ax2.set_title('Gradient Value')
ax2.plot(xvals, grad_vals)

plt.show()