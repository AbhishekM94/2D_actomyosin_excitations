from dolfin import *
import numpy 
import matplotlib.pyplot as plt
import os
from scipy import pi, linspace, loadtxt, meshgrid, exp , cos, sqrt

#parameters of the equation
c = as_vector([0.0000000000000000001,0]) 
D = 0.05
theta = 0.5 #for Crank Nicholson time stepping
px = 0.5
py = 0.5
sigma = 0.005
A = 1.0

# Time-stepping parameters
t_start = 0.01
dt = 0.01
t_end = 150*dt

# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]

# Create periodic boundary condition
pbc = PeriodicBoundary()

#Definition of mesh
nx = ny = 50
mesh = UnitSquareMesh(nx,ny)
V = FunctionSpace(mesh,'P',1,constrained_domain=PeriodicBoundary())

#Initial condition for r
ic = Expression("A*exp(-((x[0]-px)*(x[0]-px) + (x[1]-py)*(x[1]-py))/(2*sigma))", degree = 1, px=px, py=py, sigma=sigma, A=A)

#Advection term
def adv_term(r,v):
    return v*dot(c,grad(r))*dx

#Diffusion term
def diff_term(r,v):
    return -D*inner(grad(r),grad(v))*dx

#Define variational problem
r = Function(V)
r0 = Function(V)
v = TestFunction(V)
F = (1.0/dt)*(r-r0)*v*dx + theta*adv_term(r,v) + (1.0-theta)*adv_term(r,v) - theta*diff_term(r,v) - (1.0-theta)*diff_term(r,v)

#Assigning IC to u0
r0.interpolate(ic)

#Analytical solution for pure diffusion(c ~ (0,0))
rn0 = Expression("(A/((sigma/2) + t)*4*pi*D)*exp(-((x[0]-px)*(x[0]-px) + (x[1]-px)*(x[1]-px))/(4*D*(sigma/2) + 4*D*t))", degree = 1, sigma=sigma, D=D, t=t, pi = pi, A=A, px=px, py=py) 

#Maximum Error computer
def err_compare(r,t):
    rn = exact_soln(t)
    maxdiff =  abs(rn.vector().array() - r.vector().array()).max() #abs(ue - u).max()
    print('t = %s, Maxdiff = %s' % (t, maxdiff))
    return maxdiff

#Visualization
rFile = XDMFFile('r_numerical.xdmf')
mer = numpy.zeros(int((t_end - t_start)/dt) )
rnFile = XDMFFile('r_analytical.xdmf')

#Time loop
t = t_start
i = 0
while t < t_end:
    print 'time =', t
    
    solve(F == 0, r)
    rn = interpolate(rn0, V)
    error = np.abs(rn.vector().array() - r.vector().array()).max()
    mer[i] = error
    r0.assign(r)
    t += dt
    i += 1 
    rFile.write(r, t)
    rnFile.write(rn,t)
 
rFile.close()
rnFile.close()
print mer
plt.plot(range(int(((t_end - t_start)/dt))) , mer)
plt.xlabel("time")
plt.ylabel("max error")   
plt.show() 














