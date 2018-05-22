import matplotlib.pyplot as plt
import matplotlib.tri as tri
from dolfin import *
import numpy
import os
from scipy import pi, linspace, loadtxt, meshgrid, exp , cos, sqrt
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import numpy as np

#parameters of the model (to be completed!!)
D = Constant(0.01)
theta = Constant(0.5)
e = Constant(1.5)
k_b =  Constant(1.0)                
k_u0 = Constant(1.0)                   
c_1 = Constant(0.1) 
alpha = Constant(-3.0)
theta = Constant(0.5)
tau = Constant(1.0)
B = Constant(3.0)
sig1 = Constant(-0.01)
demu = Constant(5.2)
sig2 = Constant(0.1)
chi1 = Constant(1.0)
chi0 = Constant(0.5)
c = Constant(0.1)
lambda_ = Constant(1.25)
mu = Constant(1.0)
T = Constant((0,0))


# Time-stepping parameters
t_start = 0.0
dt = 0.005
t_end = 10*dt

#mesh parameters
Lx = 4.0; Ly = 4.0
x1 = -2.0; y1 = -2.0
xend=x1+Lx
yend=y1+Ly
dx0 = dx1 = 0.1
nx = 60 #int(Lx/dx0) 
ny = 60  #int(Ly/dx1)
dim = 2

#boundary condition for u-field (have to check working)
class PeriodicBoundary(SubDomain):
# Periodic BC
   # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], x1) or near(x[1], y1)) and 
                (not ((near(x[0], x1) and near(x[1], y1+Ly)) or 
                        (near(x[0], x1+Lx) and near(x[1], y1)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], x1+Lx) and near(x[1], y1+Ly):
            y[0] = x[0] - Lx
            y[1] = x[1] - Ly
        elif near(x[0], x1+Lx):
            y[0] = x[0] - Lx
            y[1] = x[1]
        else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - Ly

#Definition of mesh, elements and functionspaces
mesh = RectangleMesh(Point(x1, y1), Point(x1+Lx, y1+Ly), nx, ny,"right/left")
V1 = VectorElement('P', triangle , 2)
V2 = VectorElement('P', triangle , 2)
V3 = FiniteElement('P', triangle , 2)
element = MixedElement([V1,V2,V3])
V = FunctionSpace(mesh, element) #constrained_domain=PeriodicBoundary())
S = FunctionSpace(mesh, V3) 
# meshgrid params for plotting
n = mesh.num_vertices()
#d = mesh.geometry().dim()

#Fixed boundary condition
#ubc = Constant(0.0)
#def ubc_boundary(x, on_boundary):
 #   return on_boundary
#bc = DirichletBC(V.sub(2), ubc, ubc_boundary)

#Initial conditions expressions
px = py = Constant(0.8)
I1 = Expression(('0.0','x[0]+x[1]'), degree = 2)
I2 = Expression("3*exp(-((x[0]-px)*(x[0]-px) + (x[1]-py)*(x[1]-py))/0.05)", px = px, py = py, degree = 2)

#Definition of required functions, test and trial
u_v_r = Function(V)
du_dv_dr = TrialFunction(V)
du,dv,dr = split(du_dv_dr)
v1_v2_v3 = TestFunction(V)
v1,v2,v3 = split(v1_v2_v3)
u_r_0 = Function(V)
u0, v0, r0, = split(u_r_0)
u0 = interpolate(I1,V.sub(0).collapse())
r0 = interpolate(I2,V.sub(2).collapse())
u,v,r = split(u_v_r)
d = u.geometric_dimension()
I = Identity(d)

#Definitions after weak formulation

def epsilon(u):
    eps = 0.5*(nabla_grad(u)+transpose(nabla_grad(u)))
    return eps

def sigma(u):
    return 2*mu*epsilon(u) + lambda_*nabla_div(u)*Identity(d) 
    
#Diffusion term
def diff_term(r, v3):
    return D*(-inner(grad(r), grad(v3)) ) * dx
    
# Advection
def adv_term(u, v1, r):
    return v1*div(grad(r)*u)*dx
    
# Reaction
"Reaction terms chosen from Deb Sankar's one dimensional model" 
def source2(r,v3):
    f = Expression("k_b*(1-(c_1*str(strain(u)))",epsilon = epsilon, k_b = k_b, c_1 = c_1, degree = 1)
    return inner(f,v3)*dx
    
def source1(r,v3):
    f = Expression("k_u0*exp(alpha*e)",k_u0 = k_u0,alpha = alpha, e = e, degree = 1)
    return f*r*v3*dx

F = (1.0/dt)*inner((u-u0),v1)*tau*dx - theta*inner(sigma(u0),epsilon(v1))*dx - (1.0-theta)*inner(sigma(u0),epsilon(v1))*dx - theta*inner(div(-((sig1*r0*chi0*demu)/1+(sig2*r0))*Identity(d)),v1)*dx - (1.0-theta)*inner(div(-((sig1*r0*chi0*demu)/1+(sig2*r0))*Identity(d)),v1)*dx \
  + theta*inner((u-u0)/dt, v2)*dx + (1.0-theta)*inner((u-u0)/dt, v2)*dx - theta*inner(v,v2)*dx - (1.0-theta)*inner(v,v2)*dx \
  + (1.0/dt)*inner((r - r0),v3)*dx + theta*inner(nabla_div(v*r),v3)*dx + (1.0-theta)*inner(nabla_div(v*r),v3)*dx - (theta*diff_term(r,v3) + (1.0-theta)*diff_term(r,v3)) + theta*(k_b*(1.0-(c_1*0.5))*v3)*dx + (1.0-theta)*(k_b*(1-(c_1*0.5))*v3)*dx - (theta*source1(r,v3) + (1.0-theta)*source1(r,v3))   

t = t_start

rFile = XDMFFile('density.xdmf')
uFile = XDMFFile('displacement.xdmf')
i = 0
while t < t_end :
    
    print 'time =', t
    solve(F==0,u_v_r)
    u,v,r = u_v_r.split(True)
    u0.assign(u)
    r0.assign(r)
    
    
    t += dt
    
    i += 1
    
    rFile.write(r, t)
    uFile.write(u, t)
uFile.close()
rFile.close()








