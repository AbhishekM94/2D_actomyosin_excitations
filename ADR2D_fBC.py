"""
2D Nonlinear Advection_Diffusion equation in a rectangular grid with Periodic/Dirichlet Boundary 
conditions
ddt(u) + c.(ddx + ddy)u = D*(d2dx2 + d2dy2)u  + reaction terms (binding and unbinding)

"""

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from dolfin import *
import numpy
import os
from scipy import pi, linspace, loadtxt, meshgrid, exp , cos, sqrt
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob

# to clear terminal and delete data from previous run
os.system('mkdir -p data')
os.system('mkdir -p result')
os.system('clear')
os.system('rm data/*.vtu')
os.system('rm data/*.pvd')
os.system('rm result/*.png')

#--------------------------------------------------------------------------------------------

Lx = 1.0; Ly = 1.0
x1 = 0.0; y1 = 0.0

#class PeriodicBoundary(SubDomain):
## Periodic BC
#   # Left boundary is "target domain" G
#    def inside(self, x, on_boundary):
#        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
#        return bool((near(x[0], x1) or near(x[1], y1)) and 
#                (not ((near(x[0], x1) and near(x[1], y1+Ly)) or 
#                        (near(x[0], x1+Lx) and near(x[1], y1)))) and on_boundary)
#
#    def map(self, x, y):
#        if near(x[0], x1+Lx) and near(x[1], y1+Ly):
#            y[0] = x[0] - Lx
#            y[1] = x[1] - Ly
#        elif near(x[0], x1+Lx):
#            y[0] = x[0] - Lx
#            y[1] = x[1]
#        else:   # near(x[1], 1)
#            y[0] = x[0]
#            y[1] = x[1] - Ly



#--------------------------------------------------------------------------------------------
# Create mesh and define function space
nx = ny = 60
mesh = RectangleMesh(Point(x1, y1), Point(x1+Lx, y1+Ly), nx, ny,"right/left")
V = FunctionSpace(mesh, 'Lagrange', 1)# constrained_domain=PeriodicBoundary())

#VV = VectorFunctionSpace(mesh, 'Lagrange', 1)

# meshgrid params for plotting
n = mesh.num_vertices()
d = mesh.geometry().dim()

# Create the triangulation
mesh_coordinates = mesh.coordinates().reshape((n, d))
triangles = numpy.asarray([cell.entities(0) for cell in cells(mesh)])
triangulation = tri.Triangulation(mesh_coordinates[:, 0],
                                  mesh_coordinates[:, 1],
                                  triangles)


#--------------------------------------------------------------------------------------------
# parameters for the problem

# Time-stepping parameters
t_start = 0.0
dt = 0.01
t_end = 200*dt
#parameters for reaction terms
D = Constant(0.0)
theta = Constant(0.5)
epsilon = Constant(1*10**(-6))
k_b =  Constant(0.4)                # abhishek : these parameters were chosen from Deb Sankar's 1D fortran code
k_u0 = Constant(0.35)                # suggest changes necessary for 2D code     
c_1 = Constant(0.10)
alpha = Constant(3.0)
theta = Constant(0.5)   

#-------------------------------------------------------
# constant velocity
c = as_vector([0.00001,0])	


# Velocity field

'''	
vex = Expression( "x[0]*x[1]" )
vform = interpolate(vex,V)
c = nabla_grad(vform)


vex = Expression(("x[1]" , "-x[0]"))
v_norm = interpolate(Expression("sqrt(x[1]*x[1] + x[0]*x[0])"),V)
c = interpolate(vex, VectorFunctionSpace(mesh, 'Lagrange', 1))
c = c/v_norm
'''

#plot(c, title='velocity_field', interactive=True)


#------------------------------------------------------

# Initial condition
px = Constant(0.2)
py = Constant(0.5)
ic = Expression("4*exp(-((x[0]-px)*(x[0]-px) + (x[1]-py)*(x[1]-py))/0.008 )", degree = 1, px=px, py=py)
#-------------------------------------------------------
# BOUNDARY CONDITION
#neumann BC : no flux

# Dirichlet BC : Absorbing boundary, u(at boundary) = 0
# Define boundary conditions
ubc = Constant(0.0)
def ubc_boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, ubc, ubc_boundary)


#--------------------------------------------------------
#Variational Calculations
u = TrialFunction(V)
v = TestFunction(V)
u0 = Function(V)


# defination after weak formulation
# Diffusion
def diff_term(u, v):
    return D*(-inner(grad(u), grad(v)) ) * dx

# Advection
def adv_term(u, v):
    return v*dot(c,grad(u))*dx
    
# Reaction
"Reaction terms chosen from Deb Sankar's one dimensional model A : will it change for 2D?" 
def source2(u,v):
    f = Expression("k_b*(1-(c_1*epsilon))",epsilon = epsilon, k_b = k_b, c_1 = c_1, degree = 1)
    return f*v*dx

def source1(u,v):
    f = Expression("k_u0*exp(alpha*epsilon)",k_u0 = k_u0, alpha = alpha, epsilon = epsilon, degree = 1)
    return f*u*v*dx
#---------------------------------------------------------------


F = (1.0/dt)*inner(u-u0,v)*dx + theta*adv_term(u,v) + (1.0-theta)*adv_term(u0,v) - ( theta*diff_term(u,v) + (1.0-theta)*diff_term(u0,v) ) + source1(u,v) - source2(u,v)  

a = lhs(F)
L = rhs(F)

u = Function(V)
u.interpolate(ic)


#plot(u)
#interactive()

# define flux
def flux(u):
    return c*u - D*grad(u)
#---------------------------------------------------------------
# open file to write data

os.system('rm -f data/*.xyz')
ff = File("data/u.xyz")
ff << u

asize = (t_end-t_start)/dt +1
u_val = numpy.empty([asize, 3])


#------------------------------------------------------------------------------------------
# Time Loop
#------------------------------------------------------------------------------------------

t = t_start

i = 0
while t < t_end :
    print 'time =', t
    
    u0.assign(u)

    solve(a==L,u,bc)

    plt.figure()
    zfaces = numpy.asarray([u(cell.midpoint()) for cell in cells(mesh)])
    plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k', vmin=0, vmax=2)
    plt.colorbar()
    kurs = "result/%i.png" % i
    plt.savefig(kurs, format='png')
    plt.close()

    TotFlux = assemble( div(flux(u))*dx )
    Totu = assemble( u*dx )

    u_val[i,0] = t
    u_val[i,1] = Totu
    u_val[i,2] = TotFlux

    print 'time =', t
    print 'total u =', Totu

    ff << u

    t += dt
    
    i += 1
    



### Visualization

def show_movie_with_slider(L, times):

    z=loadtxt(L[0])
    N=int(sqrt(len(z[:,0])))
    x=z[:,0].reshape((N,N))
    y=z[:,1].reshape((N,N))
    u=z[:,2].reshape((N,N))
    #ue=x		#A*exp(-((x-px)*(x-px) + (y-py)*(y-py))/(2*sigmaSq))

    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlim(x1,x1+Lx)
    ax.set_ylim(-0.1,2.1)
    
    lineN2,=ax.plot(x[N/2], u[N/2], 'c-')
    #lineN2_exact,=ax.plot(x[N/2], ue[N/2], 'c-', label='x=L/2')

    ax.legend(loc=9)

    divider = make_axes_locatable(ax)
    sax = divider.append_axes("bottom",size="3%",pad=0.5,
            axisbg='lightgray')
    slider = Slider(sax, r'$t$', times.min(), times.max(),
            valinit=times.min())

    # update function
    def update(val):
        t = int(slider.val/dt)
        z=loadtxt(L[t])
        u=z[:,2].reshape((N,N))
        #ue=exp(-times[t])*ue

        lineN2.set_ydata(u[N/2])
        #lineN2_exact.set_ydata(ue[N/2])
        
        plt.draw()
    slider.on_changed(update)

    plt.show()
    

L=glob.glob('data/u000*.xyz')
L.sort()
times=linspace(t_start, t_end, len(L))
show_movie_with_slider(L, times)

















