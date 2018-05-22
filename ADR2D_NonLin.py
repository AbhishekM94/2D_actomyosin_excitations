"""
Code to check Advection Diffusion Reaction equation with analytic solutions
posed as a Nonlinear Variational problem using Newton solver. 
"""
"""
One variable Advection_Diffusion equation

ddt(u) + (cx*ddx + cy*ddy)u = ddx(Dx*ddx + Dy*ddy)u + S(x0,y0)

in a rectangular grid with a velocity field
c, where grad(c) = 0 and c = 0 on x=0 line and y=0 line.
cx= c0*x, cy= -c0*y and Dx=D0*(c0*x)^2, Dy=D0*(c0*y)^2
S(x0,y0) = 0 for source1		: exact soln - exsol1(t)
         = 1 at all time for source2	: exact soln - exsol2(t)

Domain of solution : x,y >= 0 i.e. the first quadrant. We used a finite domain only
so there will be finite size effects.
-------------------------------------------------------------------
Detailed Reference :
http://www.sciencedirect.com/science/article/pii/S0307904X99000050
-------------------------------------------------------------------
 

"""

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from dolfin import *
import numpy
import os
from scipy import pi, linspace, loadtxt, meshgrid, exp, cos, sqrt, power


# to clear terminal and delete data from previous run
os.system('mkdir -p data')
os.system('mkdir -p result')
os.system('rm data/*.vtu')
os.system('rm data/*.pvd')
os.system('rm result/*.png')
os.system('clear')

#--------------------------------------------------------------------------------------------
# System size definition 
# system = (x1, x1+Lx) x (y1, y1+Ly)
Lx = 16.0; Ly = 16.0
x1 = 0.0001; y1 = 0.0001

#--------------------------------------------------------------------------------------------
# Create mesh and define function space
dx0 = dx1 = 0.1
nx = int(Lx/dx0) 
ny = int(Ly/dx1)
mesh = RectangleMesh(Point(x1, y1), Point(x1+Lx, y1+Ly), nx, ny,"right/left")
V = FunctionSpace(mesh, 'Lagrange', 1)

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
t_start = 0.1
dt = 0.0005
t_end = t_start + 50*dt

c0 = 1.0
D0 = 1.0
x0 = 4.02
y0 = 8.02

theta = Constant(0.5)

#-------------------------------------------------------
# Velocity field

cx = Expression("c0*x[0]", c0=c0, degree = 1)
cy = Expression("-c0*x[1]", c0=c0, degree = 1)
vex = Expression(("cx" , "cy"), cx=cx, cy=cy, degree = 1)
v_norm = interpolate(Expression("sqrt(cx*cx + cy*cy)", cx=cx, cy=cy, degree = 1),V)
VV = VectorFunctionSpace(mesh, 'Lagrange', 1)
c = interpolate(vex, VV)

# to check velocity field 
#plot(c, interactive=True,title = "vel_field")


#------------------------------------------------------
#------------ EXACT SOLUTION --------------------------
# this function will blow-up in case x,y=0/t_start=0/D0=0/c0=0/x0,y0=0 !
t = t_start

# for source1 i.e. f=0
def exsol1(t):
    p = Expression("sqrt( log(x[0]/x0)*log(x[0]/x0) + log(x[1]/y0)*log(x[1]/y0) )/c0", x0=x0, y0=y0, c0=c0, degree = 1) 

    uext1 = Expression("pow( x[0]*y0/(x[1]*x0) , 1/(2*c0*D0) )/(4*pi*D0*c0*c0*t * sqrt(x0*y0*x[0]*x[1]) )",
    c0=c0, D0=D0, pi=pi, x0=x0, y0=y0, t=t ,degree = 1)
    uext2 = Expression("exp( (-p*p -2*(1+c0*c0*D0*D0)*t*t)/(4*D0*t))", p=p, c0=c0, D0=D0, t=t, degree = 1)

    uext = Expression("uext1*uext2", uext1=uext1, uext2=uext2, degree = 1)

    ue = interpolate(uext, V)
    return ue

# for source2 i.e. f=1 at x0,y0 at all t
#-------------------------------------------
# JIT compiled expression for Bessel function
code = '''
#include <math.h>
#include <boost/math/special_functions/bessel.hpp>
using boost::math::cyl_bessel_i;
using boost::math::cyl_bessel_j;
using boost::math::cyl_bessel_k;
using boost::math::cyl_neumann;

namespace dolfin {
    class MyFun : public Expression
    {
        double c0,D0,x0,y0;
        public:
            MyFun(): Expression() {};
        void eval(Array<double>& values, const Array<double>& x) const {
            double f = ((sqrt( logf(x[0]/x0)*logf(x[0]/x0) + logf(x[1]/y0)*logf(x[1]/y0) )/c0)*sqrt(1.0+c0*c0*D0*D0))/(sqrt(2.0)*D0) ;
            values[0] = cyl_bessel_k(0,f);
        }

        void update(double _c0, double _D0, double _x0, double _y0) {
            c0 = _c0;
            D0 = _D0;
            x0 = _x0;
            y0 = _y0;
        }
    };
}'''

uext2=Expression(code, degree = 1)
uext2.update(c0,D0,x0,y0)
#ss = interpolate(ext2,V)
#plot(ss, interactive=True)

def exsol2(t):
    
    uext1 = Expression("pow( x[0]*y0/(x[1]*x0) , 1/(2*c0*D0) )/(2*pi*D0*c0*c0*t * sqrt(x0*y0*x[0]*x[1]) )",
    c0=c0, D0=D0, pi=pi, x0=x0, y0=y0, t=t, degree = 1)
    
    
    uext = Expression("uext1*uext2", uext1=uext1, uext2=uext2, degree = 1)

    ue = project(uext, V)
    return ue


# choose the exact solution    <<<< CHOICE TO BE MADE HERE >>>>
exsol = exsol2


#------------ INITIAL CONDITION -----------------------
def gaussianIC():
    gus = Expression("exp(-((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0))/(eps*eps) )", x0=x0, y0=y0, eps=0.2, degree = 1)
    gic = interpolate(gus,V)
    return gic

# Plot the exact solution at t=t_start
ic = exsol(t)
#plot(ic, interactive=True, title="Initial_condition")

#------------------------------------------------------------------------------
# Variational Formulation
#------------------------------------------------------------------------------
# Define trial and test function and solution at previous time-step
u = Function(V)
v = TestFunction(V)
u0 = Function(V)
du = TrialFunction(V)

#---------------------------------------------------------------
# constructing the diffusion tensor

dim = 2
W = TensorFunctionSpace(mesh, 'Lagrange', 1, shape=(dim,dim))
D = interpolate(Expression((("D0*c0*c0*x[0]*x[0]", "0"),("0", "D0*c0*c0*x[1]*x[1]")), c0=c0, D0=D0, degree = 1), W)

# definition after weak formulation-----------------------------
# Diffusion
def diff_term(u, v):
    return (-inner(dot(D,grad(u)), grad(v))) * dx

# Advection
def adv_term1(u, v):
    return -inner(grad(v), c*u)*dx

def adv_term2(u, v):
    return v*div(c*u)*dx

# Reaction
def source1(u, v):
    f=Expression("0.0", degree = 1)
    return f*v*dx
    
def source2(u, v):
    eps = 0.5
    f = Expression("exp(-((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0))/(eps*eps) )", x0=x0, y0=y0, eps=eps, degree = 1)
    return f*v*dx

# choose a reaction term	<<<< CHOICE TO BE MADE HERE >>>>
source = source2

#---------------------------------------------------------------

u0 = ic

F = (1.0/dt)*inner(u-u0,v)*dx + theta*adv_term2(u,v) + (1.0-theta)*adv_term2(u0,v) - ( theta*diff_term(u,v) + (1.0-theta)*diff_term(u0,v) ) - source(u,v)


# Jacobian
J = derivative(F, u, du)


# Setup nonlinear solver
problem = NonlinearVariationalProblem(F, u, J=J)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['newton_solver']['absolute_tolerance']=1.0e-6
prm['newton_solver']['relative_tolerance']=1.0e-5
prm['newton_solver']['maximum_iterations']=25
prm['newton_solver']['relaxation_parameter']=1.0


# define flux
def flux(u):
    return c*u - dot(D,grad(u))

def plotflux():
    flx = project( flux(u), VV)
    plot(flx, interactive = True, title="Flux Vector")

#---------------------------------------------------------------
# open file to write data

os.system('rm -f data/*.xyz')
ff = File("data/u.xyz")
ff << u

asize = (t_end-t_start)/dt +1
u_val = numpy.empty([asize, 2])


#------------------------------------------------------------------------------------------
# Time Loop
#------------------------------------------------------------------------------------------

def err_compare(u,t) :
    ue = exsol(t)
    maxdiff = abs(ue.vector().array() - u.vector().array()).max()
    print('t = %s, Maxdiff = %s' % (t, maxdiff))
    return ue

i = 0
while t < t_end :

    
#    u_err = project(u - uer, V) 	# can be ploted to see error values over the reg
    
    
    solver.solve()		#problem, u.vector()
    u0.assign(u)

    uer = err_compare(u,t)
    
    m = i % 1
    if m == 0.0 :
       plt.figure()
       plt.axes().set_aspect('equal')
       zfaces = numpy.asarray([u(cell.midpoint()) for cell in cells(mesh)])
       plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k', vmin=0, vmax=0.2)		#vmin=0, vmax=0.5
       plt.colorbar()
       kurs = "result/%i.png" % i
       plt.savefig(kurs, format='png')
       plt.close()

       #plotflux()   # to visualize flux vector fild

    Totu = assemble( u*dx )

    u_val[i,0] = t
    u_val[i,1] = Totu

    t += dt
    
    i += 1
    

# save numpy time series data in file
numpy.savetxt('u.txt', u_val)	# to check the total u conservation


