import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sympy import *
x, y, u, z, ϵ = symbols("x y u z ϵ")

# choose approximation point
α = 0.4
β = 0.98
ρ = 0.9
u0 = (α*β)**(1/(1-α))

# Define the F function
expr = 1/(exp(z)*u**α - x) - β*α*exp(ρ*z + ϵ)*x**(α-1)/(exp(ρ*z + ϵ)*x**(α) - y)

# Find the derivatives
Fy = lambdify([x, y, u, z, ϵ],diff(expr,y))
Fx = lambdify([x, y, u, z, ϵ],diff(expr,x))
Fu = lambdify([x, y, u, z, ϵ],diff(expr,u))
Fz = lambdify([x, y, u, z, ϵ],diff(expr,z))
Fϵ = lambdify([x, y, u, z, ϵ],diff(expr,ϵ))
Fyy = lambdify([x, y, u, z, ϵ],diff(expr,y,y))
Fxx = lambdify([x, y, u, z, ϵ],diff(expr,x,x))
Fuu = lambdify([x, y, u, z, ϵ],diff(expr,u,u))
Fzz = lambdify([x, y, u, z, ϵ],diff(expr,z,z))
Fϵϵ = lambdify([x, y, u, z, ϵ],diff(expr,ϵ,ϵ))
Fyx = lambdify([x, y, u, z, ϵ],diff(expr,y,x))
Fyu = lambdify([x, y, u, z, ϵ],diff(expr,y,u))
Fyz = lambdify([x, y, u, z, ϵ],diff(expr,y,z))
Fyϵ = lambdify([x, y, u, z, ϵ],diff(expr,y,ϵ))
Fxu = lambdify([x, y, u, z, ϵ],diff(expr,u,x))
Fxϵ = lambdify([x, y, u, z, ϵ],diff(expr,x,ϵ))
Fxz = lambdify([x, y, u, z, ϵ],diff(expr,x,z))
Fzϵ = lambdify([x, y, u, z, ϵ],diff(expr,z,ϵ))

# Compute first order policy function
Δ = lambda x, y, u, z, ϵ: Fx(x, y, u, z, ϵ)**2 - 4*Fy(x, y, u, z, ϵ)*Fu(x, y, u, z, ϵ)
xu = lambda x, y, u, z, ϵ: (-Fx(x, y, u, z, ϵ) - (Δ(x, y, u, z, ϵ))**0.5)/(2*Fy(x, y, u, z, ϵ))

# Compute second order policy function
xuu = lambda x, y, u, z, ϵ: - (Fyy(x, y, u, z, ϵ)*xu(x, y, u, z, ϵ)**4. + 2.* Fyx(x, y, u, z, ϵ)*xu(x, y, u, z, ϵ)**3 + \
                      2.* Fyu(x, y, u, z, ϵ)*xu(x, y, u, z, ϵ)**3. + Fxx(x, y, u, z, ϵ)*xu(x, y, u, z, ϵ)**2 + \
                      2.* Fux(x, y, u, z, ϵ)*xu(x, y, u, z, ϵ) + Fuu(x, y, u, z, ϵ))/ \
                        (Fy(x, y, u, z, ϵ)*xu(x, y, u, z, ϵ)**2. + Fy(x, y, u, z, ϵ)*xu(x, y, u, z, ϵ) + \
                         Fx(x, y, u, z, ϵ))

Φ = lambda k: xu(y0,x0,u0)*(k - u0) + 0.5*xuu(y0,x0,u0)*(k - u0)

# set up a grid
min = u0-0.1
max = u0 + 0.1
nstep = 50
ugrid = np.linspace(min, max, num=nstep)
ygrid = np.zeros(nstep)
trueval = np.zeros(nstep)

#  get solutions at each point on the grid
for i in range(0, nstep):
    # find exact solutions
    # create anonymous function
    ygrid[i] = Φ(ugrid[i])
    trueval[i] = α*β*ugrid[i]**α


# plot results
plt.plot(ugrid, ygrid, 'k-')
plt.plot(ugrid, trueval, 'b-')
plt.legend(['Exact','2nd'], loc=1)
plt.title('Policy function in BM model')
plt.xlabel('K')
plt.ylabel('Investment')
plt.show()