# ConvexSwitch - Julia Package

[![Build Status](https://travis-ci.org/YeeJeremy/ConvexSwitch.jl.svg?branch=master)](https://travis-ci.org/YeeJeremy/ConvexSwitch.jl)

[![Coverage Status](https://coveralls.io/repos/YeeJeremy/ConvexSwitch.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/YeeJeremy/ConvexSwitch.jl?branch=master)

[![codecov.io](http://codecov.io/github/YeeJeremy/ConvexSwitch.jl/coverage.svg?branch=master)](http://codecov.io/github/YeeJeremy/ConvexSwitch.jl?branch=master)

## Authors

Juri Hinz and Jeremy Yee

## About

This Julia package provides a method in which convex switching systems
can be numerically solved. Please submit any issues through GitHub or 
my email (jeremyyee@outlook.com.au). We have also implemented these
algorithms in the *R* language. Please see my GitHub page.

Note: The code will be multi-threaded at a later date.

## Problem Setting

A convex switching system is basically a Markov decision process with:
* a finite number of time points
* a Markov process consisting of:
      1. a controlled Markov chain with a finite number of possible realizations (positions) 
      2. a continuous process that evolves linearly i.e. **X<sub>t+1</sub> = W<sub>t+1</sub> X<sub>t</sub>**
      where **W<sub>t+1</sub>** is a matrix with random entries
* reward functions that are convex and Lipschitz continuous in the continuous 
  process
* a finite number of actions

This Julia package approximates all the value functions in the Bellman
recursion and also computes their lower and upper bounds.  The
following code demonstrates this.

## Example: Bermuda Put Option - Value Function Approximation

Let us consider the valuation of a Bermuda put option with strike
price **40** and expiry date of **1** year. The underlying asset price
follows a geometric Brownian motion. We assume the option is
exercisable at 51 evenly spaced time points, including one at
beginning and one at the end of the year. We first set our parameters.

~~~
# Load Packages
using Distributions
using ConvexSwitch

# Parameters
rate = 0.06 ## Interest rate
step = 0.02 ## Time step between decision epochs
vol = 0.2 ## Volatility of stock price process
ndec = 51 ## Number of decision epochs
~~~

The following then sets the transition probabilities for the controlled Markov
chain. If the positions are fully controllable, we can use a matrix instead.

~~~
# Control Matrix
fullcontrol = true
if fullcontrol
    control = ones(Array{Int64}(2, 2));
    control[2, 1] = 2;
else
    control = zeros(Array{Float64}(2, 2, 2))
    control[1, 1, 1] = 1
    control[1, 2, 1] = 1
    control[2, 1, 2] = 1
    control[2, 2, 1] = 1
end
~~~

Next, we define an equally spaced grid ranging from 10 to 100 and
comprising 181 grid points.

~~~
# Grid
gnum = 181
grid = ones(gnum, 2)
grid[:, 2] = linspace(10, 100, gnum)
~~~

Introduce the derivative of the reward and scrap functions:

~~~
# Derivative of scrao
strike = 40
function Scrap(state::Array{Float64, 1}, p::Int64)
    result = zero(Array{Float64, 1}(2))
    if p == 2  # if option alive at the end exercise
        if state[2] <= strike
            result[1] = strike
            result[2] = -1
            discount = exp(-rate * step * (ndec - 1))
            result = discount * result
        end
        
    end
    return(result)
end

# Derivative of reward
function Reward(time::Int64, state::Array{Float64, 1}, p::Int64, a::Int64)
    result = zero(Array{Float64,1}(2))
    if (p == 2) & (a == 2)
        discount = exp(rate * step * (ndec - 1 - time)) 
        result = discount * Scrap(state, 2)
    end
    return(result)
end
~~~

Finally, we define the sampling of disturbances **(W<sub>t</sub>)**
which the code assumes tho be identically distributed across time.

~~~
# Disturbances
dnum = 10000
disturb = zeros(Array{Float64}(2, 2, dnum))
disturb[1, 1, :] = 1
q = linspace(0, 1, dnum + 2)[2:(dnum + 1)]
disturb[2, 2, :] = exp((rate - 0.5 * vol^2) * step + vol * sqrt(step) * quantile(Normal(), q))
weight = ones(dnum) / dnum
~~~

We encapsulate all the model information into an object.

~~~
# Create the immutable css type
css = CSS(grid, weight, disturb, control, Reward, Scrap, ndec)
~~~


Now we are ready to perform the Bellman recursion using the fast method.

~~~
# Location of the random entries in the disturb matrix
rIndex = [2 2]

# Function to contruct the search structure
function search(x::Matrix{Float64})
    NearestNeighbors.KDTree(x)
end
    
# Bellman
DMat = PermMat(css, rIndex, search)  ## construct fast matrices
bellman = Bellman(css, DMat)
~~~

The immutable object **bellman** contains our approximations of the
value functions, continuation value functions and prescribed policy at
each grid point. The value function of the option can be plotted using
the following.

~~~
# Plot
value = sum(bellman.value[:, :, 2, 1] .* grid, 2)
using PyPlot
plot(grid[:,2], value)
xlabel("Stock Price")
ylabel("Option Value")
~~~

## Example: Bermuda Put Option - Bounds

Having computed the function approximations above, we can now calculate
the bounds on the value of the option. This is performed using a pathwise
dynamic programming approach. Suppose that the current price of the underlying stock 
is **36**.

~~~
start = [1.; 36.] ## starting state
~~~

We then generate a set of sample paths for the price and disturbances for 
the nested simulation. They will be used to calculate the required martingale
increments.

~~~
# Path Disturbances
srand(1234)
pathnum = 500
path_disturb = zeros(Array{Float64}(2, 2, ndec - 1, pathnum))
path_disturb[1, 1, :, :] = 1
path_disturb[2, 2, :, :] = exp((rate - 0.5 * vol^2) * step +
                               vol * sqrt(step) * rand(Normal(), (ndec - 1) * pathnum))
# Generate paths
path = Path(start, path_disturb)

# Subsimulation disturbances
subsimnum = 500
subsim = zeros(Array{Float64}(2, 2, pathnum, subsimnum, ndec - 1))
subsim[1, 1, :, :, :] = 1
subsim[2, 2, :, :, :] = exp((rate - 0.5 * vol^2) * step +
                            vol * sqrt(step) * rand(Normal(), (ndec - 1) * pathnum * subsimnum))
subsim_weight = ones(subsimnum) / subsimnum

# Martingale increments
mart = Mart(bellman, path, subsim, subsim_weight, search, css.grid)
~~~

Having obtained the prescribed policy for our sample paths using

~~~
## Get path policy
path_policy = PathPolicy(css, bellman, path)
~~~

the pathwise methods is then used to obtain estimates for the lower
and upper bound for the value of the option.

~~~
bounds = OptimalBounds(css, path, mart, path_policy)

## Lower and upper bound estimates and standard errors
println(bounds.lowermean[2], " (", bounds.lowerse[2], ") \n",
        bounds.uppermean[2]," (", bounds.lowerse[2], ")")
~~~

The above gives us:
* **4.4747491804414965 (0.005471056812184923)** for the lower bound
* **4.4996462680908030 (0.005471056812184923)** for the upper bound


## Conclusion

Tighter bounds can be achived by either improving the value function approximations by using:
* a more dense grid
* a larger disturbance sampling

or by obtaining more suitable martingale increments through 
* a larger number of sample paths
* a larger number of nested simulations

The above methods are very versatile and we have employed it for the purpose of option valuation,
optimal resource extraction, partially observable Markov decision processes and optimal 
battery control.
