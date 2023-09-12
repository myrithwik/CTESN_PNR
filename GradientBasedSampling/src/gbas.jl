using Pkg
Pkg.activate(".")
# Pkg.instantiate()
using Surrogates
using Plots
using SurrogatesPolyChaos
using PolyChaos
using LinearAlgebra

# Define the 2d Rosenbrock function
function Rosenbrock2d(x)
    x1 = x[1]
    x2 = x[2]
    return (1 - x1)^2 + 100 * (x2 - x1^2)^2
end

# Make a helper function to plot any given function
function plotFunction(xys, zs, lower_bound, upper_bound, f, plotTitle)
    num_samples = length(xys)
    x, y = lower_bound[1]:upper_bound[1], lower_bound[2]:upper_bound[2]
    p1 = surface(x, y, (x1, x2) -> f((x1, x2)))
    xs = [xy[1] for xy in xys]
    ys = [xy[2] for xy in xys]
    p2 = contour(x, y, (x1, x2) -> f((x1, x2)))
    scatter!(xs, ys)
    plot(p1, p2, title=plotTitle)
end

# Plot the True Rosenbrock Function
num_samples = 100
lb = [0.0,0.0]
ub = [8.0,8.0]
xys = sample(num_samples, lb, ub, SobolSample())
zs = Rosenbrock2d.(xys)
plotFunction(xys, zs, lb, ub, Rosenbrock2d, "True Rosenbrock2d Function")

# Train and plot PloynomialChaos Surrogate model to fit the rosenbrock2d 
num_samples = 100
lb = [0.0,0.0]
ub = [8.0,8.0]
xys = sample(num_samples, lb, ub, SobolSample())
zs = Rosenbrock2d.(xys)
poly_surrogate = PolynomialChaosSurrogate(xys, zs,  lb, ub)
plotFunction(xys, zs, lb, ub, poly_surrogate, "Ploy Surrogate on Rosenbrock")

# Evaluating the baseline model off of 20 test samples
num_test = 25
lb = [0.0,0.0]
ub = [8.0,8.0]
x_test = sample(num_test, lb, ub, SobolSample())
y_predicted = poly_surrogate.(x_test)
y_true = Rosenbrock2d.(x_test)
mse_poly = norm(y_true - y_predicted, 2) / num_test

# Building a sampleset of intial points using Sobol Sampling and the rest with gradient based adaptive sampling
init_sample_num = 17
lb = [0.0,0.0]
ub = [8.0,8.0]
xys = sample(init_sample_num, lb, ub, SobolSample())
zs = Rosenbrock2d.(xys)
# sample the remaining points (83) using gradient based adaptive sampling lola technique

# Train and plot a PolynomialChaosSurrogate model with this new sampleset
# Evaluating the new model