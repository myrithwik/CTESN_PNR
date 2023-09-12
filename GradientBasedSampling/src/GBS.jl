using Pkg
Pkg.activate(".")
# Pkg.instantiate()
using Surrogates
using Plots
# using SurrogatesPolyChaos
using PolyChaos
using LinearAlgebra
# using Zygote
# using Flux
# using ForwardDiff

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
xys = sample(num_samples, lb, ub, Surrogates.LatinHypercubeSample())
zs = Rosenbrock2d.(xys)
plotFunction(xys, zs, lb, ub, Rosenbrock2d, "True Rosenbrock2d Function")

# Train and plot PloynomialChaos Surrogate model to fit the rosenbrock2d 
num_samples = 100
lb = [0.0,0.0]
ub = [8.0,8.0]
xys = sample(num_samples, lb, ub, Surrogates.LatinHypercubeSample())
zs = Rosenbrock2d.(xys)
poly_surrogate = PolynomialChaosSurrogate(xys, zs,  lb, ub)
plotFunction(xys, zs, lb, ub, poly_surrogate, "Ploy Surrogate on Rosenbrock")

# Evaluating the baseline model off of 20 test samples
num_test = 25
lb = [0.0,0.0]
ub = [8.0,8.0]
x_test = sample(num_test, lb, ub, Surrogates.LatinHypercubeSample())
y_predicted = poly_surrogate.(x_test)
y_true = Rosenbrock2d.(x_test)
mse_poly = norm(y_true - y_predicted, 2) / num_test

function evaluate(model)
    num_test = 100
    lb = [0.0,0.0]
    ub = [8.0,8.0]
    x_test = sample(num_test, lb, ub, Surrogates.LatinHypercubeSample())
    y_predicted = model.(x_test)
    y_true = Rosenbrock2d.(x_test)
    mse_poly = norm(y_true - y_predicted, 2) / num_test
    return mse_poly
end

# Building a sampleset of intial points using Latin Hypercube Sampling and the rest with gradient based adaptive sampling
init_sample_num = 100
lb = [0.0,0.0]
ub = [8.0,8.0]
xys = sample(init_sample_num, lb, ub, Surrogates.LatinHypercubeSample())
x = []
y = []
for pair in xys
    push!(x, pair[1])
    push!(y, pair[2])
end
parameters = [x, y]
zs = Rosenbrock2d.(xys)

# sample the remaining points (80) using model dependant gradient based adaptive sampling

# Helper function to get the first derivative of the cubic rbf
function rho_prime(r)
    return 3 * (r)^2
end

# Helper function to get the second derivative of the cubic rbf
function rho_double_prime(r)
    return 6 * r
end

function calc_hessian(betaSurr, dimension, candidate, sample_points)
    hessian = zeros(Float64, dimension, dimension)
    min_dist = Inf
    ident = Matrix(1.0I, dimension, dimension) # identity matrix for hessian calculation
    for (index, input) in enumerate(sample_points)
        diff = collect(candidate .- input)
        h = zeros(dimension, dimension)
        dist = norm(diff)
        min_dist = min(min_dist, dist) # Finding the minimum distance from the candidate point to the sample points
        if dist == 0 # If the candidate point is already in the sample set return basic values
            return 0, ident
        end

        # Calculating the hessian of the rbf at the candidate point
        h += (3 * (dist)^2) * ident
        coeff = (((6 * dist) - (3 * (dist)^2)) / dist)
        h += coeff * ((diff * transpose(diff)) / dist)
        h = (betaSurr.coeff[index] / dist) .* h
        hessian += h
    end
    return min_dist, hessian
end

# Run model dependant gradient based sampling for 8 iterations (10 new samples each iteration)
let num_iteration = 0, model_mse = 0
    while num_iteration < 8
        # Build the rbf based on the already sampled points and evlauate it
        initial_rbf = RadialBasis(xys, zs, lb, ub, rad=cubicRadial)
        model_mse = evaluate(initial_rbf)
        println(model_mse)
        dimension = size(initial_rbf.lb)[1]

        num_new_samples = 10
        candidates = sample(80, lb, ub, Surrogates.LatinHypercubeSample())
        # Make a list for the K(X) * d^2 values of all candidate points
        candidate_eval = []
        # Create the 2 x 2 identity matrix
        ident = Matrix(1.0I, dimension, dimension)
        for candidate in candidates
            min_dist, hessian = calc_hessian(initial_rbf, dimension, candidate, xys)

            if min_dist == 0
                continue
            end

            k = 2
            eigen_val = eigen(hessian)
            ev = eigen_val.values[dimension - k + 1:dimension]
            K = 0
    
            # Calculate the K value
            for e_val in ev
                K += e_val^2
            end
            K = sqrt(K)
            # If the K value is too small then set it at a small value
            if K < 10^-8
                K = 10^-8
            end
            append!(candidate_eval, K * min_dist^2) 
        end
        # Sort candidate_eval to get the indices with the top num_new_samples values in candidate_eval
        indices = partialsortperm(candidate_eval, 1:num_new_samples, rev=true)
        new_samples = []
        for index in indices
            # add top candidate points to the new sample list
            sample = []
            for dim in 1:dimension
                append!(sample, candidates[index][dim])
            end
            sample = Tuple(sample)
            # println(sample)
            push!(new_samples, sample)
        end
        
        # new_samples is the samples list to be returned
        print("Iteration Number ", num_iteration, ": ")
        println(new_samples)
        append!(xys, new_samples)
        append!(zs, Rosenbrock2d.(new_samples))
        num_iteration += 1
    end
end
# Train and plot a PolynomialChaosSurrogate model with this new sampleset
# Evaluating the new model

# TODO:
# Change initial sampling to Latin Hypercube? 
# Check the accuracy of the RBF and see if it is enough
# if not then do the algorithm
## algorithm
# Calculate the hessian of the RBF
# get the top p eigen values (p is the number of terms in the polynomial)
# Sum the squared value of all of the eigen values
# the squared sum is K
# if K < epsilon then it equals epsilon, epsilon = 10^-8
# Use the DIRECT method to solve max: K(x) * d(x)^2
# Have a helper function get the K value and then for the d(x) value
# The DIRECT optimization needs to return x points
# Update the rbf based on the new points and re-run algorithm

# REPALCE SOBOL SAMPLING WITH LATIN Hypercube
# Calculate the hessian manually
# Run the algorithm for each candidate point sampled through latin Hypercube
# calculate the hessian for each candidate point and calulcate distance for each point
# put K(x) * d(x)^2 for each point into a list and then find n max as sampling points