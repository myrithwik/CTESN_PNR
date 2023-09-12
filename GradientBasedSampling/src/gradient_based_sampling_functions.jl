"""
    calc_hessian(dimension, candidate, sample_points)

Return the minimum distance and hessian of an rbf at a specific point

Inputs
    betaSurr -> Radial Basis Function that is the surrogate model for the function
    dimension -> int (number of dimensions of the problem)
    candidate -> Tuple(Float64) candidate point to calculate the hessian at
    sample_points -> list[Tuple(Float64)] list of points already sampled for the model

Outputs
    min_dist -> minimum distance of the candidate point to the sample_points
    hessian -> hessian of the rbf at the candidate point

Example
    min_dist, hessian = calc_hessian(2, (6.7, 3.5), [(2.3, 3.4), (3.4, 4.5), (4.5, 5.6)])
"""
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

"""
    gradient_based_sampling

Gather a list of new sampling points based on hte gradient of the surrogate model at various candidate points

Inputs
    betaSurr -> radial basis function: surrogate model to predict the function
    sample_points -> list[Tuple(Float64)] list of points already sampled for the model
    testParams -> list[Tuple(FLoat64)] list of candidate points for new samples
    points_per_iter -> Int number of points to add to the sample set per iteration
    num_eigen_values -> Int number of largest eigen values to consider while calculating the K values

Outputs
    new_samples -> list[Tuple(Float64)] list of new sample points to be added to the sample set for the model

Example
    new_samples = (radialBasisFunction, [(2.3, 3.4), (3.4, 4.5), (4.5, 5.6)], [(3.3, 2.4), (2.4, 3.5), (2.5, 3.6)], 2, 2)
"""
function gradient_based_sampling(betaSurr, sample_points, testParams, points_per_iter, num_eigen_values)
    dimension = size(betaSurr.LB) # dimension of the model
    candidate_eval = [] # a list of evaluations for the candidate sampling points
    for candidate in testParams
        # calculate the hessian and the minimum distance of the model at the candiate point
        min_dist, hessian = calc_hessian(betaSurr, dimension, candidate, sample_points)

        if min_dist == 0 # If the candidate point is already in the sample set then skip the point
            continue
        end

        # find the k largest eigenvalues
        k = num_eigen_values
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
    # Sort the candidate evaluation list to get the top points_per_iter points
    indices = partialsortperm(candidate_eval, 1:points_per_iter, rev=true)
    new_samples = []
    # Add build a tuple of the new sample
    for index in indices
        sample = []
        for dim in 1:dimension
            append!(sample, testParams[index][dim])
        end

        sample = Tuple(sample)
        # Push each new samples into the new_samples list
        push!(new_samples, sample)
    end
    return new_samples
end