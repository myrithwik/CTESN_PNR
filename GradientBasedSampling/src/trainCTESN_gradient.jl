using Pkg
Pkg.activate(".")
# Pkg.instantiate()
ENV["GKSwstype"] = "100"
using PowerSimulationsDynamics
PSID = PowerSimulationsDynamics
using PowerSystems
using Logging
configure_logging(console_level=Logging.Error)
using Sundials
using LightGraphs
using Plots
using OrdinaryDiffEq
using QuasiMonteCarlo
using LinearAlgebra
using Surrogates
SURR = Surrogates
using CSV
using DataFrames
using Distributed
using Statistics
using Random


file_dir = joinpath(pwd(), "src", )
include(joinpath(file_dir, "models/system_models.jl"))
include(joinpath(file_dir, "ctesn_functions.jl"))
include(joinpath(file_dir, "experimentParameters.jl")) # This is where all the experimental variables are defined
include(joinpath(file_dir, "gradient_based_sampling_functions.jl")) # This is wehre all of the gradient based sampling functions are defined

sysSize = 144
testSize = 200
initial_n = 50
total_iters = 10
points_per_iter = 10
sys = build_system(sysSize)

busCap, totalGen, ibrBus, ibrGen, syncGen = getSystemProperties(sys);

gen = [gen for gen in syncGen if occursin("Trip", gen.name)][1]
genTrip = GeneratorTrip(tripTime, PSY.get_component(PSY.DynamicGenerator, sys, gen.name))

rSol, N, stateIndex, stateLabels, simStep, resSize, res_time = simulate_reservoir(sys, maxSimStep, genTrip, gen.name); # Simulate system and use solution to drive reservoir
rSol = reduce(hcat, rSol.(simStep))
numSteps = length(simStep);
freqIndex = 1:length(syncGen) - 1

ibr_zones = [ibrBus[0 .< ibrBus .< 40], ibrBus[40 .< ibrBus .< 80], ibrBus[80 .< ibrBus .< 120], ibrBus[120 .< ibrBus .< 160]]

LB = [0.098, 0.099, 0.097, 0.096, 0.0985, 0.0995, 0.0975, 0.0965]
UB = [0.702, 0.704, 0.706, 0.708, 0.398, 0.396, 0.394, 0.392]

trainSamples = QuasiMonteCarlo.sample(initial_n, LB, UB, QuasiMonteCarlo.LatinHypercubeSample()) # Generate quasi-random samples from interior of parameter space
testParams = QuasiMonteCarlo.sample(testSize, LB, UB, QuasiMonteCarlo.SobolSample()) # Sample parameter sapce to generate test samples

actual = simSystem!.(Ref(sys), eachcol(testParams), Ref(ibr_zones), Ref(busCap), Ref(totalGen), Ref(simStep), Ref(genTrip))

trainParams, Wouts, surr = nonlinear_mapping!(sys, busCap, ibr_zones, trainSamples, totalGen, rSol, stateIndex, simStep, genTrip); # Get RBF weights, trainParams, that map r(t) to x(t)
D = SURR._construct_rbf_interp_matrix(surr.x, first(surr.x), surr.lb, surr.ub, surr.phi, surr.dim_poly, surr.scale_factor, surr.sparse)
betaSurr = RadialBasis(trainParams, Wouts, LB, UB, rad=cubicRadial) # Build RBF that maps parmaters, p, to trainParams 

predict = nonlinear_predict.(eachcol(testParams), Ref(surr), Ref(betaSurr), Ref(D), Ref(numSteps))
predict = [transpose(p) for p in predict]

rmse = zeros(testSize, total_iters + 1)
comp_time = zeros(total_iters)

rmse[:, 1] = [sqrt.(mean(sum((predict[i] - actual[i][stateIndex, :]).^2, dims=2))) for i in 1:testSize]

for i = 1:total_iters
    num_eigen_values = size(betaSurr.LB)
    comp_time[i] = @elapsed newtrainSamples = gradient_based_sampling(betaSurr, sample_points, testParams, points_per_iter, num_eigen_values)

    newtrainParams, newWouts, surr = nonlinear_mapping!(sys, busCap, ibr_zones, newtrainSamples, totalGen, rSol, stateIndex, simStep, genTrip); # Get RBF weights, trainParams, that map r(t) to x(t)
    add_point!(betaSurr, newtrainParams, newWouts)

    predict = nonlinear_predict.(eachcol(testParams), Ref(surr), Ref(betaSurr), Ref(D), Ref(numSteps))
    predict = [transpose(p) for p in predict]

    rmse[:, i + 1] = [sqrt.(mean(sum((predict[i] - actual[i][stateIndex, :]).^2, dims=2))) for i in 1:testSize]
end 

CSV.write("results/gradient/predict_rmse.csv", DataFrame(rmse, :auto), header=false)
CSV.write("results/gradient/comp_time.csv", DataFrame(reshape(comp_time, total_iters, 1), :auto), header=false)