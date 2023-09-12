#= 
minimalESN:
- Julia version: 1.6.2
- Author: rithwikmylavarapu
- Date: 2022-03-05 =#

#= 
A minimalistic Echo State Networks demo with Mackey-Glass (delay 17) data
in "plain" Julia.
from https://mantas.info/code/simple_esn/
(c) 2020 Mantas LukoÅ¡eviÄius
Distributed under MIT license https://opensource.org/licenses/MIT =#

using DelimitedFiles
import Random
using Plots
using LinearAlgebra
using Base

@time begin
# load the data
trainLen = 1000
testLen = 4000
initLen = 100
data = readdlm("SampleESN/MackeyGlass_t17.txt")

# plot some of it
# sample plot
p1 = plot(data[1:2000], leg=false, title="A sample of data", reuse=false)

# generate the ESN reservoir
inSize = outSize = 1
resSize = 4000
a = 0.3 # leaking rate


Random.seed!(42)
Win = (rand(resSize, 1 + inSize) .- 0.5) .* 1
W = rand(resSize, resSize) .- 0.5
# normalizing and setting spectral radius
print("Computing spectral radius...")
rhoW = maximum(abs.(eigvals(W)))
println("done.")
W .*= (1.25 / rhoW)

# allocated memory for the design (collected states) matrix
X = zeros(1 + inSize + resSize, trainLen - initLen)
# set the corresponding target matrix directly
Yt = transpose(data[initLen + 2:trainLen + 1])

##### Maybe this is the break in the code where I should train again with new W
# run the reservoir with the data and collect X
x = zeros(resSize, 1)
for t = 1:trainLen
    u = data[t]
    global x = (1 - a) .* x .+ a .* tanh.(Win * [1;u] .+ W * x)
    if t > initLen
        X[:,t - initLen] = [1;u;x]
    end
end

# train the output by ridge regression
reg = 1e-8  # regularization coefficient
# direct equations from texts:
# X_T = transpose(X)
# Wout = Yt*X_T * inv(X*X_T + reg*I)
# using Julia backslash solver:
Wout = transpose((X * transpose(X) + reg * I) \ (X * transpose(Yt)))

##### Maybe I change W based on this Wout and go back up to retrain

# run the trained ESN in a generative mode. no need to initialize here,
# because x is initialized with training data and we continue from there.
Y = zeros(outSize, testLen)
u = data[trainLen + 1]
print(size(x))
println(size(Wout))
for t = 1:testLen
	global x = (1 - a) .* x .+ a .* tanh.(Win * [1;u] .+ W * x)
	y = Wout * [1;u;x]
	Y[:,t] = y
	###### Look Into this Part of Code
	# generative mode:
	 global u = y
	# this would be a predictive mode:
	# global u = data[trainLen + t + 1]
	### Understand this above Block
end

# compute MSE for the first errorLen time steps
errorLen = testLen
# mse = sum( abs2.(data[trainLen + 2:trainLen + errorLen + 1] .-
# 	Y[1,1:errorLen]) ) / errorLen
mse = abs.(data[trainLen + 2:trainLen + errorLen + 1] .-
    Y[1,1:errorLen])
# println("MSE = $mse")

# top10  = partialsort(mse, floor(Int64, errorLen / 10), rev=true)
sorted = sort(mse, rev=true)
top5 = sorted[1:floor(Int64, errorLen / 20)]
println("Worst Error: ", top5[1])
println("Average Worst Error: ", Statistics.mean(top5))

# plot some signals
p2 = plot(data[trainLen + 2:trainLen + testLen + 1], c=RGB(0, 0.75, 0), label="Target signal", reuse=false)
plot!(transpose(Y), c=:blue, label="Free-running predicted signal")
title!("Target and generated signals y(n) starting at n=0")

# p3 = plot(transpose(X[1:20,1:200]), leg=false)
# title!("Some reservoir activations x(n)")

p3 = plot(sorted[1:floor(Int64, errorLen / 20)])
title!("Error for Each Point)")

p4 = bar(transpose(Wout), leg=false)
title!("Output weights Wout")

# display all 4 plots
plot(p1, p2, p3, p4, size=(1200, 800))
end