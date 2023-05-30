cd(@__DIR__)
  using Pkg
  Pkg.activate(".")
  using Serialization
  using LinearAlgebra
  using Plots
  using DataFrames
  using Statistics
  using Random
  using Flux
  include("funk.jl")
  using .funk

# ======================================================================================= #
# /////////////////////////////////////////////////////////////////////////////////////// #
# =================================== Data Extaction ==================================== #

import XLSX
xf = XLSX.readxlsx("Real estate valuation data set.xlsx")
sh = xf["pirmas"]
sh[:]

data = Float32.(sh[:][2:end,2:8]); sum(isnan.(data))
titles = reshape(String.(sh[:][1,2:8]), 1, 7)
# "X1 transaction date"
# "X2 house age"
# "X3 distance to the nearest MRT station"
# "X4 number of convenience stores"
# "X5 latitude"
# "X6 longitude"
# "Y house price of unit area"
titles[1:6] .= ["Y vs X1: Transaction date",
"Y vs X2: House age",
"Y vs X3: Distance to MRT station",
"Y vs X4: Number of stores",
"Y vs X5: Latitude",
"Y vs X6: Longitude"]

# ======================================================================================= #
# /////////////////////////////////////////////////////////////////////////////////////// #
# ================================ Data Standardization ================================= #

n = size(data,1)
μ = vec(mean(data,dims=1))
σ = vec( std(data,dims=1))
q = quantile!(data[:,7], [0.5, 0.9])

for i in 1:n
  data[i,:] .= (data[i,:]-μ)./σ
end

μ = vec(mean(data,dims=1))
σ = vec( std(data,dims=1))
c = round.(cor(data,dims=1),digits=2)

# ======================================================================================= #
# /////////////////////////////////////////////////////////////////////////////////////// #
# ================================= Data transformation ================================= #
# 1. Get acquainted with the data. Plot it and try different transformations.
# Try to understand the possible relationship between input and output.

data_tran = copy(data)

scatter(data[:,1],data[:,7])
# data_tran[:,1] .= sin.(data[:,1]*2)
# scatter(sin.(data[:,1]*2),data[:,7])

scatter(data[:,2],data[:,7])
scatter((data[:,2].-0.7).^2,data[:,7])
λ = 5
data_tran[:,2] .= [(λ*x+1)^(1/λ) for x in (data[:,2].-0.7).^2]
scatter(data_tran[:,2],data[:,7])

scatter(data[:,3],data[:,7])
scatter((data[:,3].+1).^(-1/2),data[:,7])
λ = 5
data_tran[:,3] .= [(λ*x+1)^(1/λ) for x in (data[:,3].+1).^(-1/2)]
scatter(data_tran[:,3],data[:,7])

scatter(data[:,4],data[:,7])

scatter(data[:,5],data[:,7])
scatter(data[:,6],data[:,7])
scatter(data[:,5],data[:,6])

Plots.scatter(data[:,1:6],data_tran[:,7],layout = (3,2), title=titles, titlefont = font(12), size=(700,700), label = false)
μ = vec(mean(data_tran,dims=1))
σ = vec( std(data_tran,dims=1))
for i in 1:n
  data_tran[i,:] .= (data_tran[i,:]-μ)./σ
end
Plots.scatter(data_tran[:,1:6],data_tran[:,7],layout = (3,2), title=titles, titlefont = font(12), size=(700,700), label = false)


# using CairoMakie
# using PairPlots

# # The simplest table format is just a named tuple of vectors.
# # You can also pass a DataFrame, or any other Tables.jl compatible object.
# table = (;
#   x₁ = data[:,1],
#   x₂ = data[:,2],
#   x₃ = data[:,3],
#   x₄ = data[:,4],
#   x₅ = data[:,5],
#   x₆ = data[:,6],
#   y  = data[:,7],
# )

# table = (;
#   x₁ = data_tran[:,1],
#   x₂ = data_tran[:,2],
#   x₃ = data_tran[:,3],
#   x₄ = data_tran[:,4],
#   x₅ = data_tran[:,5],
#   x₆ = data_tran[:,6],
#   y  = data_tran[:,7],
# )

# pairplot(table)
# c = round.(cor(data_tran,dims=1),digits=2)

# ======================================================================================= #
# /////////////////////////////////////////////////////////////////////////////////////// #
# ========================================= PCA ========================================= #
# 2. Do the principal component analysis (PCA).

using MultivariateStats
fake = copy(data[:,1:6])
fake[:,2] .= fake[:,2].*100
data_train, data_test = funk.data_partition(fake,0.9)
data_train, data_test = funk.data_partition(data[:,1:6],0.9)

# Σ = data_train'*data_train./n;   # Covariance matrix
# U = reverse(eigvecs(Σ),dims=2);  # Eigenvectors
# e = reverse(eigvals(Σ));         # Eigenvalues (Squared error to the fitted line)
# contr = e./sum(e)*100            # Variance explained (basically e components)
# cumsum(contr)                    # Variance explained acumulated
# Xpr = data_test[:,1:6]*U;        # Projected data into PC space

model_PCA = fit(PCA, data_train'; maxoutdim=6)
y = predict(model_PCA, data_test')
# U = projection(model_PCA)' # Projection matrix (row vectors x1,x2,x3,...)
x = reconstruct(model_PCA, y)'


# ────────────────────────────────────────────────────────────────────────────────────────
#      Original data:          PC1       PC2       PC3        PC4        PC5        PC6
# ────────────────────────────────────────────────────────────────────────────────────────
# SS Loadings (Eigenvalues)  2.58235   1.03524   0.979791  0.566765   0.542521   0.129185
# Variance explained         0.442498  0.177393  0.167892  0.0971177  0.0929634  0.0221365
# Cumulative variance        0.442498  0.619891  0.787782  0.8849     0.977863   1.0
# ────────────────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────────────────
#     Transformed data:        PC1       PC2       PC3        PC4        PC5        PC6
# ────────────────────────────────────────────────────────────────────────────────────────
# SS Loadings (Eigenvalues)  2.76681   1.03887   0.887374  0.590067   0.558842   0.207613
# Variance explained         0.457356  0.171726  0.146683  0.0975385  0.0923769  0.0343185
# Cumulative variance        0.457356  0.629083  0.775766  0.873305   0.965681   1.0
# ────────────────────────────────────────────────────────────────────────────────────────

data_train, data_test = funk.data_partition(data_tran[:,1:6],0.9)
model_PCA = fit(PCA, data_train'; maxoutdim=6)
y = predict(model_PCA, data_test')
# U = projection(model_PCA)' # Projection matrix (row vectors x1,x2,x3,...)
x = reconstruct(model_PCA, y)'

# Optimize the number of principal components, i.e. determine
# how many and which components you need to get good generalization
# performance.
# >> Five will suffice, as P1+PC2+PC3+PC4+PC5 accounts for over 90% of proportion in variation in the data.

# ======================================================================================= #
# /////////////////////////////////////////////////////////////////////////////////////// #
data_train, data_test = funk.data_partition(data,0.9);                                    #
Xtest, Ytest = collect(data_test[:,1:6]'), collect(data_test[:,7]');                      #
model = Dense(6,1); loss(x, y) = Flux.mse(model(x),y);                                    #
# /////////////////////////////////////////////////////////////////////////////////////// #
# ================================== Linear Regression ================================== #

data_shift = copy(data)
interval = floor(Int, 0.1*n)
n_train = 100

Err_tr = Array{Float32}(undef,0); Err_ts = Array{Float32}(undef,0)
for cv in 1:10
data_shift = circshift(data_shift, (interval,0))
data_test = data_shift[1:interval,:]
data_train = data_shift[interval+1:end,:]

X, Y =        collect(data_train[:,1:6]'), collect(data_train[:,7]');
Xtest, Ytest = collect(data_test[:,1:6]'), collect(data_test[:,7]');
dat = Flux.DataLoader((X,Y); batchsize=10, shuffle=true);

model = Dense(6,1)
loss(x, y) = Flux.mse(model(x),y);
evalcb() = @show(loss(Xtest, Ytest))

for i in 1:n_train
  Flux.train!(loss, Flux.params(model), dat, Flux.ADAM(), cb = Flux.throttle(evalcb, 1000))
  push!(Err_tr,loss(X, Y))
  push!(Err_ts,loss(Xtest, Ytest))
end
end
Err_tr = mean(reshape(Err_tr,length(Err_tr)÷10,10),dims=2)
Err_ts = mean(reshape(Err_ts,length(Err_ts)÷10,10),dims=2)

begin
  Plots.plot([-1.5:0.01:1.5],[-1.5:0.01:1.5],label="y=ŷ",xlabel="ŷ", ylabel="y"); 
  p1 = Plots.scatter!(model(Xtest)',Ytest',label="test datapoints",title = "Linear Regression: y vs ŷ")

  l_LR = round(minimum(Err_ts),digits=2)
  Plots.plot(Err_tr[:], label = "train",xlabel="iterations", ylabel="mean squared error"); 
  annotate!((75, 0.62, Plots.text("Last error = $l_LR",10)));
  p2 = Plots.plot!(mean(Err_ts, dims=2), label = "test",title = "Linear Regression: error graph during training")

  Plots.plot(p1, p2, layout = grid(2, 1, heights=[0.5, 0.5]), size = (600,600))
end
# savefig("LR_OG.png") 

# ======================================================================================= #
# /////////////////////////////////////////////////////////////////////////////////////// #
# ================================ Non-linear Regression ================================ #

data_shift = copy(data_tran)
interval = floor(Int, 0.1*n)
n_train = 100

Err_tr = Array{Float32}(undef,0); Err_ts = Array{Float32}(undef,0)
for cv in 1:10
data_shift = circshift(data_shift, (interval,0))
data_test = data_shift[1:interval,:]
data_train = data_shift[interval+1:end,:]

X, Y =        collect(data_train[:,1:6]'), collect(data_train[:,7]');
Xtest, Ytest = collect(data_test[:,1:6]'), collect(data_test[:,7]');
dat = Flux.DataLoader((X,Y); batchsize=10, shuffle=true);

model = Dense(6,1)
loss(x, y) = Flux.mse(model(x),y);
evalcb() = @show(loss(Xtest, Ytest))

for i in 1:n_train
  Flux.train!(loss, Flux.params(model), dat, Flux.ADAM(), cb = Flux.throttle(evalcb, 1000))
  push!(Err_tr,loss(X, Y))
  push!(Err_ts,loss(Xtest, Ytest))
end
end
Err_tr = mean(reshape(Err_tr,length(Err_tr)÷10,10),dims=2)
Err_ts = mean(reshape(Err_ts,length(Err_ts)÷10,10),dims=2)

begin
  Plots.plot([-1.5:0.01:1.5],[-1.5:0.01:1.5],label="y=ŷ",xlabel="ŷ", ylabel="y"); 
  p1 = Plots.scatter!(model(Xtest)',Ytest',label="test datapoints",title = "Non-linear Regression: y vs ŷ")

  l_mLR = round(minimum(Err_ts),digits=2)
  Plots.plot(Err_tr, label = "train",xlabel="iterations", ylabel="mean squared error"); 
  annotate!((75, 0.6, Plots.text("Last error = $l_mLR",10)));
  p2 = Plots.plot!(Err_ts, label = "test",title = "Non-linear Regression: error graph during training")

  Plots.plot(p1, p2, layout = grid(2, 1, heights=[0.5, 0.5]), size = (600,600))
end
# savefig("LR_TRAN.png") 

# ======================================================================================= #
# /////////////////////////////////////////////////////////////////////////////////////// #
# ================================== Neural Regression ================================== #

data_shift = copy(data)
interval = floor(Int, 0.1*n)
n_train = 100

Err_tr = Array{Float32}(undef,0); Err_ts = Array{Float32}(undef,0)
for cv in 1:10
data_shift = circshift(data_shift, (interval,0))
data_test = data_shift[1:interval,:]
data_train = data_shift[interval+1:end,:]

X, Y =        collect(data_train[:,1:6]'), collect(data_train[:,7]');
Xtest, Ytest = collect(data_test[:,1:6]'), collect(data_test[:,7]');
dat = Flux.DataLoader((X,Y); batchsize=10, shuffle=true);

model = Chain( Dense(6,10,tanh), Dense(10,6,tanh), Dense(6,1) )
loss(x, y) = Flux.mse(model(x),y);
evalcb() = @show(loss(Xtest, Ytest))

for i in 1:n_train
  Flux.train!(loss, Flux.params(model), dat, Adam())
  push!(Err_tr,loss(X, Y))
  push!(Err_ts,loss(Xtest, Ytest))
end
end
Err_tr = mean(reshape(Err_tr,length(Err_tr)÷10,10),dims=2)
Err_ts = mean(reshape(Err_ts,length(Err_ts)÷10,10),dims=2)

begin
  Plots.plot([-1.5:0.01:1.5],[-1.5:0.01:1.5],label="y=ŷ",xlabel="ŷ", ylabel="y"); 
  p1 = Plots.scatter!(model(Xtest)',Ytest',label="test datapoints",title = "ANN on original data: y vs ŷ")

  l_NN_og = round(minimum(Err_ts),digits=2)
  Plots.plot(Err_tr, label = "train",xlabel="iterations", ylabel="mean squared error"); 
  annotate!((75, 0.4, Plots.text("Lowest error = $l_NN_og",10)));
  p2 = Plots.plot!(Err_ts, label = "test",title = "ANN: error graph during training")

  Plots.plot(p1, p2, layout = grid(2, 1, heights=[0.5, 0.5]), size = (600,600))
end
# savefig("NN_OG.png") 

# ----------------------------------- Transformed Data ----------------------------------- #

data_shift = copy(data_tran)
interval = floor(Int, 0.1*n)
n_train = 100

Err_tr = Array{Float32}(undef,0); Err_ts = Array{Float32}(undef,0)
for cv in 1:10
data_shift = circshift(data_shift, (interval,0))
data_test = data_shift[1:interval,:]
data_train = data_shift[interval+1:end,:]

X, Y =        collect(data_train[:,1:6]'), collect(data_train[:,7]');
Xtest, Ytest = collect(data_test[:,1:6]'), collect(data_test[:,7]');
dat = Flux.DataLoader((X,Y); batchsize=10, shuffle=true);

model = Chain( Dense(6,10,tanh), Dense(10,6,tanh), Dense(6,1) )
loss(x, y) = Flux.mse(model(x),y);
evalcb() = @show(loss(Xtest, Ytest))

for i in 1:n_train
  Flux.train!(loss, Flux.params(model), dat, Adam())
  push!(Err_tr,loss(X, Y))
  push!(Err_ts,loss(Xtest, Ytest))
end
end
Err_tr = mean(reshape(Err_tr,length(Err_tr)÷10,10),dims=2)
Err_ts = mean(reshape(Err_ts,length(Err_ts)÷10,10),dims=2)

begin
  Plots.plot([-1.5:0.01:1.5],[-1.5:0.01:1.5],label="y=ŷ",xlabel="ŷ", ylabel="y"); 
  p1 = Plots.scatter!(model(Xtest)',Ytest',label="test datapoints",title = "ANN on transformed data: y vs ŷ")

  l_NN_tr = round(minimum(Err_ts),digits=2)
  Plots.plot(Err_tr, label = "train",xlabel="iterations", ylabel="mean squared error"); 
  annotate!((75, 0.4, Plots.text("Lowest error = $l_NN_tr",10)));
  p2 = Plots.plot!(Err_ts, label = "test",title = "ANN: error graph during training")

  Plots.plot(p1, p2, layout = grid(2, 1, heights=[0.5, 0.5]), size = (600,600))
end
# savefig("NN_TRAN.png") 

# -------------------------------------- PCA Data --------------------------------------- #
nPCA = 4
U = projection(model_PCA)'[:,1:nPCA]
Z = data_tran[:,1:6]*U
data_shift = copy(hcat(Z,data_tran[:,7]))
interval = floor(Int, 0.1*n)
n_train = 100

Err_tr = Array{Float32}(undef,0); Err_ts = Array{Float32}(undef,0);
for cv in 1:10
data_shift = circshift(data_shift, (interval,0))
data_test = data_shift[1:interval,:]
data_train = data_shift[interval+1:end,:]

X, Y = collect(data_train[:,1:nPCA]'), collect(data_train[:,nPCA+1]');
Xtest, Ytest = collect(data_test[:,1:nPCA]'), collect(data_test[:,nPCA+1]');
dat = Flux.DataLoader((X,Y); batchsize=10, shuffle=true);

model = Chain( Dense(nPCA,10,tanh), Dense(10,6,tanh), Dense(6,1) )
loss(x, y) = Flux.mse(model(x),y);
evalcb() = @show(loss(Xtest, Ytest))

for i in 1:n_train
  Flux.train!(loss, Flux.params(model), dat, Adam())
  push!(Err_tr,loss(X, Y))
  push!(Err_ts,loss(Xtest, Ytest))
end
end
Err_tr = mean(reshape(Err_tr,length(Err_tr)÷10,10),dims=2)
Err_ts = mean(reshape(Err_ts,length(Err_ts)÷10,10),dims=2)

begin
  Plots.plot([-1.5:0.01:1.5],[-1.5:0.01:1.5],label="y=ŷ",xlabel="ŷ", ylabel="y"); 
  p1 = Plots.scatter!(model(Xtest)',Ytest',label="test datapoints",title = "ANN on PCA 4 data: y vs ŷ")

  l_NN_PCA = round(minimum(Err_ts),digits=2)
  Plots.plot(Err_tr, label = "train",xlabel="iterations", ylabel="mean squared error"); 
  annotate!((75, 0.4, Plots.text("Lowest error = $l_NN_PCA",10)));
  p2 = Plots.plot!(Err_ts, label = "test",title = "ANN: error graph during training")

  Plots.plot(p1, p2, layout = grid(2, 1, heights=[0.5, 0.5]), size = (600,600))
end
savefig("NN_PCA4.png") 

# ======================================================================================= #
# /////////////////////////////////////////////////////////////////////////////////////// #
# ================================ Recurrent Neural Net ================================= #

data_shift = copy(data_tran)
interval = floor(Int, 0.1*n)
n_train = 100

Err_tr = Array{Float32}(undef,0); Err_ts = Array{Float32}(undef,0)
for cv in 1:10
data_shift = circshift(data_shift, (interval,0))
data_test = data_shift[1:interval,:]
data_train = data_shift[interval+1:end,:]

XY_tr_exp = data_train |> funk.data_meanForDiscrete |> funk.data_expansion_RNN |> collect
XY_ts_exp = data_test  |> funk.data_meanForDiscrete |> funk.data_expansion_RNN |> collect
Xtest = [x for (x,y) in XY_ts_exp]; Ytest = [y for (x,y) in XY_ts_exp]
X     = [x for (x,y) in XY_tr_exp]; Y     = [y for (x,y) in XY_tr_exp]

model_RNN = Chain(RNN(1 => 5), Dense(5 => 1)); # Priima Vector{Float32}, o ne Float32 ar Vector{Float64}
function eval_model(seq)
  Flux.reset!(model_RNN)
  lirst([model_RNN([xi]) for xi in seq][end])
end
loss(x, y) = Flux.mse(eval_model(x), y)
evalcb() = @show(mean(loss.(Xtest, Ytest)))

for i in 1:n_train
  Flux.train!(loss, Flux.params(model_RNN), XY_tr_exp, Adam(), cb = Flux.throttle(evalcb, 100))
  push!(Err_tr,mean(loss.(X, Y)))
  push!(Err_ts,mean(loss.(Xtest, Ytest)))
end
end
Err_tr = mean(reshape(Err_tr,length(Err_tr)÷10,10),dims=2)
Err_ts = mean(reshape(Err_ts,length(Err_ts)÷10,10),dims=2)

begin
  Plots.plot([-1.5:0.01:1.5],[-1.5:0.01:1.5],label="y=ŷ",xlabel="ŷ", ylabel="y"); 
  p1 = Plots.scatter!(eval_model.(Xtest),Ytest,label="test datapoints",title = "RNN on transformed data: y vs ŷ")

  l_RNN = round(minimum(Err_ts),digits=2)
  Plots.plot(Err_tr, label = "train",xlabel="iterations", ylabel="mean squared error"); 
  annotate!((75, 0.45, Plots.text("Lowest error = $l_RNN",10)));
  p2 = Plots.plot!(Err_ts, label = "test",title = "RNN: error graph during training")

  Plots.plot(p1, p2, layout = grid(2, 1, heights=[0.5, 0.5]), size = (600,600))
end
# savefig("NN_RNN.png")


# ======================================================================================= #
# /////////////////////////////////////////////////////////////////////////////////////// #
# ===================================== RNN vs ANN ====================================== #

data_shift = copy(data_tran)
interval = floor(Int, 0.1*n)
n_train = 100

Err_tr = Array{Float32}(undef,0); Err_ts = Array{Float32}(undef,0)
for cv in 1:10
data_shift = circshift(data_shift, (interval,0))
data_test = data_shift[1:interval,:]
data_train = data_shift[interval+1:end,:]

X, Y =        collect(data_train[:,1]'), collect(data_train[:,7]');
Xtest, Ytest = collect(data_test[:,1]'), collect(data_test[:,7]');
dat = Flux.DataLoader((X,Y); batchsize=10, shuffle=true);

# model = Chain( Dense(1,10,tanh), Dense(10,6,tanh), Dense(6,1) )
model = Chain( Dense(1,5,tanh),Dense(5,1) )
loss(x, y) = Flux.mse(model(x),y);
evalcb() = @show(loss(Xtest, Ytest))

for i in 1:n_train
  Flux.train!(loss, Flux.params(model), dat, Adam())
  push!(Err_tr,loss(X, Y))
  push!(Err_ts,loss(Xtest, Ytest))
end
end
Err_tr = mean(reshape(Err_tr,length(Err_tr)÷10,10),dims=2)
Err_ts = mean(reshape(Err_ts,length(Err_ts)÷10,10),dims=2)

begin
  Plots.plot([-1.5:0.01:1.5],[-1.5:0.01:1.5],label="y=ŷ",xlabel="ŷ", ylabel="y"); 
  p1 = Plots.scatter!(model(Xtest)',Ytest',label="test datapoints",title = "ANN on transformed data: y vs ŷ")

  l_NN_t = round(minimum(Err_ts),digits=2)
  Plots.plot(Err_tr, label = "train",xlabel="iterations", ylabel="mean squared error"); 
  annotate!((75, 1.04, Plots.text("Lowest error = $l_NN_t",10)));
  p2 = Plots.plot!(Err_ts, label = "test",title = "ANN: error graph during training")

  Plots.plot(p1, p2, layout = grid(2, 1, heights=[0.5, 0.5]), size = (600,600))
end
savefig("NN_time.png") 

# 8.43+4.44+3.77
# 5.78+16.76+13+6.88+9.26+3.53+10+18+6.5+2.5+5.78+36.69+5
# 6.01+5.78
# 67.23+13.74+69.30+75.50
