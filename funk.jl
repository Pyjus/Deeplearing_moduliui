module funk

using Random
using Flux
using DataFrames
using Statistics

export data_partition
export data_expansion_RNN

function data_partition(data, at = 0.9)
    n = size(data,1)
    idx = Random.shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
end

function data_meanForDiscrete(data)
    df = DataFrame(x=data[:,1], y=data[:,7])
    sort!(df, :x)
    uni = unique!(df[:,1])
    l = length(uni)
    M = rand(l)
    for i in 1:l
      M[i] = mean(Matrix(groupby(df, :x)[i])[:,2])
    end
    # plot(df[:,1],df[:,2]); plot!(uni,M)
    return M
end

function data_expansion_RNN(time_series)
  function vecit!(V,seq)
    for row in eachrow(seq)
      push!(V,vec(row))
    end
  end

  SEQ = Array{Array{Float32}}(undef,0); seq = time_series[1:end-1];   vecit!(SEQ,seq)
  LAB = Array{Float32}(undef,0);        lab = time_series[2:end]; LAB = vcat(LAB,lab)

  for i in 1:length(time_series)-2
    seq = hcat(seq,circshift(time_series,-i)[1:end-i])[1:end-1,:]
    vecit!(SEQ,seq)
    lab = time_series[i+2:end]
    LAB = vcat(LAB,lab)
  end

  return zip(SEQ,LAB)
  # return Flux.DataLoader((SEQ,LAB); batchsize=10, shuffle=true);
end

end


# using LsqFit
# # data_train, data_test = funk.data_partition(data_tran,0.9);
# # serialize("train_test.dat", (data_train, data_test))
# data_train, data_test = deserialize("train_test.dat");
# X, Y = data_train[:,1:6], data_train[:,7];
# Xtest, Ytest = data_test[:,1:6], data_test[:,7];
# function multimodel(x, p)
#   p[1]*x[:,1]+p[2]*x[:,2]+p[3]*x[:,3]+p[4]*x[:,4]+p[5]*x[:,5]+p[6]*x[:,6]
# end
# fit = LsqFit.curve_fit(multimodel, X, Y, ones(6));
# Plots.plot([-1.5:0.01:1.5],[-1.5:0.01:1.5]);
# Plots.scatter!(Ytest,multimodel(Xtest,fit.param));

# --------------------------------------------------------------------------------------- #


# --------------------------------------------------------------------------------------- #