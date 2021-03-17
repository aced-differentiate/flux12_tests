# testing with Flux 11

flush(stdout)

println("starting!")

using Distributed

addprocs(parse(Int, ENV["SLURM_NTASKS"])-1)

@everywhere include("train_fcn_flux11.jl")
using DataFrames

#nums_pts = [20000, 20000, 20000]
nums_pts = [120, 120, 120]
nums_conv = [2]
atom_fea_lens = [80]
pool_types = ["mean"]
crys_fea_lens = [40]
nums_hidden_layers = [1]
lrs = [0.002]
features = [["Group", "Row", "X", "Atomic radius", "Block"]]
nums_bins = [[18, 9, 10, 10, 4]]
logspaceds = [[false, false, false, true, false]]

param_sets = [p for p in Iterators.product(nums_pts, nums_conv, atom_fea_lens, pool_types, crys_fea_lens, nums_hidden_layers, lrs, features, nums_bins, logspaceds)]

println("parameter sets built")

results = pmap(cgcnn_train, param_sets)

output = DataFrame(num_pts=Int[], start_err=Float32[], end_err=Float32[], mae=Float32[], train_time=Float32[])

for i in 1:prod(size(param_sets))
    params = param_sets[i]
    result = results[i]
    row = (params[1], result...)
    push!(output, row)
end

CSV.write("flux11.csv", output)

