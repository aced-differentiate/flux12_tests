using Pkg
Pkg.activate("/home/rkurchin/CGCNN/MP/") # replace with path to environment that has Flux 12 but is otherwise the same
using CSV
using DataFrames
using Serialization
using SparseArrays
using Random, Statistics
using Flux
using Flux: @epochs
using Flux: Optimiser, ExpDecay
using Flux: gradient, glorot_uniform
using ChemistryFeaturization
using SimpleWeightedGraphs
using AtomicGraphNets
using DelimitedFiles
using Distributed

function cgcnn_train(args)
    println(args)
    num_pts, num_conv, atom_fea_len, pool_type, crys_fea_len, num_hidden_layers, lr, features, num_bins, logspaced = args
    cutoff_radius=8.0
    max_num_nbr=12
    decay_fcn=inverse_square
    use_voronoi = true

    # basic setup
    train_frac = 0.8
    num_epochs = 10
    num_train = Int32(round(train_frac * num_pts))
    num_test = num_pts - num_train
    # where to find the data
    prop = "formation_energy_per_atom"
    id = "task_id"
    info = CSV.read(string(prop,".csv"), DataFrame)
    y = Array(Float32.(info[!, Symbol(prop)]))
    graphdir = "graphs/"
    num_features = sum(num_bins) # we'll use this later
    atom_feature_vecs, featurization = make_feature_vectors(Symbol.(features), nbins=num_bins, logspaced=logspaced)

    # shuffle data and pick out subset
    indices = shuffle(1:size(info,1))[1:num_pts]
    info = info[indices,:]
    output = y[indices]

    # read in and featurize the graphs
    inputs = AtomGraph[]
    bad_indices = Int32[]

    for r in eachrow(info)
        index = rownumber(r)
        gr_path = string(graphdir, r[Symbol(id)], ".jls")
        try
            atomgraph = deserialize(gr_path)
            feature_mat = hcat([atom_feature_vecs[e] for e in atomgraph.elements]...)
            add_features!(atomgraph, feature_mat, featurization)
            push!(inputs, atomgraph)
        catch e
            println(gr_path)
            push!(bad_indices, index)
        end
    end

    # clean up after any unreadable files...this is a stopgap for now
    sort!(bad_indices, rev=true)
    for i in bad_indices
        deleteat!(output, i)
    end

    # pick out train/test sets
    train_output = output[1:num_train]
    test_output = output[num_train+1:end]
    train_input = inputs[1:num_train]
    test_input = inputs[num_train+1:end]
    train_data = zip(train_input, train_output)

    # build the network
    model = Xie_model(num_features; num_conv=num_conv, atom_conv_feature_length=atom_fea_len, pool_type=pool_type, pooled_feature_length=crys_fea_len, num_hidden_layers=num_hidden_layers)

    # define loss function
    loss(x,y) = Flux.Losses.mse(model(x), y)
    evalcb() = @show(mean(loss.(test_input, test_output)))
    start_err = evalcb()

    # train
    opt = ADAM(lr)
    _, train_time, mem, _, _ = @timed @epochs num_epochs Flux.train!(loss, params(model), train_data, opt, cb=Flux.throttle(evalcb, 5))

    end_err = evalcb()

    loss_mae(x,y) = Flux.Losses.mae(model(x),y)
    end_mae = mean(loss_mae.(test_input, test_output))
    flush(stdout)
    start_err, end_err, end_mae, train_time
end
