### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ 16e00373-71a1-4283-8c16-b324dbe43ee7
# setup for local environment used in git
begin
	import Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()
end

# ╔═╡ 559162b0-1b34-4cca-be73-cdc4e3836152
# load the packages
begin
	using MetaGraphs
	using GeometricFlux, Flux, GraphSignals, JLD2, PlutoUI, Graphs, SparseArrays
	using Flux: @functor, onecold, onehotbatch
	using Statistics: mean
	using Random, LinearAlgebra
	import Zygote
end

# ╔═╡ 0ff5c895-082d-4bbe-9d2f-30b45bf81bb4
# setup PyCall for loading some outer dataset
begin
	# the is my environment set in Conda, which is equiped with the library in colab notebook
	ENV["PYTHON"] = raw"C:\tools\Anaconda3\envs\cs224w\python.exe"
	Pkg.build("PyCall")
	using PyCall
end

# ╔═╡ 5f6a3e05-3174-4b10-b210-5f8ae6134bb5
using GraphPlot

# ╔═╡ 6d26b4a0-86f7-11ec-0ade-39f3fbd0ee00
md"""
# CS224W - Notebook 4

This notebook is focused on heterogeneous graphs.
"""

# ╔═╡ ae1d9dba-5a8d-4024-80a8-148d5010a409
md"""
## Setup 
"""

# ╔═╡ da8830f5-ca23-40d1-99fc-d19d409408b0
# global device is set to cpu, one could modify it to `gpu`, and load CUDA package
device = cpu

# ╔═╡ c33aa990-831e-4b1b-9699-48044e7d0421
md"""
## 1) Heterogeneous Graph

Among the julia pacakges, the `MetaGraphs.jl` and `MetaGraphsNext.jl` packages are built for heterogeneous graphs. In this notebook, we will use `MetaGraphs.jl`.
"""

# ╔═╡ 7bb3fd65-f5e6-4b19-87cc-6f86fd0128b1
md"""
In this part, we will work with karate club graph.
"""

# ╔═╡ 57b9018e-1a8d-4e90-b93a-39b22b32190a
G_karate = smallgraph(:karate)

# ╔═╡ 0e82c52d-8680-4d3b-8241-8d32f64367df
# also inject it with the node clubs
clubs = ["Mr. Hi", "Mr. Hi", "Mr. Hi", "Mr. Hi", "Mr. Hi", "Mr. Hi", "Mr. Hi", "Mr. Hi", "Mr. Hi", "Officer", "Mr. Hi", "Mr. Hi", "Mr. Hi", "Mr. Hi", "Officer", "Officer", "Mr. Hi", "Mr. Hi", "Officer", "Mr. Hi", "Officer", "Mr. Hi", "Officer", "Officer", "Officer", "Officer", "Officer", "Officer", "Officer", "Officer", "Officer", "Officer", "Officer", "Officer"]

# ╔═╡ 034f544f-38d4-4d07-8653-a6c755124250
md"""
Now we will construct a heterogeneous graph using MetaGraph.
"""

# ╔═╡ 355f8b57-6005-4389-a261-eaf41f4ab0cd
G_karate_meta = MetaGraph(G_karate)

# ╔═╡ c6ce9afc-1ac9-41c2-98c9-b2375a1a0591
md"""
### Question 1.1: Assigning Node Type and Node Features

In `MetaGraphs.jl`, we use `set_prop!` or `set_indexing_prop!` functions to add these attributions.
"""

# ╔═╡ dda8586a-c9f1-410a-8b77-c802c252452e
club_map = Dict(enumerate([if c == "Mr. Hi"; 0 else 1 end for c in clubs]))

# ╔═╡ 26e3a1d3-1378-4b4b-8c48-6427ead0a4c1
begin
	function assign_node_types(G, community_map::Dict)
		for (v, _) in community_map
			set_prop!(G, v, :type, :club)
		end
	end
	
	function assign_node_labels(G, community_map::Dict)
		for (v, attr) in community_map
			set_prop!(G, v, :club, attr)
		end
	end

	function assign_node_features(G)
		for v in vertices(G)
			set_prop!(G, v, :feature, ones(Float32, 5))
		end
	end

	# inject features and other attributes
	for v in vertices(G_karate_meta)
		clear_props!(G_karate_meta, v)
	end
	assign_node_types(G_karate_meta, club_map)
	assign_node_labels(G_karate_meta, club_map)
	assign_node_features(G_karate_meta)
end

# ╔═╡ e9458fe1-f818-47cd-b076-857d8ea24d9f
let
	node_id = 21
	props(G_karate_meta, node_id)
end

# ╔═╡ 460de337-3a97-44cd-87b3-9f3f8e025d83
md"""
### Question 1.2: Assigning Edges Types

`set_prop!` can also add attributes to edges. 

Here, we set `e0` to edges within club "Mr. Hi", `e1` to edges within club "Officier", and `e2` between them.
"""

# ╔═╡ c3baf04e-2742-48a2-8e03-e71a64576b8c
begin
	function assign_edge_types(G, community_map)
		for e in collect(edges(G))
			e_i = e.src
			e_j = e.dst
			edge_type = community_map[e_i] + community_map[e_j]

			if edge_type == 0
				set_prop!(G, e, :type, :e0)
			elseif edge_type == 1
				set_prop!(G, e, :type, :e1)
			else
				set_prop!(G, e, :type, :e2)
			end
		end
	end

	for e in collect(edges(G_karate_meta))
		clear_props!(G_karate_meta, e)
	end
	assign_edge_types(G_karate_meta, club_map)
end

# ╔═╡ f25d2e03-c8d0-4a14-8184-62c35564dc81
let
	edge_idx = 16
	n1, n2 = 1, 32

	edge = collect(edges(G_karate_meta))[edge_idx]
	(props(G_karate_meta, edge), props(G_karate_meta, n1), props(G_karate_meta, n2))
end

# ╔═╡ 86e3d0a3-8961-43fa-a8b6-95d01cd4625f
md"""
### Heterogeneous Graph Visulization
"""

# ╔═╡ ed8dcfb3-f80f-43fe-b67e-e386ae3fd06c
gplot(
	G_karate_meta, 
	nodelabel=vertices(G_karate_meta), 
	#edgelabel=[props(G_karate_meta, e)[:type] for e in edges(G_karate_meta)],
	#EDGELABELSIZE=3.0,
	nodefillc=[if club_map[v] == 0; "turquoise" else "red" end for v in vertices(G_karate_meta)],
	edgestrokec=[begin
		t = props(G_karate_meta, e)[:type]
		if t == :e0
			"lightgray"
		elseif t == :e1
			"green"
		else
			"black"
		end
	end for e in edges(G_karate_meta)],
)

# ╔═╡ 00d6ab54-2faa-4e9c-a4e5-106ecb9dd02a
md"""
### Question 1.3 How Many nodes are of each type.

Since we only use such one package, not like python used networkx and deepSNAP in colab notebook, we are keeping working on the same data structure.
"""

# ╔═╡ a351faa0-1487-46da-88b5-0ef67cb8743f
begin
	function get_nodes_per_type(G)
		num_nodes_n0 = filter_vertices(G, :club, 0) |> length∘collect
		num_nodes_n1 = filter_vertices(G, :club, 1) |> length∘collect

		num_nodes_n0, num_nodes_n1
	end

	get_nodes_per_type(G_karate_meta)
end

# ╔═╡ 397fffd4-4443-4739-b347-c0a59a183485
md"""
### Question 1.4 Message Types - How many edges are of each message type
"""

# ╔═╡ d5d51aa2-17d6-4b60-b26b-23701a458439
begin
	function get_num_message_edges(G)
		[(t, (length∘collect∘filter_edges)(G, :type, t)) for t in [:e0, :e1, :e2]]
	end

	function get_edges_types(G)
		map(edges(G)) do e
			props(G, e)[:type]
		end |> (sort∘collect∘Set)
	end

	get_num_message_edges(G_karate_meta)
end

# ╔═╡ 811fbb91-61b3-4337-b958-8608a01c2e83
md"""
### Question 1.5 Dataset Splitting - How many nodes are in each dataset split?

It seems that no splitting function was implemented in `MetaGraphs.jl`, so here we need to implement it ourselves. It is not hard.
"""

# ╔═╡ cea96b77-2d00-48a3-a68f-fbfc32fa3172
begin
	function graph_dataset(G)
		vs = vertices(G)

		# to keep the index ordering
		dataset = hcat([props(G, v)[:feature] for v in vs]...)

		dataset, vs
	end

	function split_dataset(vs, split_ratio)
		train, val, test = split_ratio

		vs = shuffle(vs)
		train_num = (train * length(vs)) |> Int∘floor
		val_num = (val * length(vs)) |> Int∘floor
		test_num = length(vs) - train_num - val_num

		vs[1:train_num], vs[train_num+1:train_num+val_num], vs[end-test_num+1:end]
	end

	function compute_dataset_split_counts(dataset)
		res = Dict()
		for (k, v) in dataset
			res[k] = length(v)
		end

		res
	end
	
	dataset, vs = graph_dataset(G_karate_meta)
	dataset_train, dataset_val, dataset_test = split_dataset(vs, [0.8, 0.0, 0.2])
	datasets = Dict("train"=>dataset_train, "val"=>dataset_val, "test"=>dataset_test)

	dataset_splits = compute_dataset_split_counts(datasets)
end

# ╔═╡ dbc6fdc5-7038-4cf6-b435-9e75770cf24a
md"""
## 2) Heterogeneous Graph Node Property Prediction
"""

# ╔═╡ 6208cf1d-fe42-43a6-a082-5803b6aa43c7
md"""
In `GeometricFlux.jl`, we didnot find some layers implementations for heterogeneous graph, so we need to implement it.
"""

# ╔═╡ 9ec59996-8966-48e7-b2bf-424286547eaf
md"""
### Implement `HeteroGNNConv` layer

Similar to the notebook in colab, we use GraphSAGE model here for every single type of nodes.
"""

# ╔═╡ ed58c7ad-1cc5-4c6b-b51c-863a00b1922d
is_directed(G_karate_meta)

# ╔═╡ 8ab2db23-b6d2-4423-9f7f-e5c5c853c974
begin
	struct HeteroGNNConv{M<:AbstractMatrix, N, V}
		Ws::M
		Wd::M
		type_name::N
		type_value::V
	end

	# these codes take GeometricFlux.jl as template
	function HeteroGNNConv(in_channel_src, in_channel_dst, out_channels, types::Pair;
                 init=Flux.glorot_normal)
	    Ws = init(out_channels, in_channel_src)
		Wd = init(out_channels, in_channel_dst)
	    HeteroGNNConv(Ws, Wd, types...)
	end

	@functor HeteroGNNConv
	Flux.trainable(l::HeteroGNNConv) = (l.Ws, l.Wd)

	function (hl::HeteroGNNConv)(mg::MetaGraph, x::AbstractMatrix)
		# this conv layer works for symmetric graph, while not test for asymmetric ones
		message_edges = hcat(map(filter_edges(mg, hl.type_name, hl.type_value)) do e
			[e.src, e.dst]
		end...)
		adj = Zygote.ignore() do
			sparse(
				message_edges[1, :], 
				message_edges[2, :], 
				ones(Int32, size(message_edges, 2)), 
				nv(mg), nv(mg))
		end

		ds = Set()
		if !is_directed(mg)
			adj = adj + adj'
			ds = Set(message_edges)
		else
			ds = Set(message_edges[2, :])
		end
		idx = Zygote.ignore() do 
			t = zeros(Float32, nv(mg))  # to modify
			t[collect(ds)] .= 1.0f0
			t
		end

			
		Ã = Zygote.ignore() do
			Float32.((adj ./ (x -> if x == 0; 1.0 else x end).(sum(adj, dims=2)))')
		end
		
		(hl.Ws * x .+ hl.Wd * x * Ã, idx)
	end
	
	function Base.show(io::IO, l::HeteroGNNConv)
	    out_dim, src_dim = size(l.Ws)
		_, dst_dim = size(l.Wd)
	    print(io, "HeteroGNNConv(($src_dim, $dst_dim) => $out_dim")
	    print(io, ")")
	end
end

# ╔═╡ 6cf7ad42-c648-4dce-a62f-9eb8237fbf4c
md"""
Then update the aggragating rules into one of multiple node types:

$h_v^{(l)[m]} = W^{(l)[m]} \cdot \text{CONCAT} \Big( W_d^{(l)[m]} \cdot h_v^{(l-1)}, W_s^{(l)[m]} \cdot AGG(\{h_u^{(l-1)}, \forall u \in N_{m}(v) \})\Big)$

(from notebook 5 in colab), where $AGG$ is a mean function. Now, based on the layer, we will implement `HeteroGNNConvWrapper` layer. Since $W^{(l)[m]}$ could be divided into two submatrices, and multiply it on the inner components, we directly use one matrix for each part, and then sum them.

$h_v^{(l)[m]} = W_l^{(l)[m]} \cdot h_v^{(l-1)} + W_r^{(l)[m]} \cdot AGG(\{h_u^{(l-1)}, \forall u \in N_{m}(v) \})$

Then we will construct the wrapper layer, which could be simplify mean of all the `HeteroGNNConv` layers, or the announced semantic label attention:

$e_{m} = \frac{1}{|V_{d}|} \sum_{v \in V_{d}} q_{attn}^T \cdot tanh \Big( W_{attn}^{(l)} \cdot h_v^{(l)[m]} + b \Big)$
"""

# ╔═╡ 52d9d26a-4769-4a2a-9594-872ef4fad1a1
md"""
### Initialize Heterogeneous GNN Layers
"""

# ╔═╡ 4ecffc5f-6a73-40d6-9b07-afb7bf983327
function generate_convs(mg::AbstractMetaGraph, hidden_size)
	map(get_edges_types(mg)) do v
		HeteroGNNConv(hidden_size, hidden_size, hidden_size, :type=>v)
	end
end

# ╔═╡ 8e3eaf8f-2869-452b-913c-053ff2fbf6e8
md"""
### HeteroGNN
"""

# ╔═╡ 728b1923-c18d-42cc-94cc-6b4d6fed5d7e
begin
	struct HeteroGNN
		convs1
		bns1
		convs2
		bns2
		mps
		mg::AbstractMetaGraph
	end

	@functor HeteroGNN
	Flux.trainable(g::HeteroGNN) = (
		Flux.trainable(g.convs1),
		Flux.trainable(g.bns1),
		Flux.trainable(g.convs2),
		Flux.trainable(g.bns2),
		Flux.trainable(g.mps)
	)
	
	# for simpliciy, we assume all features have same dimension
	function HeteroGNN(mg, in_dim, hidden_dim, out_dim)
		in_convs = map(get_edges_types(mg)) do v
			HeteroGNNConv(in_dim, in_dim, hidden_dim, :type=>v)
		end
		
		HeteroGNN(
			HeteroGNNConvWrapper(in_convs, :attn, hidden_dim=>hidden_dim),
			BatchNorm(hidden_dim, relu),
			HeteroGNNConvWrapper(generate_convs(mg, hidden_dim), :attn, hidden_dim=>hidden_dim),
			BatchNorm(hidden_dim, relu),
			Dense(hidden_dim, out_dim),
			mg
		)
	end

	function (m::HeteroGNN)(x)
		m.convs1(m.mg, x) |>
			m.bns1 |>
			xx -> m.convs2(m.mg, xx) |>
			m.bns2 |>
			m.mps
	end
end

# ╔═╡ ceaa1aa5-8a52-4e80-9a50-a9fa3d440e3c
begin
	struct HeteroGNNConvWrapper{M<:AbstractMatrix}
		convs
		aggr
		W::M
		b::M
		q::M
	end

	@functor HeteroGNNConvWrapper
	function Flux.trainable(w::HeteroGNNConvWrapper)
		(w.W, w.b, w.q, Flux.trainable.(w.convs)...)
	end

	function HeteroGNNConvWrapper(convs, aggr, dims::Pair; init=Flux.glorot_normal)
		in_dim, out_dim = dims
		
		HeteroGNNConvWrapper(convs, aggr, init(out_dim, in_dim), init(out_dim, 1), init(out_dim, 1))
	end

	function (hw::HeteroGNNConvWrapper)(mg::AbstractMetaGraph, x)
		h_i = map(hw.convs) do l
			l(mg, x)
		end
		hs = cat([h[1] for h in h_i]..., dims=3)  # features × n_vertices × n_layers
		idx = hcat([h[2] for h in h_i]...) |> x -> reshape(x, (1, size(x)...)) # 1 × n_v × n_layers
		
		# the mean algorithm
		if hw.aggr == :mean
			 dropdims(sum(hs .* idx, dims=3), dims=3) ./ 
			 	dropdims(sum(idx, dims=3), dims=3)
		end

		# the semantice label attention
		
		e_m = hw.q' * tanh.(hw.W * reshape(hs, size(hs, 1), :) .+ hw.b)  # 1 × (n_nodes*n_layers)
		e_m = sum(reshape(e_m, size(hs, 2), :) .* dropdims(idx, dims=1), dims=1)  # n_layers
		e_m = e_m ./ sum(dropdims(idx, dims=1), dims=1)  # n_layers

		α_md = exp.(e_m) .* dropdims(idx, dims=1)  # n_nodes × n_layers
		α_md = α_md ./ sum(α_md, dims=2)  # normalization
		dropdims(sum(hs .* reshape(α_md, 1, size(α_md)...), dims=3), dims=3)
	end

	function Base.show(io::IO, hw::HeteroGNNConvWrapper)
		_, in_dim = size(hw.W)
		print(io, "HeteroGNNConvWrapper( => $in_dim")
		print(io, ", total ", length(hw.convs), " layers")
	    print(io, ")")
	end
end

# ╔═╡ 4a27f85f-a40f-4456-9183-e0e8aba7e021
md"""
### Training and Testing

Here we test for the karate dataset.
"""

# ╔═╡ d9688347-77a9-4dea-9c02-69d9512f91bf
begin
	Base.@kwdef struct Args
		lr::Float32 = 0.01
		epochs::Int = 10
	end

	args = Args()
end

# ╔═╡ 789f7b96-3254-4f71-8656-33e5130f7744
function evaluate(model, data, loss_fn, idx=Colon())
	x, y = data

	ŷ = model(x)

	loss = loss_fn(ŷ[:, idx], y[:, idx])
	acc = mean(onecold(ŷ[:, idx], 0:1) .== onecold(y[:, idx], 0:1))

	loss, acc
end

# ╔═╡ e4b377f8-30cc-4699-88ab-2ecafdee4365
begin
	model = HeteroGNN(G_karate_meta, size(dataset, 1), 5, 2)

	opt = ADAM(args.lr)

	loss_fn(ŷ, y) = Flux.Losses.logitcrossentropy(ŷ, y)
	labels = map(1:nv(G_karate_meta)) do v
		props(G_karate_meta, v)[:club]
	end |> x -> onehotbatch(x, 0:1)
	
	with_terminal() do
		for epoch in 1:args.epochs
			ps = params(model)

			gs = gradient(ps) do
				ŷ = model(dataset)

				loss_fn(ŷ[:, dataset_train], labels[:, dataset_train])
			end

			Flux.update!(opt, ps, gs)

			loss, acc = evaluate(model, (dataset, labels), loss_fn, dataset_train)
			println("Epoch ", epoch, ", Loss: ", loss, ", Acc: ", acc)
		end

		println("Test: ")
		loss, acc = evaluate(model, (dataset, labels), loss_fn, dataset_test)
		println("Loss: ", loss, ", Acc: ", acc)
	end
end

# ╔═╡ Cell order:
# ╠═6d26b4a0-86f7-11ec-0ade-39f3fbd0ee00
# ╟─ae1d9dba-5a8d-4024-80a8-148d5010a409
# ╠═16e00373-71a1-4283-8c16-b324dbe43ee7
# ╠═559162b0-1b34-4cca-be73-cdc4e3836152
# ╠═0ff5c895-082d-4bbe-9d2f-30b45bf81bb4
# ╠═da8830f5-ca23-40d1-99fc-d19d409408b0
# ╠═c33aa990-831e-4b1b-9699-48044e7d0421
# ╟─7bb3fd65-f5e6-4b19-87cc-6f86fd0128b1
# ╠═57b9018e-1a8d-4e90-b93a-39b22b32190a
# ╠═0e82c52d-8680-4d3b-8241-8d32f64367df
# ╟─034f544f-38d4-4d07-8653-a6c755124250
# ╠═355f8b57-6005-4389-a261-eaf41f4ab0cd
# ╟─c6ce9afc-1ac9-41c2-98c9-b2375a1a0591
# ╠═dda8586a-c9f1-410a-8b77-c802c252452e
# ╠═26e3a1d3-1378-4b4b-8c48-6427ead0a4c1
# ╠═e9458fe1-f818-47cd-b076-857d8ea24d9f
# ╟─460de337-3a97-44cd-87b3-9f3f8e025d83
# ╠═c3baf04e-2742-48a2-8e03-e71a64576b8c
# ╠═f25d2e03-c8d0-4a14-8184-62c35564dc81
# ╟─86e3d0a3-8961-43fa-a8b6-95d01cd4625f
# ╠═5f6a3e05-3174-4b10-b210-5f8ae6134bb5
# ╠═ed8dcfb3-f80f-43fe-b67e-e386ae3fd06c
# ╟─00d6ab54-2faa-4e9c-a4e5-106ecb9dd02a
# ╠═a351faa0-1487-46da-88b5-0ef67cb8743f
# ╟─397fffd4-4443-4739-b347-c0a59a183485
# ╠═d5d51aa2-17d6-4b60-b26b-23701a458439
# ╟─811fbb91-61b3-4337-b958-8608a01c2e83
# ╠═cea96b77-2d00-48a3-a68f-fbfc32fa3172
# ╟─dbc6fdc5-7038-4cf6-b435-9e75770cf24a
# ╟─6208cf1d-fe42-43a6-a082-5803b6aa43c7
# ╟─9ec59996-8966-48e7-b2bf-424286547eaf
# ╠═ed58c7ad-1cc5-4c6b-b51c-863a00b1922d
# ╠═8ab2db23-b6d2-4423-9f7f-e5c5c853c974
# ╟─6cf7ad42-c648-4dce-a62f-9eb8237fbf4c
# ╠═ceaa1aa5-8a52-4e80-9a50-a9fa3d440e3c
# ╟─52d9d26a-4769-4a2a-9594-872ef4fad1a1
# ╠═4ecffc5f-6a73-40d6-9b07-afb7bf983327
# ╟─8e3eaf8f-2869-452b-913c-053ff2fbf6e8
# ╠═728b1923-c18d-42cc-94cc-6b4d6fed5d7e
# ╟─4a27f85f-a40f-4456-9183-e0e8aba7e021
# ╠═d9688347-77a9-4dea-9c02-69d9512f91bf
# ╠═789f7b96-3254-4f71-8656-33e5130f7744
# ╠═e4b377f8-30cc-4699-88ab-2ecafdee4365
