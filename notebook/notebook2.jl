### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ d925094a-7f4c-4bdb-ad86-a6f9d3d17d27
begin
	import Pkg
	# One may need to modify the path
	ENV["PYTHON"] = raw"C:\tools\Anaconda3\envs\cs224w\python.exe"
	Pkg.build("PyCall")
	using PyCall
end

# ╔═╡ 7ba01ad4-8019-4635-af52-b59017968630
using GeometricFlux

# ╔═╡ fa788def-e2d3-4afb-98a2-37e1b2dedac3
using Flux

# ╔═╡ 760580a3-98e4-4c77-970f-3d6dac81098a
using PlutoUI

# ╔═╡ 9b9ec843-e328-4964-ab7f-65176397d227
using Graphs

# ╔═╡ a7a463a0-ae0b-40aa-930b-4e8d33b1d8df
using GraphSignals  # for featured graph to construct GCNConv layer

# ╔═╡ 48086045-2640-477b-880b-82c47fc7bcf7
using Flux: onecold, onehotbatch

# ╔═╡ 0c6a82d7-7f55-4895-9ff8-eeb0ceb2f2a3
using LinearAlgebra: dot

# ╔═╡ d6bf7b0e-f042-453c-aab7-46ec8651a4a3
using Zygote

# ╔═╡ 2f2c2200-6895-11ec-25f5-497e6fabd0f1
md"""
# Notebook 2
"""

# ╔═╡ 914e7475-98c9-49c3-8fde-cfdc94612062
md"""
In the colab 2 of cs224w, the PyTorch Geometric was introduced. Here, we'll use the `GeometricFlux.jl` package.
"""

# ╔═╡ 93cd2223-8906-4c02-954e-4530ba26c38a
md"""
### Setup
"""

# ╔═╡ dbe21b78-581f-42e3-bd27-51b037400982
md"""
`torch_geometric.datasets` provides some graph datasets, and in `GeometrixFlux.jl`, we can load the cora dataset [(doc)](https://fluxml.ai/GeometricFlux.jl/stable/#Load-dataset-1):
"""

# ╔═╡ 71583d68-6a3c-48c0-8c78-b1c137c72c24
readdir(joinpath(pkgdir(GeometricFlux), "data"))

# ╔═╡ d713b09c-bd1f-427a-8bc2-14d99c548985
md"""
To keep the same content of colab in cs224, we can use `PyCall` load the torch package.
"""

# ╔═╡ 0a52ecbc-b016-4bc2-9975-44413456e43c
begin
	torch_geometric = pyimport("torch_geometric")
	TUDataset = torch_geometric.datasets.TUDataset
end

# ╔═╡ 2aa87ffc-574c-41cb-be8a-1bb6028c5692
# load dataset
begin
	root = "./enzymes"
	name = "ENZYMES"
	
	pyg_dataset = TUDataset(root, name)
end

# ╔═╡ 7f3cfb6b-2da0-4786-9e00-42fd6da4305d
md"""
### Question 1: What is the number of classes and number of features in the ENZYMES dataset? (5 points)
"""

# ╔═╡ ddcb0349-9983-4c5b-ae9c-af425cbc0044
pyg_dataset.num_features

# ╔═╡ 5ade533b-b6a4-45aa-a1d5-97f71f0fd5de
begin
	function get_num_classes(pyg_dataset)
		pyg_dataset.num_classes
	end
	
	function get_num_features(pyg_dataset)
		pyg_dataset.num_features
	end
end

# ╔═╡ 716fa49d-e5e5-48b0-b613-101ecc4574dd
"""
$(name) dataset has $(get_num_classes(pyg_dataset)) classes and $(get_num_features(pyg_dataset)) features.
"""

# ╔═╡ c7eb3fce-dbda-481e-a76e-93f61c080d30
md"""
### Question 2: What is the label of the graph with index 100 in the ENZYMES dataset?
"""

# ╔═╡ 864e3373-f7bd-4711-b08a-3871f3876485
function get_graph_class(pyg_dataset, idx)
	pyg_dataset[idx].y.numpy()
end

# ╔═╡ f0ecca78-b416-4612-b56c-2b64e5508c8e
let
	idx = 101  # one-based
	@with_terminal println(pyg_dataset[1], 
		"\nGraph with idx $(idx) has label ", get_graph_class(pyg_dataset, idx))
end

# ╔═╡ 8c9d82cc-fcda-4794-837b-d462ddc85a5c
md"""
### Question 3: How many edges does the graph with index 200 have?
"""

# ╔═╡ 8ab89862-c20b-4c1a-935f-060381dc97ab
function get_graph_num_edges(pyg_dataset, idx)
	size(pyg_dataset[idx].edge_index.numpy(), 2)
end

# ╔═╡ f8c2de98-8212-4fb1-acfe-a0a4619a0a14
let
	idx = 201
	@with_terminal println("Graph with index ", idx, " has ", get_graph_num_edges(pyg_dataset, idx), " edges.")
end

# ╔═╡ 096b63e8-c98b-4ddf-b6bc-3dc6af0c92c6
md"""
## 2) Open Graph Benchmark (OGB)

### Dataset and Data

Note that, the python library may ask whether to refresh the dataset, and then the pluto notebook would get stucked according to its design, see [issue](https://github.com/fonsp/Pluto.jl/issues/1394). Thus one needs to run the _ogbn.py_ script to download it manually, then the following notebook scripts could load the dataset.
"""

# ╔═╡ 1010d3c7-f41e-40a0-929b-c012a57e3871
begin
	T = torch_geometric.transforms
	PygNodePropPredDataset = pyimport("ogb.nodeproppred").PygNodePropPredDataset
	
	pickle = pyimport("pickle")
end

# ╔═╡ 70913956-8747-4e85-ba8c-22ebf3817f12
begin
	dataset_name = "ogbn-arxiv"
	# dataset = PygNodePropPredDataset(name=dataset_name, transform=T.ToSparseTensor())
	
	
	dataset = []
	@pywith pybuiltin("open")(dataset_name*".pkl","rb") as f begin
		push!(dataset, pickle.load(f))
	end
	dataset = dataset[]
	
	@with_terminal println("The $(dataset_name) dataset has ", length(dataset), " graph")
end

# ╔═╡ 59dd6973-2faf-4fd3-b231-4415811ddc02
begin
	data = dataset[1]
	@with_terminal println(data)
end

# ╔═╡ ed44dad6-4d70-4bdf-9f57-ee6d91fba5c7
md"""
### Question 4: How many features are in the ogbn-arxiv graph?
"""

# ╔═╡ a20644df-8ab4-4325-9e8d-aabc982b2b29
function graph_num_features(data)
	data.num_features
end

# ╔═╡ 26163f42-3605-41ea-b4b6-be90a9fd080f
@with_terminal println("The graph has ", graph_num_features(data), " features.")

# ╔═╡ f8bad514-6fc6-4ae5-a742-501d4b9004f7
md"""
## 3) GNN: Node Property Prediction 

The origin colab notebook used PyTorch Geometric library, and respectively, here we will utilize `GeometricFlux.jl` package.
"""

# ╔═╡ 48de33b4-1897-4826-a191-ccc7d8b16384
# In pluto, the `data` varaible is global, and all other cells are bounded to that variable. Thus when we change the `data` content, the other related cells will refresh automatically.

# ╔═╡ 1e77ac28-40d1-4f61-900e-a39622181fff
begin
	data.adj_t = data.adj_t.to_symmetric()
	split_idx = dataset.get_idx_split()
	train_idx = split_idx["train"].numpy() .+ 1  # one-based idx
end

# ╔═╡ eb2a6e97-b32e-4b41-b211-311f441c0110
md"""
To use `GeometricFlux.jl`, we need to build a `FeaturedGraph` with the origin data. According to the [description](https://ogb.stanford.edu/docs/nodeprop/) of the _ogbn-arxiv_ dataset, we know that `x` denotes the embeddings to identify the papers, `y` as the labels of the subject areas, and `adj_t` for the vertices. And it is a supervised learning task to train a network to classify these papers.
"""

# ╔═╡ 3c1495e1-e214-4a5a-882b-1cdc2a67a853
adj_t = data.adj_t

# ╔═╡ 7c9215a7-ceb4-4a41-a780-e7908eab78f5
begin
	graph = Graph(length(data.x))
	
	local rows = adj_t.storage.row().numpy() .+ 1  # one-based
	local cols = adj_t.storage.col().numpy() .+ 1
	
	for e in zip(rows, cols)
		add_edge!(graph, e...)
	end
	
	fg = FeaturedGraph(graph.fadjlist)  # according to the changes of distincted packages, we use the adjacency list for compatibility
end

# ╔═╡ 062c7365-1709-4f5d-ae8f-8e4c0ad47e62
# and store the features in the vector form, downloaded from the tensor
features = data.x.numpy() |> transpose  # in Flux, batch number is in the last dimension

# ╔═╡ 0b8e30ce-e965-4fd4-b63a-f56d3cededeb
# same to the labels
labels = data.y.numpy() .+ 1 |> x -> reshape(x, length(x))

# ╔═╡ 3ba26273-061a-48de-9558-64242510a3e8
md"""
Here is the model of GCN will be used. $(PlutoUI.LocalResource("pics/cs224w-colab2-3.png"))
"""

# ╔═╡ b7e617fa-890f-4ed3-9a6c-f8a68d5203df
# arguments
Base.@kwdef struct Args
	num_layers::Int  = 3
	hidden_dim::Int  = 256
	dropout::Float32 = 0.5
	lr::Float32      = 0.01
	epochs::Int      = 100
	n_classes::Int   = dataset.num_classes
end

# ╔═╡ 205d8706-a3ff-4659-91e8-b5d75151da92
# during my test, I encountered the _OutOfMemoryError_, so I just modify some of the implementation to bypass it
function (l::GCNConv)(x::AbstractMatrix)
	fg = l.fg
	x_times_Ã = Zygote.ignore() do
		x * fg.graph.S * (GraphSignals.degree_matrix(fg) ^ (-1))
	end
	l.σ.(l.weight * x_times_Ã .+ l.bias)
end

# ╔═╡ ebabe40a-ba02-471c-90d5-453c6b30a28a
function construct_model(in_dim, hidden_dim, out_dim, num_layers, dropout)
	in_layer = [
		GCNConv(fg, in_dim => hidden_dim),
		BatchNorm(hidden_dim, relu),
		Dropout(dropout),
	]
	hidden_layers = vcat(map(2:num_layers-1) do _
		[
			GCNConv(fg, hidden_dim => hidden_dim),
			BatchNorm(hidden_dim, relu),
			Dropout(dropout),
		]
	end...)
	
	Chain(
		in_layer...,
		hidden_layers...,
		GCNConv(fg, hidden_dim => out_dim),
		logsoftmax,
	)
end

# ╔═╡ 8ab221e8-e0be-4d9d-9602-fb9de83601fe
begin
	args = Args()
	
	model = construct_model(
		dataset.num_features,
		args.hidden_dim,
		dataset.num_classes,
		args.num_layers,
		args.dropout
	)
end

# ╔═╡ 2c9b5945-21a8-49af-8b86-d78c61e6e75e
# some utils for train and test
begin
	function evaluate(m, dataloader, loss_fn, idx=Colon())
		loss = 0.0
		acc = 0.0
		N = 0
		
		for (x,y) in dataloader
			ŷ = m(x)[:, idx]
			y = y[:, idx]
			loss += loss_fn(y, ŷ)
			res = onecold(ŷ, 1:args.n_classes) .== onecold(y, 1:args.n_classes)
			N += length(res)
			acc += sum(res)
		end

		acc /= N
		loss /= N
		
		(loss, acc)
	end
	
	function train(m, data, train_idx, optimizer, loss_fn)
		## The corresponded tasks:
		# 1. zero the gradient and optimizer (no need in Flux.jl)
		# 2. Feed the data into the model

		x, y = data[]  # now know it has only one batch
		for epoch in 1:args.epochs
			# get trainable parameters of the model
			ps = params(m)

			# compute gradient with loss function
			gs = gradient(ps) do
				ŷ = m(x)
				loss_fn(ŷ[:, train_idx], y[:, train_idx])
			end

			# update the parameter with gradient computed
			Flux.update!(optimizer, ps, gs)

			# print log
			loss, acc = evaluate(m, data, loss_fn, train_idx)
			println("Epoch ", epoch, ": loss = ", loss, ", acc = ", acc)
		end
	end
	
	nothing
end

# ╔═╡ e7795865-a619-4470-9534-500340d6013b
# train the model
let
	# train_idx was set in the former cell
	train_labels = onehotbatch(labels, 1:args.n_classes)
	dataloader = [(features, train_labels)]
	loss_fn(y, ŷ) = -dot(y, ŷ)  # in the colab, loss_fn is the negative log likelihood loss function, and the usage here is same to the crossentropy loss, thus just sum them and get it negative.
	opt = ADAM(args.lr)
	
	@with_terminal train(model, dataloader, train_idx, opt, loss_fn)
end

# ╔═╡ 81247bf0-7390-4762-b62e-2e1f64771152
# test the model
let
	test_idx = setdiff(1:size(features, 2), train_idx)
	test_labels = onehotbatch(labels, 1:args.n_classes)
	
	dataloader = [(features, test_labels)]
	loss_fn(y, ŷ) = -dot(y, ŷ)

	evaluate(model, dataloader, loss_fn, test_idx)
end

# ╔═╡ 5807af5e-c7ac-42f9-a259-eba7710655a6


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
GeometricFlux = "7e08b658-56d3-11e9-2997-919d5b31e4ea"
GraphSignals = "3ebe565e-a4b5-49c6-aed2-300248c3a9c1"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
Flux = "~0.12.8"
GeometricFlux = "~0.8.0"
GraphSignals = "~0.3.9"
Graphs = "~1.5.1"
PlutoUI = "~0.7.29"
PyCall = "~1.93.0"
Zygote = "~0.6.34"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9faf218ea18c51fcccaf956c8d39614c9d30fe8b"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.2"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "265b06e2b1f6a216e0e8f183d28e4d354eab3220"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.2.1"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BinaryProvider]]
deps = ["Libdl", "Logging", "SHA"]
git-tree-sha1 = "ecdec412a9abc8db54c0efc5548c64dfce072058"
uuid = "b99e7846-7c00-51b0-8f62-c81ae34c0232"
version = "0.5.10"

[[Blosc]]
deps = ["Blosc_jll"]
git-tree-sha1 = "575bdd70552dd9a7eaeba08ef2533226cdc50779"
uuid = "a74b3585-a348-5f62-a45c-50e91977d574"
version = "0.7.2"

[[Blosc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Lz4_jll", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "91d6baa911283650df649d0aea7c28639273ae7b"
uuid = "0b7ba130-8d10-5ba8-a3d6-c5182647fed9"
version = "1.21.1+0"

[[BufferedStreams]]
deps = ["Compat", "Test"]
git-tree-sha1 = "5d55b9486590fdda5905c275bb21ce1f0754020f"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.0.0"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "49f14b6c56a2da47608fe30aed711b5882264d7a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.9.11"

[[CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "429a1a05348ce948a96adbdd873fbe6d9e5e052f"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.6.2"

[[ChainRules]]
deps = ["ChainRulesCore", "Compat", "LinearAlgebra", "Random", "RealDot", "Statistics"]
git-tree-sha1 = "c6366ec79d9e62cd11030bba0945712eb4013712"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.17.0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "4c26b4e9e91ca528ea212927326ece5918a04b47"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.2"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6cdc8832ba11c7695f494c9d9a1c31e90959ce0f"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.6.0"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataDeps]]
deps = ["BinaryProvider", "HTTP", "Libdl", "Reexport", "SHA", "p7zip_jll"]
git-tree-sha1 = "4f0e41ff461d42cfc62ff0de4f1cd44c6e6b3771"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.7"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "cfdfef912b7f93e4b848e80b9befdf9e331bc05a"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.1"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "9bc5dac3c8b6706b58ad5ce24cffd9861f07c94f"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.9.0"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2db648b6712831ecb333eae76dbfd1c156ca13bb"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.2"

[[FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "04d13bfa8ef11720c24e4d840c0033d145537df7"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.17"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Flux]]
deps = ["AbstractTrees", "Adapt", "ArrayInterface", "CUDA", "CodecZlib", "Colors", "DelimitedFiles", "Functors", "Juno", "LinearAlgebra", "MacroTools", "NNlib", "NNlibCUDA", "Pkg", "Printf", "Random", "Reexport", "SHA", "SparseArrays", "Statistics", "StatsBase", "Test", "ZipFile", "Zygote"]
git-tree-sha1 = "e8b37bb43c01eed0418821d1f9d20eca5ba6ab21"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.12.8"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2b72a5624e289ee18256111657663721d59c143e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.24"

[[Functors]]
git-tree-sha1 = "e4768c3b7f597d5a352afa09874d16e3c3f6ead2"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.7"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GPUArrays]]
deps = ["Adapt", "LinearAlgebra", "Printf", "Random", "Serialization", "Statistics"]
git-tree-sha1 = "d9681e61fbce7dde48684b40bdb1a319c4083be7"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.1.3"

[[GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "2cac236070c2c4b36de54ae9146b55ee2c34ac7a"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.13.10"

[[GeometricFlux]]
deps = ["CUDA", "ChainRulesCore", "DataStructures", "FillArrays", "Flux", "GraphMLDatasets", "GraphSignals", "Graphs", "LinearAlgebra", "NNlib", "NNlibCUDA", "Random", "Reexport", "Statistics", "Zygote"]
git-tree-sha1 = "3f999980fd4a3569e253eba67512b22ef806619b"
uuid = "7e08b658-56d3-11e9-2997-919d5b31e4ea"
version = "0.8.0"

[[GraphMLDatasets]]
deps = ["CSV", "CodecZlib", "DataDeps", "DataFrames", "DelimitedFiles", "FileIO", "Graphs", "HTTP", "InteractiveUtils", "JLD2", "JSON", "MAT", "NPZ", "Pickle", "PyCall", "SparseArrays", "ZipFile"]
git-tree-sha1 = "2cd21c14b75d5cb031fdcf77a99d19489d9cf6a6"
uuid = "21828b05-d3b3-40ad-870e-a4bc2f52d5e8"
version = "0.1.5"

[[GraphSignals]]
deps = ["CUDA", "ChainRulesCore", "FillArrays", "Functors", "Graphs", "LinearAlgebra", "NNlib", "NNlibCUDA", "SimpleWeightedGraphs", "SparseArrays", "StatsBase"]
git-tree-sha1 = "30773e012210cd6315edbe11a1b06c6e3f66b761"
uuid = "3ebe565e-a4b5-49c6-aed2-300248c3a9c1"
version = "0.3.9"

[[Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "d727758173afef0af878b29ac364a0eca299fc6b"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.5.1"

[[HDF5]]
deps = ["Blosc", "Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires"]
git-tree-sha1 = "698c099c6613d7b7f151832868728f426abe698b"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.15.7"

[[HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "fd83fa0bde42e01952757f01149dd968c06c4dba"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.0+1"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "006127162a51f0effbdfaab5ac0c83f8eb7ea8f3"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.4"

[[IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "8d70835a3759cdd75881426fced1508bb7b7e1b6"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.1"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InternedStrings]]
deps = ["Random", "Test"]
git-tree-sha1 = "eb05b5625bc5d821b8075a77e4c421933e20c76b"
uuid = "7d512f48-7fb1-5a58-b986-67e6dc259f01"
version = "0.7.0"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLD2]]
deps = ["DataStructures", "FileIO", "MacroTools", "Mmap", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "09ef0c32a26f80b465d808a1ba1e85775a282c97"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.17"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "7cc22e69995e2329cc047a879395b2b74647ab5f"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.7.0"

[[LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "62115afed394c016c2d3096c5b85c407b48be96b"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.13+1"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5d494bc6e85c4c9b626ee0cab05daa4085486ab1"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.3+0"

[[MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "37d418e2f20f0fcdc78214f763f1066b74ca1e1b"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.2"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NNlib]]
deps = ["Adapt", "ChainRulesCore", "Compat", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "2eb305b13eaed91d7da14269bf17ce6664bfee3d"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.7.31"

[[NNlibCUDA]]
deps = ["CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics"]
git-tree-sha1 = "a2dc748c9f6615197b6b97c10bcce829830574c9"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.1.11"

[[NPZ]]
deps = ["Compat", "ZipFile"]
git-tree-sha1 = "fbfb3c151b0308236d854c555b43cdd84c1e5ebf"
uuid = "15e1cf62-19b3-5cfa-8e77-841668bca605"
version = "0.4.1"

[[NaNMath]]
git-tree-sha1 = "f755f36b19a5116bb580de457cda0c140153f283"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.6"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "d7fa6237da8004be601e19bd6666083056649918"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.3"

[[Pickle]]
deps = ["DataStructures", "InternedStrings", "Serialization", "SparseArrays", "Strided", "ZipFile"]
git-tree-sha1 = "b4054944f1bfb956fb38fb54ee760e33c5507d35"
uuid = "fbb45041-c46e-462f-888f-7c521cafbc2c"
version = "0.2.10"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "7711172ace7c40dc8449b7aed9d2d6f1cf56a5bd"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.29"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "db3a23166af8aebf4db5ef87ac5b00d36eb771e2"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "71fd4022ecd0c6d20180e23ff1b3e05a143959c2"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.93.0"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Random123]]
deps = ["Libdl", "Random", "RandomNumbers"]
git-tree-sha1 = "0e8b146557ad1c6deb1367655e052276690e71a3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.4.2"

[[RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "8f82019e525f4d5c669692772a6f4b0a58b06a6a"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.2.0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "244586bc07462d22aed0113af9c731f2a518c93e"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.10"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays", "Test"]
git-tree-sha1 = "a6f404cc44d3d3b28c793ec0eb59af709d827e4e"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.2.1"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e08890d19787ec25029113e88c34ec20cac1c91e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.0.0"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "7f5a513baec6f122401abfc8e9c074fdac54f6c1"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.4.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "d88665adc9bcf45903013af0982e2fd05ae3d0a6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "51383f2d367eb3b444c961d485c565e4c0cf4ba0"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.14"

[[Strided]]
deps = ["LinearAlgebra", "TupleTools"]
git-tree-sha1 = "4d581938087ca90eab9bd4bb6d270edaefd70dcd"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "1.1.2"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "7cb456f358e8f9d102a8b25e8dfedf58fa5689bc"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.13"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[TupleTools]]
git-tree-sha1 = "3c712976c47707ff893cf6ba4354aa14db1d8938"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.3.0"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VersionParsing]]
git-tree-sha1 = "e575cf85535c7c3292b4d89d89cc29e8c3098e47"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.1"

[[WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "c69f9da3ff2f4f02e811c3323c22e5dfcb584cfa"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.1"

[[ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "3593e69e469d2111389a9bd06bac1f3d730ac6de"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.4"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "IRTools", "InteractiveUtils", "LinearAlgebra", "MacroTools", "NaNMath", "Random", "Requires", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "88a4d79f4e389456d5a90d79d53d1738860ef0a5"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.34"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─2f2c2200-6895-11ec-25f5-497e6fabd0f1
# ╠═914e7475-98c9-49c3-8fde-cfdc94612062
# ╟─93cd2223-8906-4c02-954e-4530ba26c38a
# ╠═7ba01ad4-8019-4635-af52-b59017968630
# ╠═fa788def-e2d3-4afb-98a2-37e1b2dedac3
# ╠═760580a3-98e4-4c77-970f-3d6dac81098a
# ╟─dbe21b78-581f-42e3-bd27-51b037400982
# ╠═71583d68-6a3c-48c0-8c78-b1c137c72c24
# ╟─d713b09c-bd1f-427a-8bc2-14d99c548985
# ╠═d925094a-7f4c-4bdb-ad86-a6f9d3d17d27
# ╠═0a52ecbc-b016-4bc2-9975-44413456e43c
# ╠═2aa87ffc-574c-41cb-be8a-1bb6028c5692
# ╟─7f3cfb6b-2da0-4786-9e00-42fd6da4305d
# ╠═ddcb0349-9983-4c5b-ae9c-af425cbc0044
# ╠═5ade533b-b6a4-45aa-a1d5-97f71f0fd5de
# ╟─716fa49d-e5e5-48b0-b613-101ecc4574dd
# ╟─c7eb3fce-dbda-481e-a76e-93f61c080d30
# ╠═864e3373-f7bd-4711-b08a-3871f3876485
# ╠═f0ecca78-b416-4612-b56c-2b64e5508c8e
# ╟─8c9d82cc-fcda-4794-837b-d462ddc85a5c
# ╠═8ab89862-c20b-4c1a-935f-060381dc97ab
# ╠═f8c2de98-8212-4fb1-acfe-a0a4619a0a14
# ╟─096b63e8-c98b-4ddf-b6bc-3dc6af0c92c6
# ╠═1010d3c7-f41e-40a0-929b-c012a57e3871
# ╠═70913956-8747-4e85-ba8c-22ebf3817f12
# ╠═59dd6973-2faf-4fd3-b231-4415811ddc02
# ╟─ed44dad6-4d70-4bdf-9f57-ee6d91fba5c7
# ╠═a20644df-8ab4-4325-9e8d-aabc982b2b29
# ╠═26163f42-3605-41ea-b4b6-be90a9fd080f
# ╟─f8bad514-6fc6-4ae5-a742-501d4b9004f7
# ╠═48de33b4-1897-4826-a191-ccc7d8b16384
# ╠═9b9ec843-e328-4964-ab7f-65176397d227
# ╠═a7a463a0-ae0b-40aa-930b-4e8d33b1d8df
# ╠═1e77ac28-40d1-4f61-900e-a39622181fff
# ╟─eb2a6e97-b32e-4b41-b211-311f441c0110
# ╠═3c1495e1-e214-4a5a-882b-1cdc2a67a853
# ╠═7c9215a7-ceb4-4a41-a780-e7908eab78f5
# ╠═062c7365-1709-4f5d-ae8f-8e4c0ad47e62
# ╠═0b8e30ce-e965-4fd4-b63a-f56d3cededeb
# ╟─3ba26273-061a-48de-9558-64242510a3e8
# ╠═ebabe40a-ba02-471c-90d5-453c6b30a28a
# ╠═b7e617fa-890f-4ed3-9a6c-f8a68d5203df
# ╠═8ab221e8-e0be-4d9d-9602-fb9de83601fe
# ╠═48086045-2640-477b-880b-82c47fc7bcf7
# ╠═0c6a82d7-7f55-4895-9ff8-eeb0ceb2f2a3
# ╠═d6bf7b0e-f042-453c-aab7-46ec8651a4a3
# ╠═205d8706-a3ff-4659-91e8-b5d75151da92
# ╠═2c9b5945-21a8-49af-8b86-d78c61e6e75e
# ╠═e7795865-a619-4470-9534-500340d6013b
# ╠═81247bf0-7390-4762-b62e-2e1f64771152
# ╠═5807af5e-c7ac-42f9-a259-eba7710655a6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
