### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ a0c6195c-b9cc-4853-aa6f-0c107753b8b7
begin
	import Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()
end

# ╔═╡ ba1b80ab-2620-4177-b14f-065469b6ec92
begin
	using GeometricFlux, Flux, GraphSignals, JLD2, PlutoUI, Graphs
	using Flux: @functor, onecold
	using Statistics: mean
	using Random
	import Zygote
end

# ╔═╡ 5d2a01f0-85bd-11ec-2498-694e02d12fda
md"""
# CS224w - Notebook 4
"""

# ╔═╡ 2560ade3-4ee3-49b6-9ea4-156d78ca4765
md"""
## GAT Implementation
"""

# ╔═╡ 3516d842-8b7b-49d8-96f1-f7e7dd821000
begin
	# GeometricFlux.jl has already implemented it, named GATConv.
	# One head attention network

	# Question: since Wl and al are both trainable, why don't we use a single matrix L = al' * Wl
	struct GAT{M<:AbstractMatrix, F, G, S<:AbstractFeaturedGraph}
		Wl::M
		Wr::M
		al::M  # N×1 matrix
		ar::M
		σ::F
		σa::G  # for attention 
		fg::S
	end

	function GAT(fg::AbstractFeaturedGraph, ch::Pair{Int,Int}, σ=identity, σa=identity; init=Flux.glorot_uniform)
		in_dim, out_dim = ch

		GAT(
			init(out_dim, in_dim),
			init(out_dim, in_dim),
			init(1, out_dim),
			init(1, out_dim),
			σ,
			σa,
			fg
		)
	end

	@functor GAT
	Flux.trainable(l::GAT) = (l.Wl, l.Wr, l.al, l.ar)

	function (l::GAT)(x::AbstractMatrix)
		# x is the feature matrix
		# we need to compute for all the features
		aWl = l.al * l.Wl * x  # one-times-N matrix
		aWr = l.ar * l.Wr * x

		adj = Zygote.ignore() do
			Matrix(l.fg.graph.S)
		end
		e_ij = (aWr' .* adj) + (adj .* aWl)
		p_ij = exp.(l.σa.(e_ij))  # dense because exp would produce no zeros
		α_ij = p_ij .* (sum(p_ij, dims=1).^(-1))  # vertical

		l.σ.(l.Wr * x * α_ij)
	end

	function Base.show(io::IO, l::GAT)
		out_dim, in_dim = size(l.Wl)
	    print(io, "GAT($in_dim => $out_dim")
	    l.σ == identity || print(io, ", ", l.σ)
		l.σa == identity || print(io, ", leakyrelu")
	    print(io, ")")
	end
end

# ╔═╡ 2578d838-1954-4bf4-bf4a-4d9f0a9be6e3
md"""
Load the data.
"""

# ╔═╡ 9cde4347-8bbe-49bd-b425-af52b7cdf56c
begin
	data_path = joinpath(pkgdir(GeometricFlux), "data")
	@load joinpath(data_path, "cora_features.jld2") features
	@load joinpath(data_path, "cora_labels.jld2") labels
	@load joinpath(data_path, "cora_graph.jld2") g
end

# ╔═╡ 103afb55-8210-461d-aeb0-f260a35e86e0
fg = FeaturedGraph(g.fadjlist)

# ╔═╡ 116a8749-c66b-4fd4-b775-29042739f567
begin
	Base.@kwdef struct Args
		n_layers::Int      = 2
		hidden_dim::Int    = 32
		dropout::Float32   = 0.5
		epochs::Int        = 40
		lr::Float32        = 0.01
		leaky_coe::Float32 = 0.2
		n_labels::Int      = size(labels, 1)
	end

	args = Args()
end

# ╔═╡ 4d92d8b4-ca60-412b-a7ff-03d06576d55f
function construct_model(fg, in_dim, hidden_dim, out_dim)
	gnn = vcat(map(1:args.n_layers-1) do i
		conv_l = if i == 1
			GAT(fg, in_dim=>hidden_dim, relu, x->leakyrelu(x, args.leaky_coe))
		else
			GAT(fg, hidden_dim=>hidden_dim, relu, x->leakyrelu(x, args.leaky_coe))
		end

		[conv_l, Dropout(args.dropout)]
	end...)

	Chain(
		gnn...,
		Dense(hidden_dim, hidden_dim),
		Dropout(args.dropout),
		Dense(hidden_dim, out_dim)
	)
end

# ╔═╡ 3243c22b-77d6-45a4-bdcd-31cef37376eb
begin
	train_ratio = 0.8
	n_nodes = size(labels, 2)
	train_size = round(n_nodes * train_ratio) |> Int

	# randomly permute the idx, get the former some ask train idx
	idx_permuted = randperm(n_nodes)
	train_idx = idx_permuted[1:train_size]
	test_idx = idx_permuted[train_size+1:end]

	with_terminal() do
		println("Train size is ", length(train_idx), ", and test size is ", length(test_idx))
	end
end

# ╔═╡ 8cbbdeec-05bc-4f22-bf6f-04c041a809fd
md"""
Train the model:
"""

# ╔═╡ 01ec75df-f448-40b3-91ac-a8594292f815
function evaluate(model, data, loss_fn, idx=Colon())
	x, y = data
	ŷ = model(x)

	loss = loss_fn(ŷ[:, idx], y[:, idx])
	acc = (onecold(ŷ[:, idx], 1:args.n_labels) .== onecold(y[:, idx], 1:args.n_labels)) |> mean
	
	(loss, acc)
end

# ╔═╡ 228ad100-c34b-44db-8de5-5675b89d1e7c
begin
	loss_fn(ŷ, y) = Flux.Losses.logitcrossentropy(ŷ, y)
	model = construct_model(fg, size(features, 1), args.hidden_dim, size(labels, 1))
	opt = ADAM(args.lr)

	with_terminal() do
		for epoch in 1:args.epochs
			ps = params(model)
	
			gs = gradient(ps) do
				ŷ = model(features)
				loss_fn(ŷ[:, train_idx], labels[:, train_idx])
			end
	
			Flux.update!(opt, ps, gs)
	
			loss, acc = evaluate(model, (features, labels), loss_fn, train_idx)
			println("Epoch ", epoch, ", Loss is ", loss, ", Acc is ", acc)
		end
	end
end

# ╔═╡ e4295447-da81-465c-8cc2-69872a3e71da
model

# ╔═╡ b2f1a479-a32b-4de8-ba9c-e763aabe27de
let
	loss, acc = evaluate(model, (features, labels), loss_fn, test_idx)
	with_terminal() do 
		println("Test loss: ", loss, ", test acc: ", acc)
	end
end

# ╔═╡ Cell order:
# ╟─5d2a01f0-85bd-11ec-2498-694e02d12fda
# ╠═a0c6195c-b9cc-4853-aa6f-0c107753b8b7
# ╠═ba1b80ab-2620-4177-b14f-065469b6ec92
# ╟─2560ade3-4ee3-49b6-9ea4-156d78ca4765
# ╠═3516d842-8b7b-49d8-96f1-f7e7dd821000
# ╟─2578d838-1954-4bf4-bf4a-4d9f0a9be6e3
# ╠═9cde4347-8bbe-49bd-b425-af52b7cdf56c
# ╠═103afb55-8210-461d-aeb0-f260a35e86e0
# ╠═116a8749-c66b-4fd4-b775-29042739f567
# ╠═4d92d8b4-ca60-412b-a7ff-03d06576d55f
# ╠═3243c22b-77d6-45a4-bdcd-31cef37376eb
# ╟─8cbbdeec-05bc-4f22-bf6f-04c041a809fd
# ╠═01ec75df-f448-40b3-91ac-a8594292f815
# ╠═e4295447-da81-465c-8cc2-69872a3e71da
# ╠═228ad100-c34b-44db-8de5-5675b89d1e7c
# ╠═b2f1a479-a32b-4de8-ba9c-e763aabe27de
