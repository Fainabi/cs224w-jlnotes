### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ b77c7312-cabd-48d3-b7af-f186fbf2f290
begin
	import Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()
end

# ╔═╡ e9b554cb-b5b8-4f5c-8373-cb2515c3afcb
begin
	using GeometricFlux, Flux, GraphSignals, JLD2, PlutoUI, Graphs
	using Flux: @functor, onecold
	using Statistics: mean
	using Random
	import Zygote
end

# ╔═╡ 3fcd6690-8501-11ec-352e-fb3b6b7a6aec
md"""
# CS224w - Notebook 3

Here the task is to implement GraphSAGE, and then run on the CORA dataset.
"""

# ╔═╡ b108d306-1e7f-4b13-98d0-3a982b8d59db
md"""
## 1) GNN Layers

### GNN Stack Module

Take the template of colab-3, we product the following GNN stack module. Since the layers we construct and apply in Flux is differnet to that in PyTorch, here we skip it.
"""

# ╔═╡ f3c0c116-8c40-4032-b326-7fd73128f6d2
md"""
### GraphSage implementation

The message passing update rule is:

$h_v^{(l)} = \sigma(W_l\cdot h_v^{(l-1)} + W_r\cdot AGG(\{h_u^{(l-1)}, \forall u\in N(v)\}))$

where

$AGG(\{h_u\}) = \frac{1}{|N(v)|}\sum_{u\in N(v)}h_u$.

Here we do not need follow the form of `message`-`aggregate`-`update`, to seperate the layer into three different functions. Also, considered the design of GeometricFlux (which uses featured graph), we'd best implement them in matrix form.
"""

# ╔═╡ 792865a3-71c7-4588-a47d-8c14dd593ce0
md"""
Notes for the implementation: 

1. it seems that there's no bias in the transformation, and neither in the origin paper.
2. I add an activative function here; the reason colab does not show it is that it utilized `relu` in the former `GNNStack` framework. In the origin paper, the activative function was drawn.
3. Here no sampling is implemented.
"""

# ╔═╡ 3524283d-9f56-4e4f-88c5-fd85511f8c5e
begin
	struct GraphSage{M<:AbstractMatrix, F, S<:AbstractFeaturedGraph}
		Wl::M
		Wr::M
		σ::F
		fg::S
	end

	# these codes take GeometricFlux.jl as template
	function GraphSage(fg::AbstractFeaturedGraph, ch::Pair{Int,Int}, σ=identity;
                 init=Flux.glorot_uniform)
	    in_dim, out_dim = ch
	    Wl = init(out_dim, in_dim)
		Wr = init(out_dim, in_dim)
	    GraphSage(Wl, Wr, σ, fg)
	end

	@functor GraphSage
	Flux.trainable(l::GraphSage) = (l.Wl, l.Wr)

	function (l::GraphSage)(x::AbstractMatrix)
		fg = l.fg
	    Ã = Zygote.ignore() do
	        GraphSignals.normalized_adjacency_matrix(fg, Float32; selfloop=true)
	    end
	    l.σ.(l.Wl * x .+ l.Wr * x * Ã)
	end
	
	function Base.show(io::IO, l::GraphSage)
	    out_dim, in_dim = size(l.Wl)
	    print(io, "GraphSage($in_dim => $out_dim")
	    l.σ == identity || print(io, ", ", l.σ)
	    print(io, ")")
	end
end

# ╔═╡ 8d1e2ffd-6e39-4122-8f0a-4aceb8acd124
function construct_model(fg, in_dim, hidden_dim, out_dim, n_layers, dropout)
	gcn = vcat(map(1:n_layers-1) do i
		l = if i == 1
			GraphSage(fg, in_dim=>hidden_dim, relu)
		else
			GraphSage(fg, hidden_dim=>hidden_dim, relu)
		end

		[l, Dropout(dropout)]
	end...)

	Chain(
		gcn...,
		Dense(hidden_dim, hidden_dim), # post message passing, i.e. FC layers
		Dropout(dropout),
		Dense(hidden_dim, out_dim)
	)
end

# ╔═╡ 4572424c-e685-460b-844b-72d47545c0eb
md"""
Now load the dataset.
"""

# ╔═╡ 12277919-fc51-47c9-bfd7-c7588797523e
begin
	data_path = joinpath(pkgdir(GeometricFlux), "data")
	@load joinpath(data_path, "cora_features.jld2") features
	@load joinpath(data_path, "cora_labels.jld2") labels
	@load joinpath(data_path, "cora_graph.jld2") g
end

# ╔═╡ 722f4409-1912-4127-91cb-7f01a7a5449a
md"""
The dataset stored in the `GeometricFlux.jl` does not provide separation for train and test dataset. Thus here we need to do it manually.
"""

# ╔═╡ 7ac3bfbe-7eb7-4d51-af7e-1dd97644cf19
PlutoUI.with_terminal() do
	println("The number of nodes in the dataset is ", size(labels, 2))
end

# ╔═╡ 6dc86d65-00ee-4cd0-afa4-01c2e2272e2e
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

# ╔═╡ e29963ea-bd9e-4fb8-a472-d6302f4b6d10
md"""
Now, start training!
"""

# ╔═╡ d92bd341-bb60-4efa-9880-4196efe5d916
Base.@kwdef struct Args
	epochs::Int     = 100
	hidden_dim::Int  = 32
	n_labels::Int    = size(labels, 1)
	n_layers::Int    = 2
	dropout::Float32 = 0.5
	lr::Float32      = 0.01
end

# ╔═╡ 1b06e003-1a39-4f47-9bd1-98918005708d
begin
	args = Args()
	fg = FeaturedGraph(g.fadjlist)

	loss_fn(ŷ, y) = Flux.Losses.logitcrossentropy(ŷ, y)
	
	model = construct_model(
		fg, 
		size(features, 1), args.hidden_dim, args.n_labels, 
		args.n_layers, args.dropout)
end

# ╔═╡ 881f929d-6406-4a6c-a986-cfe30d05c594
function evaluate(model, data, loss_fn, idx=Colon())
	x, y = data
	ŷ = model(x)

	loss = loss_fn(ŷ[:, idx], y[:, idx])
	acc = (onecold(ŷ[:, idx], 1:args.n_labels) .== onecold(y[:, idx], 1:args.n_labels)) |> mean
	
	(loss, acc)
end

# ╔═╡ 8919a0e8-2c0c-4795-9731-12a722145510
begin
	with_terminal() do
		opt = ADAM(args.lr)
		
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

# ╔═╡ bdf77406-0d7f-4e58-906e-d99d8998dde3
md"""
Test: 
"""

# ╔═╡ 91ba5929-020d-411e-a96c-d5af66d92f41
let
	loss, acc = evaluate(model, (features, labels), loss_fn, test_idx)
	with_terminal() do 
		println("Test loss: ", loss, ", test acc: ", acc)
	end
end

# ╔═╡ Cell order:
# ╟─3fcd6690-8501-11ec-352e-fb3b6b7a6aec
# ╟─b108d306-1e7f-4b13-98d0-3a982b8d59db
# ╠═b77c7312-cabd-48d3-b7af-f186fbf2f290
# ╠═e9b554cb-b5b8-4f5c-8373-cb2515c3afcb
# ╟─f3c0c116-8c40-4032-b326-7fd73128f6d2
# ╟─792865a3-71c7-4588-a47d-8c14dd593ce0
# ╠═3524283d-9f56-4e4f-88c5-fd85511f8c5e
# ╠═8d1e2ffd-6e39-4122-8f0a-4aceb8acd124
# ╟─4572424c-e685-460b-844b-72d47545c0eb
# ╠═12277919-fc51-47c9-bfd7-c7588797523e
# ╟─722f4409-1912-4127-91cb-7f01a7a5449a
# ╠═7ac3bfbe-7eb7-4d51-af7e-1dd97644cf19
# ╠═6dc86d65-00ee-4cd0-afa4-01c2e2272e2e
# ╟─e29963ea-bd9e-4fb8-a472-d6302f4b6d10
# ╠═d92bd341-bb60-4efa-9880-4196efe5d916
# ╠═1b06e003-1a39-4f47-9bd1-98918005708d
# ╠═881f929d-6406-4a6c-a986-cfe30d05c594
# ╠═8919a0e8-2c0c-4795-9731-12a722145510
# ╟─bdf77406-0d7f-4e58-906e-d99d8998dde3
# ╠═91ba5929-020d-411e-a96c-d5af66d92f41
