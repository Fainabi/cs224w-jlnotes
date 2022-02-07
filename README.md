# CS224w-jlnotes

This repository is for learning [cs224w](http://web.stanford.edu/class/cs224w/). Notebooks are all written in julia. The `notebookx.jl` are notebooks corresponding to `Colab x` in the schedule of cs224w. This repository only track the codes, while not for the handwriting homeworks, especially the final assignment and project.

## Overview

CS224w had proposed 5 notebooks, which provide basic codes of graph structure in python, with the library of `networkx`, `deepSNAP`, and well-known `pytorch`, `pytorch-geometric`. In julia, the famous deep learning framework `Flux.jl` and graph package `Graphs.jl` are used (`MetaGraphs.jl` for Heterogeneous graphs). Based on them, `GeometricFlux.jl` is a deep learning package working on graphs, and provide several implementations. 

These packages helps learning ML on graphs, and also the DL framework in `Flux.jl`. In cs224w, we could learn a great amount of details about Graphs and GNNs. Yet, in the notebooks, we only contact with some of those content, as a good introduction.

Notebooks were not tested on CUDA, for that the datasets we are playing with are not too huge.

## View Notebooks

The notebooks are finished with [`Pluto.jl`](https://github.com/fonsp/Pluto.jl), a notebook engine in julia. To read these notebooks, run in julia:

```julia
julia> using Pluto

julia> Pluto.run()
```

and load these notebooks.

