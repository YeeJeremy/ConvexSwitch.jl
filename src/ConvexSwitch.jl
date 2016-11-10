################################################################################
## Copyright 2016. Juri Hinz and Jeremy Yee.
################################################################################

module ConvexSwitch

## Load required packages
using Clustering
using NearestNeighbors

## Export immutable types
export CSS, Bellman, Path, PermMat, SGrid

## Export functions
export disturb_mat, path_mat, row_rearrange, expected, expected_host, expected_fast

## Include source files
include("functions_mat.jl")
include("CSS.jl")
include("Path.jl")
include("PermMat.jl")
include("functions.jl")
include("SGrid.jl")
include("Bellman.jl")

end # module
