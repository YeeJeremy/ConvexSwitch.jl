################################################################################
## Copyright 2016. Juri Hinz and Jeremy Yee.
################################################################################

module ConvexSwitch

## Load required packages
using Clustering
using NearestNeighbors

## Export immutable types
export CSS, Bellman, Path, PermMat, SGrid, PathPolicy, Mart, OptimalBounds

## Export functions
export disturb_mat, path_mat, subsim_mat # generate disturb matrices
export row_rearrange, expected, expected_host, expected_fast # Bellman recursion
export test_policy # test run of prescribed policy

## Include source files
include("functions_mat.jl")
include("CSS.jl")
include("Path.jl")
include("PermMat.jl")
include("functions_bellman.jl")
include("SGrid.jl")
include("Bellman.jl")
include("PathPolicy.jl")
include("test_policy.jl")
include("Mart.jl")
include("OptimalBounds.jl")

end # module
