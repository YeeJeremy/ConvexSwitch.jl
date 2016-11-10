################################################################################
## Copyright 2016. Juri Hinz and Jeremy Yee.
## A collection of useful functions for Bellman recursion
################################################################################

## Determining the subgradient envelope on more than 1 point

function row_rearrange(subgrad::Array{Float64, 2}, grid::Array{Float64, 2})

    ## Extract sizes
    gnum = size(grid)[1] # number of grid points
    snum = size(grid)[2] # size of state

    ## Get the subgradient evelope
    result = Array{Float64}(gnum, snum)

    for i = 1:gnum
        result[i, :] = subgrad[indmax(subgrad * grid[i, :]), :]
    end

    return result

end

## Determining the subgradient envelope on only one point

function row_rearrange(subgrad::Array{Float64, 2}, grid::Array{Float64, 1})

    ## Extract sizes
    snum = length(grid) # size of state

    ## Get the subgradient evelope
    result = Array{Float64}(1, snum)
    result = subgrad[indmax(subgrad * grid), :]

    return result

end

## Compute the continuation value with just the row rearrangement operator

function expected(value::Array{Float64, 2}, grid::Array{Float64, 2},
                  disturb::Array{Float64, 3}, weight::Array{Float64, 1})

    ## Extract sizes
    gnum = size(value)[1]
    snum = size(value)[2]
    dnum = size(disturb)[3] # number of disturbances

    ## Approximate continuation value function
    result = zeros(Array{Float64}(gnum, snum))

    for i = 1:dnum
        subgrad = value * disturb[:, :, i] * weight[i]
        result += row_rearrange(subgrad, grid)
    end
    
    return result

end

## Compute the continuation value using k nearest neighbors using search function
## This should be used outside of the Bellman recursion

function expected_host(value::Array{Float64, 2}, grid::Array{Float64, 2},
                       disturb::Array{Float64, 3}, weight::Array{Float64, 1},
                       search::Function, k::Int64)

    ## Extract sizes
    gnum = size(value)[1]
    snum = size(value)[2]
    dnum = size(disturb)[3] # number of disturbances

    ## Construct search structure
    tree = search(transpose(grid))
    
    ## Approximate continuation value function
    result = zeros(Array{Float64}(gnum, snum))
    temp_subgrad = Array{Float64}(k, snum) # subgradients under consideration
    
    for i = 1:dnum
        temp_indices = knn(tree, disturb[:, :, i] * transpose(grid), k, true)[1]
        for g = 1:gnum
            temp_subgrad = value[temp_indices[g], :]
            result[g, :] += row_rearrange(temp_subgrad, grid[g, :]) * weight[i]
        end
    end
    
    return result

end


## Compute the continuation value using k nearest neighbors using search function
## This should used inside the Bellman recursion

function expected_host(value::Array{Float64, 2}, grid::Array{Float64, 2},
                       disturb::Array{Float64, 3}, weight::Array{Float64, 1},
                       indices::Array{Array{Int64, 1}, 1})

    ## Extract sizes
    gnum = size(value)[1]
    snum = size(value)[2]
    dnum = size(disturb)[3] # number of disturbances
    k = length(indices[1])
    
    ## Approximate continuation value function
    result = zeros(Array{Float64}(gnum, snum))
    temp_subgrad = Array{Float64}(k, snum) # subgradients under consideration
    
    for i = 1:dnum
        for g = 1:gnum
            temp_subgrad = value[indices[(i - 1) * gnum + g], :] * disturb[:, :, i]
            result[g, :] += row_rearrange(temp_subgrad, grid[g, :]) * weight[i]
        end
    end
    
    return result

end


## Compute the continuation value using the permutation methods

function expected_fast(value::Array{Float64, 2}, X::PermMat)

    ## Extract sizes
    gnum = size(value)[1]
    snum = size(value)[2]
    
    ## Approximate continuation value function
    result = zeros(Array{Float64}(gnum, snum))
    result = X.permMat[:, :, X.rnum + 1] * value * X.constMat

    for i = 1:X.rnum
        result[:, X.rIndex[i, 2]] += X.permMat[:, :, i] * value[:, X.rIndex[i, 1]]
    end
    
    return result

end
