################################################################################
## Copyright 2016. Juri Hinz and Jeremy Yee.
## Conditional expectation matrices for fast Bellman recursion
################################################################################

## Type containing the skeleton matrix and permutation matrices

immutable PermMat

    ## Matrices
    constMat::Array{Float64} # (gnum, gnum) non-random component of disturbances
    permMat::Array{Float64, 3} # (gnum, gnum, rnum + 1) permutation matrices
    rIndex::Array{Int64, 2} # location of random entries (row, column)
    rnum::Int64 # number of random entries in disturbances
    
    function PermMat(css::CSS, rIndex::Array{Int64, 2}, search::Function)
        
        ## Create search structure
        transgrid = transpose(css.grid)
        tree = search(transgrid)

        ## Create the matrices
        rnum = size(rIndex)[1];
        permMat = zero(Array{Float64}(css.gnum, css.gnum, rnum + 1))

        ## Assigning entries to matrices
        for i = 1:css.dnum 
            nearest = knn(tree, css.disturb[:, :, i] * transgrid, 1, true)[1]
            for j = 1:css.gnum
                permMat[j, nearest[j], rnum + 1] +=  css.weight[i]
                for k = 1:rnum
                    permMat[j, nearest[j], k] +=
                        css.disturb[rIndex[k, 1], rIndex[k, 2], i] * css.weight[i]
                end    
            end
        end

        ## Extract the disturbance skeleton
        constMat = css.disturb[:, :, 1]
        for i = 1:rnum
            constMat[rIndex[i, 1], rIndex[i, 2]] = 0
        end
       
        return new(constMat, permMat, rIndex, rnum)
        
    end
    
end
