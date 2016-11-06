################################################################################
## Copyright 2016. Juri Hinz and Jeremy Yee.
## Stochastic grid
################################################################################

## Stochastic grid type definition

immutable SGrid

    ## Members
    snum::Int64 # state size
    gnum::Int64 # number of grid points
    maxIter::Int64 # max iterations used in the kmeans clustering 
    grid::Array{Float64, 2} # sample paths
    
    function SGrid(paths::Array{Float64, 3}, gnum::Int64, maxIter = 100) 

        ## Extract parameters
        snum = size(paths)[3];
        tnum = size(paths)[1];
        pnum = size(paths)[2];

        ## Collect states visited
        states = Matrix{Float64}(snum, tnum * pnum);
        i = 1;
        for t = 1:tnum
            for p = 1:pnum
                states[:, i] = paths[t,p,:]
                i = i + 1;
            end
        end
        
        ## apply kmeans clustering
        clusters = kmeans(states, gnum, maxiter = maxIter)
        
        return new(snum, snum, maxIter, transpose(clusters.centers))

    end
    
end
