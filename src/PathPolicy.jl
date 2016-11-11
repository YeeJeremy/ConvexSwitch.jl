################################################################################
## Copyright 2016. Juri Hinz and Jeremy Yee.
## Refine prescribed policy for a set of sample paths for all positions
################################################################################

immutable PathPolicy

    ## Members
    policy::Array{Int64, 3}
    pathnum::Int64 # number of paths
    tnum::Int64 # number of time points
    pnum::Int64 # number of positions
    anum::Int64 # number of actions
    snum::Int64 # state size
    
    function PathPolicy(css::CSS, bellman::Bellman, path::Path)

        ## Get sizes
        pathnum = path.pathnum
        pnum = bellman.pnum
        tnum = bellman.tnum
        anum = bellman.anum
        snum = bellman.snum

        ## Construct tree structure for neighbours
        tree = NearestNeighbors.KDTree(transpose(css.grid))
        
        ## Container
        policy = Array{Int64}(pathnum, pnum, tnum - 1)
        path_state = Array{Float64}(pathnum, snum)
        container = zero(Array{Float64}(anum, snum))
        neighbours = Array{Int64}(pathnum)

        ## Obtain the prescribed policy via recursion
        t = tnum - 1 # recall that last time is scrap
        if bellman.fullcontrol # positions fully controllable
            while (t > 0)
                path_state = path.sample[t, :, :]
                neighbours = knn(tree, transpose(path_state), 1, true)[1]
                for i = 1:pathnum
                    for p = 1:pnum
                        container = zero(container)
                        for a = 1:anum
                            container[a, :] =
                                bellman.evalue[neighbours[i], :, css.control[p, a], t]
                            container[a, :] += css.Reward(t, path_state[i, :], p, a)
                        end                        
                        policy[i, p, t] = indmax(container * path_state[i, :])
                    end
                end
                t = t - 1
            end
        else # positions not fully controllable
             while (t > 0)
                path_state = path.sample[t, :, :]
                neighbours = knn(tree, transpose(path_state), 1, true)[1]
                for i = 1:pathnum
                    for p = 1:pnum
                        container = zero(container)
                        for a = 1:anum
                            for pp = 1:pnum
                                container[a, :] += css.control[p, a, pp] *
                                    bellman.evalue[neighbours[i], :, css.control[p, a], t]
                            end
                            container[a, :] += css.Reward(t, path_state[i, :], p, a)
                        end                        
                        policy[i, p, t] = indmax(container * path_state[i, :])
                    end
                end
                t = t - 1
            end
        end 
        
        return new(policy, pathnum, tnum, pnum, anum, snum)
        
    end  
    
end
