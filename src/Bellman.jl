################################################################################
## Copyright 2016. Juri Hinz and Jeremy Yee.
## Bellman recursion
################################################################################

## Bellman type definition

immutable Bellman

    ## Members
    value::Array{Float64, 4} # subgradient rep of value functions
    evalue::Array{Float64, 4} # subgradient rep of expected value functions    
    policy::Array{Int64, 3} # prescribed policy    
    method::String # slow, neighbours or fast method for continuation value
    grid::Array{Float64, 2} # the grid used
    control::Array{Float64} # prob for position transitions

    ## Parameters
    gnum::Int64 # grid size
    snum::Int64 # state size
    dnum::Int64 # disturbance sampling size
    pnum::Int64 # number of positions
    anum::Int64 # number of actions
    tnum::Int64 # number of time points
    fullcontrol::Bool # is the positions fully controllable

    ## Method using only row rearrangement
    
    function Bellman(css::CSS)
        
        method = "row-rearrange"        

        ## Extract info from CSS
        gnum = css.gnum
        snum = css.snum
        dnum = css.dnum
        pnum = css.pnum
        anum = css.anum
        tnum = css.tnum
        fullcontrol = css.fullcontrol
        grid = css.grid
        control = css.control

        ## Containers
        value = zero(Array{Float64}(gnum, snum, pnum, tnum))
        evalue = zero(Array{Float64}(gnum, snum, pnum, tnum - 1))
        policy = zero(Array{Int64}(gnum, pnum, tnum - 1))
        container = zero(Matrix{Float64}(anum, snum))
        
        ## Backwards induction
        print("Bellman: ", tnum, ".")
        for i = 1:gnum
            for p = 1:pnum
                value[i, :, p, tnum] = css.Scrap(grid[i, :], p)
            end
        end
        t = tnum - 1
        while (t > 0)
            print(t, ".")
            for p = 1:pnum
                evalue[:, :, p, t] = expected(value[:, :, p, t + 1], grid,
                                              css.disturb, css.weight)
            end
            for i = 1:gnum
                for p = 1:pnum
                    container = zero(container)
                    if css.fullcontrol # full control of positions
                        for a = 1:anum
                            container[a, :] = evalue[i, :, control[p, a], t]
                            container[a, :] += css.Reward(t, grid[i, :], p, a)
                        end                        
                    else # partial control of positions
                        for a = 1:anum
                            for pp = 1:pnum
                                container[a, :] += control[p, a, pp] * evalue[i, :, pp, t]
                            end
                            container[a, :] += css.Reward(t, grid[i, :], p, a)
                        end     
                    end
                    policy[i, p, t] = indmax(container * grid[i, :])
                    value[i, :, p, t] = container[policy[i, p, t], :]
                end
            end
            t = t - 1
        end

        print("End.\n")
        return new(value, evalue, policy, method, grid, control, gnum, snum,
                   dnum, pnum, anum, tnum, fullcontrol)

    end

    ############################################################################
    
    ## Method using nearest neighbours

    function Bellman(css::CSS, search::Function, k::Int64)
        
        method = "nearest-neighbors"        

        ## Extract info from CSS
        gnum = css.gnum
        snum = css.snum
        dnum = css.dnum
        pnum = css.pnum
        anum = css.anum
        tnum = css.tnum
        fullcontrol = css.fullcontrol
        grid = css.grid
        control = css.control

        ## Containers
        value = zero(Array{Float64}(gnum, snum, pnum, tnum))
        evalue = zero(Array{Float64}(gnum, snum, pnum, tnum - 1))
        policy = zero(Array{Int64}(gnum, pnum, tnum - 1))
        container = zero(Matrix{Float64}(anum, snum))

        ## Batch processing of neighbours for faster speed
        indices = Array{Array{Int64, 1}}(gnum * dnum)
        temp_indices = Array{Array{Int64, 1}}(gnum)
        tgrid = transpose(grid)
        tree = search(tgrid)
        for i = 1:dnum
            temp_indices = knn(tree, css.disturb[:, :, i] * tgrid, k, true)[1]
            indices[((i - 1) * gnum + 1):(i * gnum)] = temp_indices
        end        
        
        ## Backwards induction
        print("Bellman: ", tnum, ".")
        for i = 1:gnum
            for p = 1:pnum
                value[i, :, p, tnum] = css.Scrap(grid[i, :], p)
            end
        end
        t = tnum - 1
        while (t > 0)
            print(t, ".")
            for p = 1:pnum
                evalue[:, :, p, t] = expected_host(value[:, :, p, t + 1], grid,
                                                   css.disturb, css.weight, indices)
            end
            for i = 1:gnum
                for p = 1:pnum
                    container = zero(container)
                    if css.fullcontrol # full control of positions
                        for a = 1:anum
                            container[a, :] = evalue[i, :, control[p, a], t]
                            container[a, :] += css.Reward(t, grid[i, :], p, a)
                        end                        
                    else # partial control of positions
                        for a = 1:anum
                            for pp = 1:pnum
                                container[a, :] += control[p, a, pp] * evalue[i, :, pp, t]
                            end
                            container[a, :] += css.Reward(t, grid[i, :], p, a)
                        end     
                    end
                    policy[i, p] = indmax(container * grid[i, :])
                    value[i, :, p, t] = container[policy[i, p], :]
                end
            end
            t = t - 1
        end
        
        print("End.\n")
        return new(value, evalue, policy, method, grid, control, gnum, snum,
                   dnum, pnum, anum, tnum, fullcontrol)

    end
    
    ############################################################################
    
    ## Method using the permutation matrices

    function Bellman(css::CSS, X::PermMat)
        
        method = "fast"        

        ## Extract info from CSS
        gnum = css.gnum
        snum = css.snum
        dnum = css.dnum
        pnum = css.pnum
        anum = css.anum
        tnum = css.tnum
        fullcontrol = css.fullcontrol
        grid = css.grid
        control = css.control
        
        ## Containers
        value = zero(Array{Float64}(gnum, snum, pnum, tnum))
        evalue = zero(Array{Float64}(gnum, snum, pnum, tnum - 1))
        policy = zero(Array{Int64}(gnum, pnum, tnum - 1))
        container = zero(Matrix{Float64}(anum, snum))
        
        ## Backwards induction
        print("Bellman: ", tnum, ".")
        for i = 1:gnum
            for p = 1:pnum
                value[i, :, p, tnum] = css.Scrap(grid[i, :], p)
            end
        end
        t = tnum - 1
        while (t > 0)
            print(t, ".")
            for p = 1:pnum
                evalue[:, :, p, t] = expected_fast(value[:, :, p, t + 1], X)
            end
            for i = 1:gnum
                for p = 1:pnum
                    container = zero(container)
                    if css.fullcontrol # full control of positions
                        for a = 1:anum
                            container[a, :] = evalue[i, :, control[p, a], t]
                            container[a, :] += css.Reward(t, grid[i, :], p, a)
                        end                        
                    else # partial control of positions
                        for a = 1:anum
                            for pp = 1:pnum
                                container[a, :] += control[p, a, pp] * evalue[i, :, pp, t]
                            end
                            container[a, :] += css.Reward(t, grid[i, :], p, a)
                        end     
                    end
                    policy[i, p] = indmax(container * grid[i, :])
                    value[i, :, p, t] = container[policy[i, p], :]
                end
            end
            t = t - 1
        end
        
        print("End.\n")
        return new(value, evalue, policy, method, grid, control, gnum, snum,
                   dnum, pnum, anum, tnum, fullcontrol)

    end

end
