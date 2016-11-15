################################################################################
## Copyright 2016. Juri Hinz and Jeremy Yee.
## Define the type for martingale increments
################################################################################

immutable Mart

    ## Members
    mart::Array{Float64} # martingale increments
    method::String # slow, neighbours or discrete distribution
    pnum::Int64 # number of positions
    anum::Int64 # number of actions
    pathnum::Int64 # number of paths
    subsimnum::Int64 # number of subsimulation for each path
    tnum::Int64 ## number of time points
    fullcontrol::Bool
    
    ## Using only row rearrangement
    
    function Mart(bellman::Bellman, path::Path, subsim::Array{Float64, 5},
                  subsim_weight::Array{Float64, 1})

        method = "row-rearrange"

        ## Extract sizes from CSS
        pnum = bellman.pnum
        anum = bellman.anum
        pathnum = path.pathnum
        subsimnum = size(subsim)[4]
        tnum = bellman.tnum
        fullcontrol = bellman.fullcontrol
            
        ## Container
        result = zeros(Array{Float64}(tnum - 1, pnum, pathnum))
        path_state = Array{Float64}(bellman.snum)

        print("Martingales: ")
        ##  Compute martingales
        for t = 1:(tnum - 1)
            print(t, ".")
            for p = 1:pathnum
                path_state = path.sample[t, p, :]
                for pos = 1:pnum
                    ## Finding the average
                    for h = 1:subsimnum                       
                        result[t, pos, p] += subsim_weight[h] *
                            maximum(bellman.value[:, :, pos, t + 1] *
                                    subsim[:, :, p, h, t] * path_state)
                    end
                    ## Subtracting the path realization
                    path_state = path.sample[t + 1, p, :]
                    result[t, pos, p] -=
                        maximum(bellman.value[:, :, pos, t + 1] * path_state)
                end
            end    
        end

        if fullcontrol
            print("End.\n")
            return new(result, method, pnum, anum, pathnum, subsimnum, tnum,
                       fullcontrol)
        else # partial control of positions
            ## Container
            result2 = Array{Float64}(tnum - 1, pnum, anum, pathnum)
            prob_weight = Array{Float64}(pnum)            
            ## Adjust the martingales for partial control
            for t = 1:(tnum - 1)
                for pos = 1:pnum
                    for a = 1:anum
                        prob_weight = bellman.control[pos, a, :]
                        result2[t, pos, a, :] = transpose(prob_weight) * result[t, :, :]
                    end
                end
            end

            print("End.\n")
            return new(result2, method, pnum, anum, pathnum, subsimnum, tnum,
                       fullcontrol)

        end
        
    end

    ############################################################################

    ## Using nearest neighbours
    
    function Mart(bellman::Bellman, path::Path, subsim::Array{Float64, 5},
                  subsim_weight::Array{Float64, 1}, search::Function,
                  grid::Array{Float64, 2})

        method = "neighbors"

        ## Extract sizes from CSS
        pnum = bellman.pnum
        anum = bellman.anum
        pathnum = path.pathnum
        subsimnum = size(subsim)[4]
        tnum = bellman.tnum
        fullcontrol = bellman.fullcontrol
            
        ## Container
        result = zeros(Array{Float64}(tnum - 1, pnum, pathnum))
        path_state = Array{Float64}(bellman.snum)
        disturb_state = Array{Float64}(bellman.snum)

        ## Tree
        tgrid = transpose(grid)
        tree = search(tgrid)
        
        print("Martingales: ")
        ##  Compute martingales
        for t = 1:(tnum - 1)
            print(t, ".")
            for p = 1:pathnum
                path_state = path.sample[t, p, :]
                for pos = 1:pnum
                    ## Finding the average
                    for h = 1:subsimnum
                        disturb_state = subsim[:, :, p, h, t] * path_state
                        nn = knn(tree, disturb_state, 1, true)[1]
                        result[t, pos, p] += subsim_weight[h] *
                            (bellman.value[nn, :, pos, t + 1] * disturb_state)[1]
                    end
                    ## Subtracting the path realization
                    path_state = path.sample[t + 1, p, :]
                    nn = knn(tree, path_state, 1 ,true)[1]
                    result[t, pos, p] -=
                        (bellman.value[nn, :, pos, t + 1] * path_state)[1]
                end
            end    
        end

        if fullcontrol
            print("End.\n")
            return new(result, method, pnum, anum, pathnum, subsimnum, tnum,
                       fullcontrol)
        else # partial control of positions
            ## Container
            result2 = Array{Float64}(tnum - 1, pnum, anum, pathnum)
            prob_weight = Array{Float64}(pnum)            
            ## Adjust the martingales for partial control
            for t = 1:(tnum - 1)
                for pos = 1:pnum
                    for a = 1:anum
                        prob_weight = bellman.control[pos, a, :]
                        result2[t, pos, a, :] = transpose(prob_weight) * result[t, :, :]
                    end
                end
            end

            print("End.\n")
            return new(result2, method, pnum, anum, pathnum, subsimnum, tnum,
                       fullcontrol)

        end
        
    end


       ############################################################################

       ## Finite distribution case
    
       function Mart(bellman::Bellman, path::Path, path_disturb::Array{Float64, 4},
                     search::Function, grid::Array{Float64, 2})

        method = "finite-distribution"

        ## Extract sizes from CSS
        pnum = bellman.pnum
        anum = bellman.anum
        pathnum = path.pathnum
        subsimnum = 0
        tnum = bellman.tnum
        fullcontrol = bellman.fullcontrol
            
        ## Container
        result = zeros(Array{Float64}(tnum - 1, pnum, pathnum))
        path_state = Array{Float64}(bellman.snum)
        host = Array{Float64}(bellman.snum)
        disturb_host = Array{Float64}(bellman.snum)

        ## Tree
        tgrid = transpose(grid)
        tree = search(tgrid)
        
        print("Martingales: ")
        ##  Compute martingales
        for t = 1:(tnum - 1)
            print(t, ".")
            for p = 1:pathnum
                path_state = path.sample[t, p, :]
                nn1 = knn(tree, path_state, 1, true)[1]
                host = transpose(grid[nn1, :])
                disturb_host = path_disturb[:, :, t, p] * host
                nn2 = knn(tree, disturb_host, 1, true)[1][1]
                for pos = 1:pnum
                    ## Fining the average
                    result[t, pos, p] = (bellman.evalue[nn1, :, pos, t] * host)[1]
                    ## Subtracting the path realization
                    result[t, pos, p] -=
                        (bellman.value[nn2, :, pos, t + 1] * disturb_host)[1]
                    
                end
            end    
        end

        if fullcontrol
            print("End.\n")
            return new(result, method, pnum, anum, pathnum, subsimnum, tnum,
                       fullcontrol)
        else # partial control of positions
            ## Container
            result2 = Array{Float64}(tnum - 1, pnum, anum, pathnum)
            prob_weight = Array{Float64}(pnum)            
            ## Adjust the martingales for partial control
            for t = 1:(tnum - 1)
                for pos = 1:pnum
                    for a = 1:anum
                        prob_weight = bellman.control[pos, a, :]
                        result2[t, pos, a, :] = transpose(prob_weight) * result[t, :, :]
                    end
                end
            end

            print("End.\n")
            return new(result2, method, pnum, anum, pathnum, subsimnum, tnum,
                       fullcontrol)

        end
        
    end

end
