################################################################################
## Copyright 2016. Juri Hinz and Jeremy Yee.
## Solutio diagnostics
################################################################################

immutable OptimalBounds

    ## Members
    upper::Array{Float64, 3} # upper bound realizations
    lower::Array{Float64, 3} # lower bound realizations
    uppermean::Array{Float64, 1} # mean upper bound
    lowermean::Array{Float64, 1} # mean lower bound
    upperse::Array{Float64, 1} # standard error of upper bound mean
    lowerse::Array{Float64, 1} # standard error of lower bound mean
   
    function OptimalBounds(css::CSS, path::Path, mart::Mart,
                           pathPolicy::PathPolicy)

        ## Get information
        tnum = css.tnum
        pnum = css.pnum
        pathnum = path.pathnum
        snum = css.snum
        anum = css.anum
        Scrap = css.Scrap
        Reward = css.Reward
        control = css.control

        ## Containers
        upper = zeros(Array{Float64}(tnum, pnum, pathnum))
        lower = zeros(Array{Float64}(tnum, pnum, pathnum))
        state = Array{Float64}(snum)

        print("Optimality Bounds: ", tnum, ".")
        ## Last time point
        for pos = 1:pnum
            for p = 1:pathnum              
                upper[tnum, pos, p] = dot(Scrap(path.sample[tnum, p, :], pos),
                                          path.sample[tnum, p, :])
            end
        end
        lower[tnum, :, :] = upper[tnum, :, :]

        ## Bellman recursion
        t = tnum - 1
        if css.fullcontrol
            while (t > 0)
                print(t, ".")
                for p = 1:pathnum
                    state = path.sample[t, p, :]
                    for pos = 1:pnum
                        ## Lower
                        action = pathPolicy.policy[p, pos, t]
                        next = control[pos, action]
                        lower[t, pos, p] = dot(Reward(t,state, pos, action), state) +
                        mart.mart[t, next, p] + lower[t + 1, next, p]
                        ## Upper
                        next = control[pos, 1]
                        upper[t, pos, p] = dot(Reward(t, state, pos, 1), state) +
                        mart.mart[t, next, p] + upper[t + 1, next, p]
                        ## Find pathwise maximising action
                        for a = 1:anum
                            next = control[pos, a]
                            temp = dot(Reward(t, state, pos, a), state) +
                            mart.mart[t, next, p] + upper[t + 1, next, p]
                            upper[t, pos, p] = maximum([temp; upper[t, pos, p]])
                        end                        
                    end
                end
                t = t - 1
            end                      
        else ## not full control of positions
            prob_weight = Array{Float64}(pnum)
            while (t > 0)
                print(t, ".")
                for p = 1:pathnum
                    state = path.sample[t, p, :]
                    for pos = 1:pnum
                        ## Lower
                        action = pathPolicy.policy[p, pos, t]
                        prob_weight = control[pos, action, :]
                        lower[t, pos, p] = dot(Reward(t, state, pos, action), state) +
                        mart.mart[t, pos, action, p] + dot(lower[t + 1, :, p], prob_weight)
                        ## Upper                        
                        prob_weight = control[pos, 1, :]
                        upper[t, pos, p] = dot(Reward(t, state, pos, 1), state) +
                        mart.mart[t, pos, 1, p] + dot(upper[t + 1, :, p], prob_weight)
                        ## Find pathwise maximising action
                        for a = 1:anum
                            prob_weight = control[pos, a, :]
                            temp = dot(Reward(t, state, pos, a), state) +
                            mart.mart[t, pos, a, p] + dot(upper[t + 1, :, p], prob_weight)
                            upper[t, pos, p] = maximum([temp; upper[t, pos, p]])
                        end                        
                    end
                end
                t = t - 1
            end             
        end
       
        ## Compute the means and standard errors
        uppermean = mean(upper[1, :, :], 2)[:]
        lowermean = mean(lower[1, :, :], 2)[:]
        upperse = (std(upper[1, :, :], 2) / sqrt(pathnum))[:]
        lowerse = (std(lower[1, :, :], 2) / sqrt(pathnum))[:]

        print("End.\n")
        
        return new(upper, lower, uppermean, lowermean, upperse, lowerse)
                
    end

end
