################################################################################
## Copyright 2016. Juri Hinz and Jeremy Yee.
## Test the prescribed policy for a set of sample paths
################################################################################

function test_policy(css::CSS, path::Path, pathPolicy::PathPolicy, position::Int64)

    ## Get sizes
    pathnum = path.pathnum
    pnum = css.pnum
    tnum = css.tnum
    anum = css.anum
    snum = css.snum

    ## Container
    result = zeros(Array{Float64}(pathnum))
    state = Array{Float64}(1, snum)
    
    ## Test run of policy
    if css.fullcontrol # positions are fully controllable
        for i = 1:pathnum
            p = position
            for t = 1:(tnum - 1)
                state = path.sample[t, i, :]
                action = pathPolicy.policy[i, p, t]
                result[i] += dot(css.Reward(t, state, p, action), state)
                p = css.control[p, action]
            end
            result[i] += dot(css.Scrap(path.sample[tnum, i, :], p),
                             path.sample[tnum, i, :])
        end
    else
        ## Cumulative probabilities
        cum_prob = cumsum(css.control, 3)
        ## Test run of policy
        for i = 1:pathnum
            p = position
            for t = 1:(tnum - 1)
                state = path.sample[t, i, :]
                action = pathPolicy.policy[i, p, t]
                result[i] += dot(css.Reward(t, state, p, action), state)
                ## Get next position
                rand_unif = rand()
                for pp = 1:pnum
                    if (rand_unif <= cum_prob[p, action, pp])
                        p = pp
                        break
                    end
                end
            end
            result[i] += dot(css.Scrap(path.sample[tnum, i, :], p),
                             path.sample[tnum, i, :])
        end        
    end

    return result
    
end
