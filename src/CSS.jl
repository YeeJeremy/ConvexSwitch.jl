################################################################################
## Copyright 2016. Juri Hinz and Jeremy Yee.
## Convex switching system model
################################################################################

## Define CSS type

immutable CSS
    
    ## Model
    grid::Array{Float64, 2} # (gnum, snum) grid
    weight::Array{Float64, 1} # (dnum) weights of sampled disturbances
    disturb::Array{Float64, 3} # (snum, snum, dnum) sampled distrubances
    control::Array # transition probabilities for positions
    Reward::Function # reward
    Scrap::Function # reward at last time point
    
    ## Parameters
    gnum::Int64 # grid size
    snum::Int64 # state size
    dnum::Int64 # disturbance sampling size
    pnum::Int64 # number of positions
    anum::Int64 # number of actions
    tnum::Int64 # number of time points
    fullcontrol::Bool # is the positions fully controllable
       
    ## Assigning the parameters
    function CSS(grid::Array{Float64, 2}, weight::Array{Float64, 1},
                 disturb::Array{Float64, 3}, control::Array,
                 Reward::Function, Scrap::Function, tnum::Int64)

        ## Extract parameters
        gnum = size(grid)[1]
        snum = size(grid)[2]
        dnum = length(weight)
        pnum = size(control)[1]
        anum = size(control)[2]

        if ndims(control) == 3
            fullcontrol = false
        elseif ndims(control) == 2
            fullcontrol = true
        else
            error("control not in correct format")
        end

        return new(grid, weight, disturb, control, Reward, Scrap, gnum, snum,
                   dnum, pnum, anum, tnum, fullcontrol)

    end

end
