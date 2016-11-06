################################################################################
## Copyright 2016. Juri Hinz and Jeremy Yee.
## Path simulation
################################################################################

## Path type definition

immutable Path

    ## Members
    snum::Int64 # state size
    tnum::Int64 # time points
    pnum::Int64 # number of paths
    sample::Array{Float64, 3} # sample paths

    function Path(start::Array{Float64, 1}, disturb::Array{Float64, 4})
        
        ## Extract parameters
        snum = length(start);
        tnum = size(disturb)[3] + 1;
        pnum = size(disturb)[4];
        sample = Array{Float64}(tnum, pnum, snum);
    
        ## Simulate paths
        for p = 1:pnum
            sample[1, p, :] = start;
            for t = 2:tnum
                sample[t, p, :] = disturb[:, :, t - 1, p] * sample[t - 1, p, :];
            end
        end

        return new(snum, tnum, pnum, sample);

    end

end
