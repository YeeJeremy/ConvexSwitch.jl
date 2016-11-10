################################################################################
## Copyright 2016. Juri Hinz and Jeremy Yee.
## Obtain disturbance or path disurb matrices from modif, rIndex and skeleton
## disturb matrix if user not comfortable setting it themselves
################################################################################

function disturb_mat(W::Array{Float64, 2}, rIndex::Array{Int64, 2},
                     modif::Array{Float64, 2})

    ## Get sizes
    snum = size(W)[1]
    dnum = size(modif)[2]
    rnum = size(modif)[1]
    result = zeros(Array{Float64}(snum, snum, dnum))

    ## Construct distubances
    for k = 1:dnum
        result[:, :, k] = W
        for i = 1:rnum
            result[rIndex[i, 1], rIndex[i, 2], k] = modif[i, k]
        end                                                        
    end 

    return result
    
end


function path_mat(W::Array{Float64, 2}, rIndex::Array{Int64, 2},
                  modif::Array{Float64, 3})

    ## Get sizes
    snum = size(W)[1]
    dnum = size(modif)[2]
    rnum = size(modif)[1]
    tnum = size(modif)[3]
    result = zeros(Array{Float64}(snum, snum, dnum, tnum))

    ## Construct distubances
    for t = 1:tnum
        for k = 1:dnum
            result[:, :, k, t] = W
            for i = 1:rnum
                result[rIndex[i, 1], rIndex[i, 2], k, t] = modif[i, k, t]
            end                                                        
        end
    end

    return result
    
end
