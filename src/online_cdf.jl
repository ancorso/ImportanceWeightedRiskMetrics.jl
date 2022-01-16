"""
Running cdf estimator
"""
mutable struct RunningCDFEstimator
    Xs::Vector{Float64}
    partial_Ws::Vector{Float64}
    last_i::Int
end

RunningCDFEstimator() = RunningCDFEstimator([], [], 0)

function RunningCDFEstimator(X::Vector{Float64}, W::Vector{Float64})
    # TODO: combine weights of duplicate entries in cost for efficiency
    perm = sortperm(X)
    Xs = X[perm]
    Ws = W[perm]
    partial_Ws = [sum(Ws[1:i]) for i=1:length(Ws)]
    sumW = sum(W)
    n = length(X)

    return RunningCDFEstimator(Xs, partial_Ws, n)
end

"""
CDF function
"""
function cdf(est::RunningCDFEstimator, x)
    idx = searchsortedlast(est.Xs, x)
    if idx < 1
        return 0.0
    else
        return est.partial_Ws[idx] / last(est.partial_Ws)
    end
end

"""
Quantile function
"""
quantile(est::RunningCDFEstimator, α::Float64) = est.Xs[searchsortedfirst(est.partial_Ws, (1 - α)*last(est.partial_Ws))]

"""
Update function
"""
function update!(est::RunningCDFEstimator, x, w)
    if est.last_i == 0
        push!(est.Xs, x)
        push!(est.partial_Ws, w)
    elseif x < first(est.Xs)
        pushfirst!(est.Xs, x)
        pushfirst!(est.partial_Ws, 0.0)
        est.partial_Ws = est.partial_Ws .+ w
    elseif x > last(est.Xs)
        push!(est.Xs, x)
        push!(est.partial_Ws, last(est.partial_Ws) + w)
    else
        new_idx = searchsortedlast(est.Xs, x) + 1
        if est.Xs[new_idx-1] == x   # special handle for efficient handling of exactly equal values
            new_idx = new_idx - 1
        else
            splice!(est.Xs, new_idx:new_idx-1, [x])
            splice!(est.partial_Ws, new_idx:new_idx-1, [est.partial_Ws[new_idx-1]])
        end
        est.partial_Ws[new_idx:end] = est.partial_Ws[new_idx:end] .+ w
    end
    est.last_i += 1
end
