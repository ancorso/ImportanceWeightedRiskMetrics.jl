"""
Running cdf estimator
"""
mutable struct RunningCDFEstimator
    Xs::Vector{Float64} # stored in increasing order for easy sorting
    partial_Ws::Vector{Float64} # stored in reverse order cumsum for easy sorting (first entry corresponds to last Xs)
    partial_Xs::Vector{Float64} # stored in reverse order cumsum for easy sorting (first entry corresponds to last Xs)
    last_i::Int
end

RunningCDFEstimator() = RunningCDFEstimator([], [], [], 0)

function RunningCDFEstimator(X::Vector{Float64}, W::Vector{Float64})
    # TODO: combine weights of duplicate entries in cost for efficiency
    perm = sortperm(X, rev=true)
    Xs = X[perm]
    partial_Xs = cumsum(Xs)
    Xs = reverse(Xs)
    Ws = W[perm]
    partial_Ws = cumsum(Ws)

    n = length(X)

    return RunningCDFEstimator(Xs, partial_Ws, partial_Xs, n)
end

"""
CDF function
"""
function cdf(est::RunningCDFEstimator, x)
    idx = searchsortedlast(est.Xs, x)
    if idx < 1
        return 0.0
    elseif idx > length(est.Xs)
        return 1.0
    else
        return 1.0 - est.partial_Ws[end - idx + 1] / est.last_i
    end
end

"""
Quantile function
"""
function quantile(est::RunningCDFEstimator, α::Float64)
    idx = searchsortedlast(est.partial_Ws, α*est.last_i)
    if idx < 1
        idx = 1
    end
    return est.Xs[end - idx + 1]
end

"""
Tail cost
"""
function tail_cost(est::RunningCDFEstimator, x)
    idx = searchsortedlast(est.Xs, x)
    if idx < 1
        return 0.0
    elseif idx > length(est.Xs)
        return est.partial_Xs[1] / est.last_i
    else
        return (est.partial_Xs[1] - est.partial_Xs[end - idx + 1]) / est.last_i
    end
end

"""
Update function
"""
function update!(est::RunningCDFEstimator, x, w)
    if est.last_i == 0
        push!(est.Xs, x)
        push!(est.partial_Ws, w)
        push!(est.partial_Xs, x)
    elseif x < first(est.Xs)
        pushfirst!(est.Xs, x)
        push!(est.partial_Ws, last(est.partial_Ws) + w)
        push!(est.partial_Xs, last(est.partial_Xs) + x)
    elseif x > last(est.Xs)
        push!(est.Xs, x)
        pushfirst!(est.partial_Ws, 0.0)
        est.partial_Ws = est.partial_Ws .+ w
        pushfirst!(est.partial_Xs, 0.0)
        est.partial_Xs = est.partial_Xs .+ x
    else
        new_idx = searchsortedlast(est.Xs, x) + 1
        if est.Xs[new_idx-1] == x   # special handle for efficient handling of exactly equal values
            w_idx =  length(est.partial_Ws) - new_idx + 2
        else
            splice!(est.Xs, new_idx:new_idx-1, [x])
            w_idx = length(est.partial_Ws) - new_idx + 2
            splice!(est.partial_Ws, w_idx:w_idx-1, [est.partial_Ws[w_idx-1]])
            splice!(est.partial_Xs, w_idx:w_idx-1, [est.partial_Xs[w_idx-1]])
        end
        est.partial_Ws[w_idx:end] = est.partial_Ws[w_idx:end] .+ w
        est.partial_Xs[w_idx:end] = est.partial_Xs[w_idx:end] .+ x
    end
    est.last_i += 1
end
