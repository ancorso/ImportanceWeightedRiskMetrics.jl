"""
Importance weighted risk metrics
"""
@with_kw struct IWRiskMetrics
    Z # cost data
    w # importance weights
    α # probability threshold

    𝒫 # emperical CDF
    est # CDF estimator

    mean # expected value
    var # Value at Risk
    bootstrap_vars # Bootstrap samples of value at risk
    var_cdf # cumulative distribution of VaR
    cvar # Conditional Value at Risk
    worst # worst case
end

function IWRiskMetrics(Z,w,α, Nbootstrap=10)
    # If no failures, no cost distribution.
    if length(Z) == 0
        Z = [Inf]
        w = [1.0]
    end
    est = RunningCDFEstimator(Z, w)
    𝒫 = (x) -> cdf(est, x)
    𝔼 = mean(Z .* w)
    var = VaR(est, α)
    var_samples, VaR_cdf = bootstrap_VaR_cdf(Z, w, α, Nbootstrap)
    cvar = CVaR(Z, w, var, α)
    return IWRiskMetrics(Z=Z, w=w, α=α, 𝒫=𝒫, est=est, mean=𝔼, var=var, bootstrap_vars=var_samples, var_cdf=VaR_cdf, cvar=cvar, worst=worst_case(Z, w))
end

"""
Importance weighted Value-at-Risk and Conditional Value-at-Risk
"""
function CVaR(Z, w, var, α)
    Z_tail = Z .- var
    n = length(Z)
    for i=1:n
        Z_tail[i] = Z_tail[i] > 0.0 ? Z_tail[i] : 0.0
    end
    cvar = var + sum(Z_tail .* w)/(n*α)
    return cvar
end
VaR(est, α) = quantile(est, α)

function bootstrap_VaR_cdf(Z, w, α, Nbootstrap)
    VaRs = Float64[]
    N = length(Z)
    for i=1:Nbootstrap
        indices = sample(1:N, N)
    
        est = RunningCDFEstimator(Z[indices], w[indices])
        push!(VaRs, VaR(est, α))
    end
    est = RunningCDFEstimator(VaRs, ones(Nbootstrap))
    VaRs, (x) -> cdf(est, x)
end


"""
Wrapper for worst-case value with weighting
"""
worst_case(Z) = maximum(Z)
worst_case(Z, w) = worst_case(Z)
