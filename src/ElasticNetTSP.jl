module ElasticNetTSP

using Plots
using LinearAlgebra
using Statistics
using Noise

export make_prob, enet_TSP, enet_TSP_losses, plot_sol, get_tour, plot_tour, tour_length, is_tour, post_process!, make_circle

const N = 100
const M = 3*N
const β = 10
const init_κ = 20
const γ = 1.05

function make_circle(center, radius, num_samples)
    step = (2*π) / num_samples
    θ = 0
    res = [radius * cos(θ) + center[1] radius * sin(θ) + center[2]]
    for _ in 2:num_samples
        θ += step
        x = radius * cos(θ) + center[1]
        y = radius * sin(θ) + center[2]
        res = vcat(res, [x y])
    end
    return res
end

"""
Initialize a TSP problem instance. 

Pass :noise to init to initialize hidden cities with gaussian noise
or :circle to initialize hidden cities in a circle around centroid

Returns a cities matrix and a hidden cities matrix
"""
function make_prob(N,M; init=:circle)
    X = rand(N,2)
    centroid = [mean(X[:,1]), mean(X[:,2])]

    if init==:circle
        Y = make_circle(centroid, 0.1, M)
    elseif init==:noise
        Y = add_gauss(repeat(centroid', M, 1), 0.05)
    end
    return X, Y
end

function softmax(X, Y, β, N)
    function p_i(i)
        diff = map(y -> X[i,:] - y, collect(eachrow(Y)))
        num = exp.(-β*norm.(diff).^2)
        return num ./ sum(num)
    end
    return hcat([p_i(i) for i in 1:N]...)'
end

function softmax(P_old, X, Y, β, N)
    function p_i(i)
        diff = map(y -> X[i,:] - y, collect(eachrow(Y)))
        num = P_old[i,:] .* exp.(-β*norm.(diff).^2)
        return num ./ sum(num)
    end
    return hcat([p_i(i) for i in 1:N]...)'
end 

function make_L(M)
    L_dv = 2*ones(M)
    L_ev = -1*ones(M-1)
    L_tridiag = SymTridiagonal(L_dv, L_ev)
    L = Array(L_tridiag)
    L[1,M] = -1
    L[M,1] = -1
    return L
end

function make_D(P, M)
    D_dv = [sum(P[:,a]) for a in 1:M]
    return Diagonal(D_dv)
end

function enet_TSP(cities, hidden_cities, iters, β, κ, γ; visualize=false, anim=nothing)
    X = cities
    Y = hidden_cities
    N = size(X)[1]
    M = size(Y)[1]
    P = softmax(X, Y, β, N)
    L = make_L(M)
    for _ in 1:iters
        Y = (κ*L + make_D(P,M)) \ (P' * X)
        P = softmax(P, X, Y, β, N)
        κ = κ / γ
        if visualize
            Y_plt = vcat(Y, Y[1,:]')
            plt = scatter(X[:,1], X[:,2], label="", color="Blue")
            scatter!(Y[:,1], Y[:,2], label="", color="Red", markershape=:rect, markersize=2)
            plot!(Y_plt[:,1], Y_plt[:,2], label="", color="Red")
            if anim == nothing
                display(plt)
            else
                frame(anim, plt)
            end
        end
    end
    return Y, P
end

function enet_TSP_losses(lossfn, cities, hidden_cities, iters, β, κ; P_o=nothing)
    X = cities
    Y = hidden_cities
    N = size(X)[1]
    M = size(Y)[1]
    if P_o == nothing
        P = softmax(X, Y, β, N)
        P_old = P
    else
        P = P_o
        P_old = P
    end
    L = make_L(M)
    losses = Float64[]
    for _ in 1:iters
        Y = (κ*L + make_D(P,M)) \ (P' * X)
        P = softmax(P, X, Y, β, N)
        push!(losses, lossfn(X, P, P_old, Y, β, κ))
        P_old = P
    end
    return losses
end

function plot_sol(X, Y)
    Y_plt = vcat(Y, Y[1,:]')
    plt = scatter(X[:,1], X[:,2], label="", color="Blue")
    scatter!(Y[:,1], Y[:,2], label="", color="Red", markershape=:rect, markersize=2)
    plot!(Y_plt[:,1], Y_plt[:,2], label="", color="Red")
    return plt
end

function is_tour(X, t)
    # Start by ensuring the tour is the correct length
    if length(t) != size(X)[1] + 1
        return false
    end

    # Now check that every city is in the tour
    for x in collect(eachrow(X))
        if x ∉ t
            return false
        end
    end
    return true
end

function post_process!(X, t)
    # Post-process to ensure a complete tour
    for x in collect(eachrow(X))
        if x ∉ t
            # Find nearest neighbor of x in tour
            nn = findmin(y->norm(y-x), t)[2]
            insert!(t, nn+1, x)
        end
    end
end

function get_tour(X, P)
    tour = Vector{Float64}[]
    for i in 1:size(P)[2]
        _, city = findmax(P[:,i])
        if X[city,:] ∉ tour
            push!(tour, X[city,:])
        end
    end
    push!(tour, tour[1])
    #post_process!(X, tour)
    return tour
end

function plot_tour(cities, t)
    t = mapreduce(permutedims, vcat, t)
    plot(t[:,1], t[:,2], label="")
    scatter!(cities[:,1], cities[:,2], label="")
end

function tour_length(t)
    l = 0
    for i in 2:length(t)
        l += norm(t[i] - t[i-1])
    end
    return l
end

end # module
