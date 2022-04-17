using Plots
using LinearAlgebra
using Statistics
using Noise

const N = 100
const M = 3*N
const β = 10
const init_κ = 20
const γ = 1.05

cities = rand(N,2)

plt = scatter(cities[:,1], cities[:,2], label="")

centroid = [mean(cities[:,1]), mean(cities[:,2])]

#scatter!([centroid[1]], [centroid[2]], label="", color="Red", markersize=10)

#hidden_cities = add_gauss(repeat(centroid', M, 1), 0.05)

function make_circle(center, radius, num_samples)
    step = 360.0 / num_samples
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

hidden_cities = make_circle(centroid, 0.1, M)

scatter!(hidden_cities[:,1], hidden_cities[:,2], color="Red")
display(plt)

#X = cities
#Y = hidden_cities

function softmax(X, Y, β)
    function p_i(i)
        diff = map(y -> X[i,:] - y, collect(eachrow(Y)))
        num = exp.(-β*norm.(diff).^2)
        return num ./ sum(num)
    end
    return hcat([p_i(i) for i in 1:N]...)'
end

function softmax(P_old, X, Y, β)
    function p_i(i)
        diff = map(y -> X[i,:] - y, collect(eachrow(Y)))
        num = P_old[i,:] .* exp.(-β*norm.(diff).^2)
        return num ./ sum(num)
    end
    return hcat([p_i(i) for i in 1:N]...)'
end 

L_dv = 2*ones(M)
L_ev = -1*ones(M-1)
L_tridiag = SymTridiagonal(L_dv, L_ev)
L = Array(L_tridiag)
L[1,M] = -1
L[M,1] = -1

function make_D(P)
    D_dv = [sum(P[:,a]) for a in 1:M]
    return Diagonal(D_dv)
end

function TSP(cities, hidden_cities, iters, β, κ, γ; visualize=false)
    X = cities
    Y = hidden_cities
    P = softmax(X, Y, β)

    for _ in 1:iters
        Y = (κ*L + make_D(P)) \ (P' * X)
        P = softmax(P, X, Y, β)
        κ = κ / γ
        if visualize
            Y_plt = vcat(Y, Y[1,:]')
            plt = scatter(X[:,1], X[:,2], label="", color="Blue")
            scatter!(Y[:,1], Y[:,2], label="", color="Red", markershape=:rect, markersize=2)
            plot!(Y_plt[:,1], Y_plt[:,2], label="", color="Red")
            display(plt)
        end
    end
    return Y, P
end

function plot_sol(X, Y)
    Y_plt = vcat(Y, Y[1,:]')
    plt = scatter(X[:,1], X[:,2], label="", color="Blue")
    scatter!(Y[:,1], Y[:,2], label="", color="Red", markershape=:rect, markersize=2)
    plot!(Y_plt[:,1], Y_plt[:,2], label="", color="Red")
    display(plt)
end

function get_tour(Y, P)
    tour = []
    prev_city = -1
    for i in 1:size(P)[2]
        _, city = findmax(P[:,i])
        if city != prev_city
            println("Day $i city: $city")
            push!(tour, Y[city,:])
            prev_city = city
        end
    end
    return tour
end

