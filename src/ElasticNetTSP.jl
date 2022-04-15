using Plots
using LinearAlgebra
using Statistics
using Noise

const N = 50
const M = 100

cities = rand(N,2)

scatter(cities[:,1], cities[:,2], label="")

centroid = [mean(cities[:,1]), mean(cities[:,2])]

#scatter!([centroid[1]], [centroid[2]], label="", color="Red", markersize=10)

hidden_cities = add_gauss(repeat(centroid', M, 1), 0.05)

scatter!(hidden_cities[:,1], hidden_cities[:,2], color="Red")

