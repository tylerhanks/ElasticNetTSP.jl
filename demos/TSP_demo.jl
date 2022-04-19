### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ 1956c43c-bfff-11ec-3824-bf1e981d6274
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()
	
	using Revise
	using ElasticNetTSP
	using Plots
end;

# ╔═╡ ab4d4987-5443-463c-b3df-5ec0145715b1
# Declare constant parameters for TSP problem
begin
	const N = 100
	const M = 3*N
	const β = 10
	const init_κ = 20
	const γ = 1.05
end;

# ╔═╡ a4c82f0c-107d-4c37-850c-434263e13afa
# Create and visualize a TSP problem instance
begin
	X, Y = make_prob(N, M)
	scatter(X[:,1], X[:,2], label="")
	scatter!(Y[:,1], Y[:,2], label="")
end

# ╔═╡ 774c7bc7-3cfb-44b6-a3d7-3a8e4fef8ae5
# Lets solve the problem and visualize the hidden cities
begin
	a = Animation()
	solY, P = TSP(X, Y, 500, β, init_κ, γ, visualize=true, anim=a)
	#plot_sol(X, solY)
	gif(a)
end

# ╔═╡ 6ae85d1f-64a9-4ce1-8432-9cd4f1528aca
# Now we can generate the tour for this solution
begin
	t = get_tour(X, P)
	plot_tour(X,t)
end

# ╔═╡ 95240481-79d2-47d8-9bbf-77d6b84df61c
begin
	is_tour(X, t)
	tour_length(t)
	typeof(t)
end

# ╔═╡ 6159fe12-8835-4c35-87fb-5e87537947d7
typeof(X)

# ╔═╡ 857a8f09-22b2-470e-bece-ebcee7c567af
# Function for doing many runs
function do_runs(num_runs, iters_per_run)
	tours = Pair{Matrix{Float64}, Vector{Vector{Float64}}}[]
	for i in 1:num_runs
		X, Y = make_prob(N, M)
		Y, P = TSP(X, Y, iters_per_run, β, init_κ, γ)
		t = get_tour(X, P)
		push!(tours, X=>t)
	end
	return tours
end

# ╔═╡ 03b8c396-d08d-4ac7-afaa-9ecfaa88b793
tours = do_runs(100, 1000);

# ╔═╡ 0e457073-9b28-4da7-b9f4-dc2c8bb0c483
reduce((acc, cur)->acc && cur, map(x->is_tour(x...), tours), init=true)

# ╔═╡ 6ba8de66-2011-4f2e-ba8a-a4e1c6d2ff88
findall(x->x==false, map(x->is_tour(x...), tours))

# ╔═╡ 9e5b3431-ac1b-4160-85ef-74f529a4bc0c
lengths = map(t->tour_length(t[2]), tours);

# ╔═╡ 4ef4d584-58de-4849-8167-67a4e689c258
begin
	# Compute average tour length
	using Statistics
	mean(lengths)
end

# ╔═╡ fa83fdca-454f-439e-8233-81cabd15a9e7
min_length, min_ind = findmin(lengths)

# ╔═╡ 0e64df92-8902-4a22-91eb-e2c4cfdec6c5
begin
	min_tour = tours[min_ind]
	plot_tour(min_tour...)
end

# ╔═╡ e9051ef2-000e-496f-b160-27cf73c4b9e7
max_length, max_ind = findmax(lengths)

# ╔═╡ cea860b2-25e1-4d82-b938-69a4de79c6cc
begin
	max_tour = tours[max_ind]
	plot_tour(max_tour...)
end

# ╔═╡ 8734b32a-7e42-44fe-b58b-2a2812b94f9a
# Calculate the median length
begin
	sorted_lengths = sort(lengths)
	med_left = sorted_lengths[50]
	med_right = sorted_lengths[51]
	median = (med_left + med_right) / 2
end

# ╔═╡ 9e07b42c-1e18-458e-a235-b4be5b32e5d3
begin
	lmed_ind = findfirst(x->x==med_left, lengths)
	lmed_tour = tours[lmed_ind]
	plot_tour(lmed_tour...)
end

# ╔═╡ d3ffb2f9-2ccc-4181-afbb-40635f65cdec
begin
	rmed_ind = findfirst(x->x==med_right, lengths)
	rmed_tour = tours[rmed_ind]
	plot_tour(rmed_tour...)
end

# ╔═╡ Cell order:
# ╠═1956c43c-bfff-11ec-3824-bf1e981d6274
# ╠═ab4d4987-5443-463c-b3df-5ec0145715b1
# ╠═a4c82f0c-107d-4c37-850c-434263e13afa
# ╠═774c7bc7-3cfb-44b6-a3d7-3a8e4fef8ae5
# ╠═6ae85d1f-64a9-4ce1-8432-9cd4f1528aca
# ╠═95240481-79d2-47d8-9bbf-77d6b84df61c
# ╠═6159fe12-8835-4c35-87fb-5e87537947d7
# ╠═857a8f09-22b2-470e-bece-ebcee7c567af
# ╠═03b8c396-d08d-4ac7-afaa-9ecfaa88b793
# ╠═0e457073-9b28-4da7-b9f4-dc2c8bb0c483
# ╠═6ba8de66-2011-4f2e-ba8a-a4e1c6d2ff88
# ╠═9e5b3431-ac1b-4160-85ef-74f529a4bc0c
# ╠═4ef4d584-58de-4849-8167-67a4e689c258
# ╠═fa83fdca-454f-439e-8233-81cabd15a9e7
# ╠═0e64df92-8902-4a22-91eb-e2c4cfdec6c5
# ╠═e9051ef2-000e-496f-b160-27cf73c4b9e7
# ╠═cea860b2-25e1-4d82-b938-69a4de79c6cc
# ╠═8734b32a-7e42-44fe-b58b-2a2812b94f9a
# ╠═9e07b42c-1e18-458e-a235-b4be5b32e5d3
# ╠═d3ffb2f9-2ccc-4181-afbb-40635f65cdec
