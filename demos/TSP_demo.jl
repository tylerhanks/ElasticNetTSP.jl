### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ 1956c43c-bfff-11ec-3824-bf1e981d6274
begin
	using Revise
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()
	
	using ElasticNetTSP
	using Plots
	using TSPLIB
end;

# ╔═╡ 56b1d4c6-c156-49b6-9090-acc40bef238c
md"""# Elastic Net TSP Notebook"""

# ╔═╡ 6d08074f-61e1-419d-837a-fb44a60e5df6
md"""### Section 1: Basic Usage"""

# ╔═╡ ee854b39-703f-41c9-aae1-6a34839f604e
md"""Declare constant parameters for TSP algorithm."""

# ╔═╡ ab4d4987-5443-463c-b3df-5ec0145715b1
begin
	const N = 100
	const M = 3*N
	const β = 10
	const init_κ = 20
	const γ = 1.05
end;

# ╔═╡ a2a7408d-4e14-4d39-86ef-c54d450b3f22
md"""Create and visualize a TSP problem instance."""

# ╔═╡ a4c82f0c-107d-4c37-850c-434263e13afa
begin
	X, Y = make_prob(N, M)
	scatter(X[:,1], X[:,2], label="")
	scatter!(Y[:,1], Y[:,2], label="")
end

# ╔═╡ 652adbd2-ffcf-43a7-81c1-fcd5bb4a12b6
md"""Solve the problem with 1,000 iterations and visualize the hidden cities."""

# ╔═╡ 774c7bc7-3cfb-44b6-a3d7-3a8e4fef8ae5
begin
	solY, P = enet_TSP(X, Y, 1000, β, init_κ, γ)
	plot_sol(X, solY)
end

# ╔═╡ 0af64786-b3f1-4ce4-80fe-a948006a9c79
md"""Generate the tour for this solution. The post-processing step is important to ensure that the tour is valid (i.e. all cities are included)."""

# ╔═╡ 6ae85d1f-64a9-4ce1-8432-9cd4f1528aca
begin
	t = get_tour(X, P)
	post_process!(X, t)
	plot_tour(X,t)
end

# ╔═╡ f7a443d2-22a4-481e-ae68-78268f7d9c99
md"""The tour length can also be computed."""

# ╔═╡ 95240481-79d2-47d8-9bbf-77d6b84df61c
tour_length(t)

# ╔═╡ 005c499f-8d16-46a8-a745-996f5655672b
md"""### Section 2: 100 Trial Experiment"""

# ╔═╡ 435e05df-7901-4e18-87a5-7db888a4f404
md"""Run the Elastic Net TSP Algorithm for 1,000 iterations on 100 different problem instances while holding the hyperparameters fixed."""

# ╔═╡ 857a8f09-22b2-470e-bece-ebcee7c567af
# Function for doing many runs
function do_runs(num_runs, iters_per_run)
	tours = Pair{Matrix{Float64}, Vector{Vector{Float64}}}[]
	for i in 1:num_runs
		X, Y = make_prob(N, M)
		Y, P = enet_TSP(X, Y, iters_per_run, β, init_κ, γ)
		t = get_tour(X, P)
		push!(tours, X=>t)
	end
	return tours
end

# ╔═╡ 03b8c396-d08d-4ac7-afaa-9ecfaa88b793
tours = do_runs(100, 1000);

# ╔═╡ 77457fd1-b5d9-44ea-b92c-2b90844ace0a
md"""The following code determines which tours require post-processing to be valid."""

# ╔═╡ 0e457073-9b28-4da7-b9f4-dc2c8bb0c483
reduce((acc, cur)->acc && cur, map(x->is_tour(x...), tours), init=true)

# ╔═╡ 6ba8de66-2011-4f2e-ba8a-a4e1c6d2ff88
bad_tours = findall(x->x==false, map(x->is_tour(x...), tours))

# ╔═╡ f94bca81-cd9c-4b64-bb06-9cc4b74c6470
for i in bad_tours
	post_process!(tours[i]...)
end

# ╔═╡ 3baffa1c-145e-4627-a113-d157c913262d
md"""We can now see that all tours are valid by re-running the above `reduce` function."""

# ╔═╡ f2547e1a-ce8f-4621-aa6e-5ce30dfcfcdc
reduce((acc, cur)->acc && cur, map(x->is_tour(x...), tours), init=true)

# ╔═╡ fdfdb986-96fb-486d-bd26-7c4d51b5a4df
md"""Now that all 100 tours are valid, we can compute the length of each tour for analysis."""

# ╔═╡ 9e5b3431-ac1b-4160-85ef-74f529a4bc0c
lengths = map(t->tour_length(t[2]), tours);

# ╔═╡ 4ef4d584-58de-4849-8167-67a4e689c258
begin
	# Compute average tour length
	using Statistics
	mean(lengths)
end

# ╔═╡ 2f4aa2c7-86c5-4772-b176-24eef97c271b
md"""First, let's see a histogram of tour lengths along with the average tour length over the 100 runs."""

# ╔═╡ 99705bd2-a351-4e02-ae8a-e588c004971a
histogram(lengths, bins=20, label="", yticks=2:2:12, xlabel="Tour length", ylabel="Num tours")

# ╔═╡ 89eebe0c-2628-42a7-acc6-b4c7acd1c3fa
md"""Now we can find and visualize the min, max, and median length tours."""

# ╔═╡ fa83fdca-454f-439e-8233-81cabd15a9e7
min_length, min_ind = findmin(lengths)

# ╔═╡ 0e64df92-8902-4a22-91eb-e2c4cfdec6c5
begin
	min_tour = tours[min_ind]
	plot_tour(min_tour...)
	plot!(title="Minimum Length Tour")
end

# ╔═╡ e9051ef2-000e-496f-b160-27cf73c4b9e7
max_length, max_ind = findmax(lengths)

# ╔═╡ cea860b2-25e1-4d82-b938-69a4de79c6cc
begin
	max_tour = tours[max_ind]
	plot_tour(max_tour...)
	plot!(title="Maximum Length Tour")
end

# ╔═╡ 0b0fff3c-24ae-409f-846e-c7bffccbac3c
md"""Note that since we did an even number of runs, the median is the average length of two runs; so I have plotted both runs that were averaged to compute the median."""

# ╔═╡ 44642e3e-a157-4112-9a91-0b88e7c3a921
md"""### Section 3 (Extra Credit): American 48 Cities TSP"""

# ╔═╡ 84236ab3-1123-4fc7-abd9-d8314357dcf2
md"""First we load the problem instance from TSPLIB."""

# ╔═╡ 1f120dc6-0b34-4774-867d-585a09c927fb
begin
	us48X = readTSPLIB(:att48).nodes
	scatter(us48X[:,1], us48X[:,2], label="")
end

# ╔═╡ 190274ac-9edf-4214-92db-bb4444e8c37b
begin
	using LinearAlgebra
	normalize!(us48X,Inf)
	centroid = [mean(us48X[:,1]), mean(us48X[:,2])]
	start = [centroid[1]+.3, centroid[2]-.55]
	us48Y = make_circle(start, 0.26, 3*48)
	scatter(us48X[:,1], us48X[:,2], label="")
	scatter!(us48Y[:,1], us48Y[:,2], label="")
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
	plot!(title="Median Length Tour 1")
end

# ╔═╡ d3ffb2f9-2ccc-4181-afbb-40635f65cdec
begin
	rmed_ind = findfirst(x->x==med_right, lengths)
	rmed_tour = tours[rmed_ind]
	plot_tour(rmed_tour...)
	plot!(title="Median Length Tour 2")
end

# ╔═╡ 5907c2d0-6f52-4449-96e5-8f2f82db2e4e
begin
	us48solY,us48P = enet_TSP(us48X, us48Y, 1000, 10, 20, 1.05)
	plot_sol(us48X, us48solY)
end

# ╔═╡ 8d392640-1939-43c4-b343-31fae390a834
begin
	us48t = get_tour(us48X, us48P)
	post_process!(us48X, us48t)
	plot_tour(us48X, us48t)
	plot!(title="Elastic Net Optimal Solution")
end

# ╔═╡ b0ea4730-65e3-4d7c-b3a4-d86c75d7148a
is_tour(us48X, us48t)

# ╔═╡ 0177c2ba-2bb8-4729-8dd2-375f984b899b
my_length = tour_length(us48t)

# ╔═╡ 47acd263-2ac5-470f-9c1b-54d1bee35c72
begin
	opt_indices = [
		1,8,38,31,44,18,7,28,6,37,19,27,17,43,30,36,46,33,20,47,21,32,39,48,5,42,24,
		10,45,35,4,26,2,29,34,41,16,22,3,23,14,25,13,11,12,15,40,9,1
	]
	opt_tour = [us48X[1,:]]
	for i in opt_indices[2:end]
		push!(opt_tour, us48X[i,:])
	end
	plot_tour(us48X, opt_tour)
	plot!(title="True Optimal Solution")
end

# ╔═╡ 4c2052fd-d14c-42c2-9d7c-21adb4d7cfa0
opt_length = tour_length(opt_tour)

# ╔═╡ da4b4f19-1b4b-4792-9df6-12b72d9b75d5
error = (my_length - opt_length) / opt_length * 100

# ╔═╡ 19248c78-99b5-4c34-88d1-4e3bcd6c6fbf
md"""Due to overfitting of initial conditions to this problem, I was able to achieve a tour that is only 3.7% worse than the optimal tour!"""

# ╔═╡ 1804d9f1-b897-4954-b0f1-22fb30da460c
md"""### Section 4 (Extra Credit): Empirical Convergence of Loss Function"""

# ╔═╡ 0a380a3b-3a4b-4b61-bbbd-2b649b0682d3
md"""First we must implement the loss function in code."""

# ╔═╡ 2581fafd-4dc2-4ad9-a722-ee36af31a0eb
function E(X, P, P_old, Y, β, κ)
	loss = 0
	N = size(X)[1]
	M = size(Y)[1]
	for i in 1:N
		for a in 1:M
			l1 = P[i,a]*norm(X[i,:] - Y[a,:])^2
			l2 = (P[i,a]*log(P[i,a]/P_old[i,a]) - P[i,a] + P_old[i,a]) / β
			loss += l1 + l2
		end
	end
	l3 = 0
	for a in 1:M
		a1 = a+1 <= M ? a+1 : 1
		l3 += norm(Y[a,:] - Y[a1,:])^2
	end
	l3 = l3 * κ / 2
	loss += l3
	return loss
end

# ╔═╡ a4059f3d-f9db-4849-a856-9e5a08a590dd
begin
	tX, tY = make_prob(N, M)
	κ = init_κ
	losses1 = enet_TSP_losses(E, tX, tY, 20, β, κ)
	plot(losses1, xlabel="Iterations", ylabel="Loss", label="", title="κ = $κ")
end

# ╔═╡ ce20e088-80b4-4d0e-bed2-fc2d3a4aa355
begin
	tγ = 1.5
	κ1 = κ / tγ
	Y1,P1 = enet_TSP(tX, tY, 20, β, κ1, 1)
	losses2 = enet_TSP_losses(E, tX, Y1, 20, β, κ1, P_o=P1)
	plot(losses2, xlabel="Iterations", ylabel="Loss", label="", title="κ = $κ1")
end

# ╔═╡ ba014ff8-994a-4db9-85dc-e05f91aa8967
begin
	κ2 = κ1 / tγ
	Y2,P2 = enet_TSP(tX, Y1, 20, β, κ2, 1)
	losses3 = enet_TSP_losses(E, tX, Y2, 20, β, κ1, P_o=P2)
	plot(losses3, xlabel="Iterations", ylabel="Loss", label="", title="κ = $κ2")
end

# ╔═╡ Cell order:
# ╟─56b1d4c6-c156-49b6-9090-acc40bef238c
# ╟─1956c43c-bfff-11ec-3824-bf1e981d6274
# ╟─6d08074f-61e1-419d-837a-fb44a60e5df6
# ╟─ee854b39-703f-41c9-aae1-6a34839f604e
# ╠═ab4d4987-5443-463c-b3df-5ec0145715b1
# ╟─a2a7408d-4e14-4d39-86ef-c54d450b3f22
# ╠═a4c82f0c-107d-4c37-850c-434263e13afa
# ╟─652adbd2-ffcf-43a7-81c1-fcd5bb4a12b6
# ╠═774c7bc7-3cfb-44b6-a3d7-3a8e4fef8ae5
# ╟─0af64786-b3f1-4ce4-80fe-a948006a9c79
# ╠═6ae85d1f-64a9-4ce1-8432-9cd4f1528aca
# ╟─f7a443d2-22a4-481e-ae68-78268f7d9c99
# ╠═95240481-79d2-47d8-9bbf-77d6b84df61c
# ╟─005c499f-8d16-46a8-a745-996f5655672b
# ╟─435e05df-7901-4e18-87a5-7db888a4f404
# ╠═857a8f09-22b2-470e-bece-ebcee7c567af
# ╠═03b8c396-d08d-4ac7-afaa-9ecfaa88b793
# ╟─77457fd1-b5d9-44ea-b92c-2b90844ace0a
# ╠═0e457073-9b28-4da7-b9f4-dc2c8bb0c483
# ╠═6ba8de66-2011-4f2e-ba8a-a4e1c6d2ff88
# ╠═f94bca81-cd9c-4b64-bb06-9cc4b74c6470
# ╟─3baffa1c-145e-4627-a113-d157c913262d
# ╠═f2547e1a-ce8f-4621-aa6e-5ce30dfcfcdc
# ╟─fdfdb986-96fb-486d-bd26-7c4d51b5a4df
# ╠═9e5b3431-ac1b-4160-85ef-74f529a4bc0c
# ╟─2f4aa2c7-86c5-4772-b176-24eef97c271b
# ╠═99705bd2-a351-4e02-ae8a-e588c004971a
# ╠═4ef4d584-58de-4849-8167-67a4e689c258
# ╟─89eebe0c-2628-42a7-acc6-b4c7acd1c3fa
# ╠═fa83fdca-454f-439e-8233-81cabd15a9e7
# ╠═0e64df92-8902-4a22-91eb-e2c4cfdec6c5
# ╠═e9051ef2-000e-496f-b160-27cf73c4b9e7
# ╠═cea860b2-25e1-4d82-b938-69a4de79c6cc
# ╠═8734b32a-7e42-44fe-b58b-2a2812b94f9a
# ╠═9e07b42c-1e18-458e-a235-b4be5b32e5d3
# ╠═d3ffb2f9-2ccc-4181-afbb-40635f65cdec
# ╟─0b0fff3c-24ae-409f-846e-c7bffccbac3c
# ╟─44642e3e-a157-4112-9a91-0b88e7c3a921
# ╟─84236ab3-1123-4fc7-abd9-d8314357dcf2
# ╠═1f120dc6-0b34-4774-867d-585a09c927fb
# ╠═190274ac-9edf-4214-92db-bb4444e8c37b
# ╠═5907c2d0-6f52-4449-96e5-8f2f82db2e4e
# ╠═8d392640-1939-43c4-b343-31fae390a834
# ╠═b0ea4730-65e3-4d7c-b3a4-d86c75d7148a
# ╠═0177c2ba-2bb8-4729-8dd2-375f984b899b
# ╠═47acd263-2ac5-470f-9c1b-54d1bee35c72
# ╠═4c2052fd-d14c-42c2-9d7c-21adb4d7cfa0
# ╠═da4b4f19-1b4b-4792-9df6-12b72d9b75d5
# ╟─19248c78-99b5-4c34-88d1-4e3bcd6c6fbf
# ╟─1804d9f1-b897-4954-b0f1-22fb30da460c
# ╟─0a380a3b-3a4b-4b61-bbbd-2b649b0682d3
# ╠═2581fafd-4dc2-4ad9-a722-ee36af31a0eb
# ╠═a4059f3d-f9db-4849-a856-9e5a08a590dd
# ╠═ce20e088-80b4-4d0e-bed2-fc2d3a4aa355
# ╠═ba014ff8-994a-4db9-85dc-e05f91aa8967
