### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ 5b929600-cafe-11ec-1079-6b4a2595f072
using Markdown

# ╔═╡ 9acf4e14-bd39-4ad9-a289-07d07216cb5b
using InteractiveUtils

# ╔═╡ 08b06106-8aaa-49a7-8c4d-b7807d71b727
using DataStructures

# ╔═╡ 902bdec3-e6a8-45b8-bfd2-7cf9b922962b
###A Pluto.jl notebook###

# ╔═╡ 648536eb-dcf7-4f5b-aa94-79b65d03a6a2
# By Jonas Nghidengwa

# ╔═╡ 61fc997d-8010-44c2-8ff8-c69df8339829
md"# Artificial Intelligence Assignment two"

# ╔═╡ 5a5b9c02-d2d7-43d5-991d-5d9004f1d113
md"# Assignment Two"

# ╔═╡ 396279cf-6e6b-468e-8aa6-772e2279058d
struct Multi-Storey
	hasAgent::Bool
	hasParcels::Bool
end

# ╔═╡ 987a1e5c-5578-4466-b438-6c13872e8803
@enum Action ME MW MU MD CO

# ╔═╡ eec1f21a-abeb-41de-882c-79151a14c10a
struct State
	a::Office1
	b::Office2
	c::Office3
	d::Office4
	e::Office5
	f::Office6
	g::Office7
	h::Office8
	j::Office9
	k::Office10
end

# ╔═╡ 25b19487-c6c5-4cda-bcbb-20d6653b2608
struct Parcels
	state::State
	parent::Union{Empty, Parcel}
	action::Union{Empty, Action}
	
end

# ╔═╡ fb1b0cb4-ed07-4ded-9099-79f99267d322
md"## Transition Model"

# ╔═╡ 3ec115bb-4b67-42c5-8431-135525a9cfcd
function get_transactions(office::Office)
	state = office.state
	transitions = Dict{Action,Office}()
	if Office.a.hasAgent
		if Office.a.hasParcel
			transition[MU] = Parcel(
				State(Office(true,fase),state.b,state.c, state.d,state.e,state.f,state.g,state.h,state.j,state.k),
				office,
				MU
			)
		end
		transitions[ME] = Office(
			State(Square(false,state.a.hasParcel),Office(true, state.b.hasParcel), state.c, state.d, state.e, state.f, state.g, state.h, state.j, state.k),
			office
			ME
		)
	elseif state.b.hasAgent
		if state.b.hasParcels
			transitions[MU] = Office(
				State(state.a, Office(true,false),state.c,state.d, state.e, state.f, state.g, state.h, state.j, state.k,
					office
					MU
			)
	end
		transitions[MW] = Office(State(
			Office(true, state.a.hasParcels),
			Office(false, state.b.hasParcels),state.c,state.d, state.e, state.f, state.g, state.h, state.j, state.k),
			office,
			MW
	)
transitions[MD] = Office(State(
	state.a,
	Office(false, state.b.hasParcels),
	Office(true, state.c.hasParcels)),
	office,
	MW
)
	elseif state.c.hasAgent
		if state.c.hasParcels
			transitions[MU] = Office(State(
				state.a,
				state.b,
				Square(true,false)),
				office,
				MU
			)
		end
		transitions[CO] = Office(State(
			state.a,
			Square(true, state.b.hasParcels),
		Square(false, state.c.hasParcels)),
		office,
		CO
	)
end
	return transitions
end
function is_goal(state::State)
	return !state.a.hasParcels && !state.c.hasParcels
end


function get_path(office1::Office)
	path = [Office]
	while !isempty(office.parent)
		office = office.parent
		movefirst!(path, office)
	end
	return path
end


function solution(office::Office explored::Array)
	path =get_path(office)
	actions = []
	for office in path
		if !isempty(office.action)
			push!(actions, node.action)
		end
	end

	cost = length(actions)

	return cost, Found $(length(actions)) step solution in $(length(explored)) steps: $(move(actions, " ->"))"
end


function failure(message::String)
	return -1, message
end
	
		

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
InteractiveUtils = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"

[compat]
DataStructures = "~0.18.11"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═902bdec3-e6a8-45b8-bfd2-7cf9b922962b
# ╠═648536eb-dcf7-4f5b-aa94-79b65d03a6a2
# ╠═5b929600-cafe-11ec-1079-6b4a2595f072
# ╠═9acf4e14-bd39-4ad9-a289-07d07216cb5b
# ╠═08b06106-8aaa-49a7-8c4d-b7807d71b727
# ╠═61fc997d-8010-44c2-8ff8-c69df8339829
# ╠═5a5b9c02-d2d7-43d5-991d-5d9004f1d113
# ╠═396279cf-6e6b-468e-8aa6-772e2279058d
# ╠═987a1e5c-5578-4466-b438-6c13872e8803
# ╠═eec1f21a-abeb-41de-882c-79151a14c10a
# ╠═25b19487-c6c5-4cda-bcbb-20d6653b2608
# ╠═fb1b0cb4-ed07-4ded-9099-79f99267d322
# ╠═3ec115bb-4b67-42c5-8431-135525a9cfcd
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
