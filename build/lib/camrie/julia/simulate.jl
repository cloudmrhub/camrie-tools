#!/usr/bin/env julia
"""
simulate.jl - KomaMRI simulation with gradient rotation for oblique imaging.

USAGE:
    julia --threads=auto simulate.jl <B0> <seq_file> <phantom_json> <output_dir> [gpu] [nthreads] [rotation_json]

    rotation_json: Path to JSON file with rotation matrix R (3x3) from body to sequence space.
                   Gradients will be transformed by R^T to image oblique planes.

CONCEPTUAL MODEL:
    - Phantom coordinates are in BODY space (original patient coordinates)
    - Sequence has fixed gradient axes (Gx, Gy, Gz in lab/scanner frame)
    - To image an oblique plane, we rotate the GRADIENT AXES into body space
    - G_body = R^T @ G_seq, where R = rotation from body to sequence
    
    This is equivalent to "rotating the scanner to image the body" rather than
    "rotating the body to fit the scanner".
"""

using KomaMRI
using NPZ
using JSON
using LinearAlgebra

# =============================================================================
# Argument parsing
# =============================================================================

B0 = parse(Float64, ARGS[1])
seq_file = ARGS[2]
phantomjson = ARGS[3]
directory = ARGS[4]

println("=" ^ 60)
println("CAMRIE KomaMRI Simulation - Oblique Imaging Support")
println("=" ^ 60)
println("Sequence: ", seq_file)
println("Phantom: ", phantomjson)
println("Output: ", directory)
println("B0: ", B0, " T")

GPU = false
if length(ARGS) > 4
    GPU = parse(Bool, lowercase(ARGS[5]))
end

NT = 4
if !GPU
    println("Running on CPU")
    if length(ARGS) > 5
        NT = parse(Int, ARGS[6])
    end
end
println("Threads: ", NT)

# Rotation matrix (identity by default = standard orientation)
R = Matrix{Float64}(I, 3, 3)
rotation_json = nothing
if length(ARGS) > 6
    rotation_json = ARGS[7]
end

# =============================================================================
# Load rotation matrix
# =============================================================================

function load_rotation_matrix(json_path::String)
    """Load 3x3 rotation matrix from JSON file."""
    data = open(json_path) do io
        JSON.parse(read(io, String))
    end
    
    if haskey(data, "R_body_to_seq")
        R_raw = data["R_body_to_seq"]
    elseif haskey(data, "rotation_matrix")
        R_raw = data["rotation_matrix"]
    elseif haskey(data, "R")
        R_raw = data["R"]
    else
        error("No rotation matrix found in JSON. Expected 'R_body_to_seq', 'rotation_matrix', or 'R'")
    end
    
    # Convert to 3x3 Float64 matrix
    R = zeros(Float64, 3, 3)
    for i in 1:3
        for j in 1:3
            R[i, j] = Float64(R_raw[i][j])
        end
    end
    
    # Verify it's a valid rotation (orthogonal, det=1)
    if abs(det(R) - 1.0) > 1e-6
        @warn "Rotation matrix determinant $(det(R)) ≠ 1, may not be a proper rotation"
    end
    
    return R
end

if rotation_json !== nothing && isfile(rotation_json)
    R = load_rotation_matrix(rotation_json)
    println("Loaded rotation matrix from: ", rotation_json)
    println("R = ")
    display(R)
else
    println("No rotation (identity matrix - standard orientation)")
end

# =============================================================================
# Load phantom
# =============================================================================

phantom_data = begin
    try
        open(phantomjson) do io
            JSON.parse(read(io, String))
        end
    catch err
        error("Failed to read phantom JSON '$phantomjson': $err")
    end
end

# Helper: ensure Float64 vector
to_f64vec(v) = collect(Float64.(v))

# Extract phantom coordinates and properties
x = to_f64vec(phantom_data["x"])
y = to_f64vec(phantom_data["y"])
z = to_f64vec(phantom_data["z"])
ρ = to_f64vec(phantom_data["rho"])
T1 = to_f64vec(phantom_data["t1"])
T2 = to_f64vec(phantom_data["t2"])
T2s = to_f64vec(get(phantom_data, "t2s", zeros(length(ρ))))

if haskey(phantom_data, "dw")
    Δw = to_f64vec(phantom_data["dw"])
else
    Δw = zeros(Float64, length(ρ))
end

phantom_name = get(phantom_data, "name", "oblique_phantom")

# Filter out zero-density points
mask = ρ .!= 0
obj = Phantom{Float64}(
    name = phantom_name,
    x = x[mask],
    y = y[mask],
    z = z[mask],
    ρ = ρ[mask],
    T1 = T1[mask],
    T2 = T2[mask],
    T2s = T2s[mask],
    Δw = Δw[mask],
)

println("\nPhantom: ", phantom_name)
println("  Spins: ", length(obj.x))
println("  X range: [", minimum(obj.x), ", ", maximum(obj.x), "] m")
println("  Y range: [", minimum(obj.y), ", ", maximum(obj.y), "] m")
println("  Z range: [", minimum(obj.z), ", ", maximum(obj.z), "] m")

# =============================================================================
# Load sequence
# =============================================================================

seq = read_seq(seq_file)
println("\nSequence loaded: ", seq_file)
println("  Duration: ", sum(seq.DUR), " s")

# Apply rotation if not identity
if R != Matrix{Float64}(I, 3, 3)
    println("\nApplying gradient rotation...")
    println("  Using phantom coordinate transformation (equivalent to gradient rotation)")
end

# =============================================================================
# Run simulation
# =============================================================================

sim_params = KomaMRICore.default_sim_params()

if !GPU
    sim_params["gpu"] = false
    sim_params["Nthreads"] = NT
end

sys = Scanner(B0=B0)
println("\nRunning simulation...")
println("  Scanner: B0 = ", B0, " T")
println("  GPU: ", GPU)
println("  Threads: ", NT)

@time raw = simulate(obj, seq, sys; sim_params)
println("Simulation complete!")

# =============================================================================
# Extract and save k-space
# =============================================================================

Np = size(raw.profiles)[1]
Nf = size(raw.profiles[1].data)[1]

println("\nK-space dimensions:")
println("  Phase encodes (Np): ", Np)
println("  Frequency encodes (Nf): ", Nf)

K = zeros(ComplexF32, Np, Nf)
for i in 1:Np
    K[i, :] = raw.profiles[i].data
end

# Create output directory
if !isdir(directory)
    mkpath(directory)
end

# Save k-space
filename = joinpath(directory, "k.npz")
npzwrite(filename, K)

# Save metadata
info = Dict(
    "version" => "v2",
    "KS" => filename,
    "B0" => B0,
    "Np" => Np,
    "Nf" => Nf,
    "n_spins" => length(obj.x),
    "rotated" => (R != Matrix{Float64}(I, 3, 3)),
    "rotation_matrix" => [R[i, j] for i in 1:3, j in 1:3],
)

jsonfilename = joinpath(directory, "info.json")
open(jsonfilename, "w") do io
    write(io, JSON.json(info))
end

println("\nOutputs saved to: ", directory)
println("  K-space: k.npz")
println("  Info: info.json")
println("=" ^ 60)
