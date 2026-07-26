
"""
    hypervolume_history(study::Py, ref_point::Vector{Float64}; step=10) -> (ns, hvs)

Return trial counts and cumulative hypervolume at each `step` completed trials.
Uses `optuna._hypervolume.compute_hypervolume` (WFG algorithm, exact).
"""
function hypervolume_history(study, ref_point::Union{Vector{Float64}, Nothing} = nothing; step::Int = 10)
    hv_fn      = pyimport("optuna._hypervolume").compute_hypervolume
    np         = pyimport("numpy")
    all_trials = pyconvert(Vector, study.trials)
    complete   = filter(t -> pyconvert(Int, t.state) == 1, all_trials)
    n          = length(complete)
    all_vals   = [pyconvert(Vector{Float64}, complete[i].values) for i in 1:n]
    all_mat    = Matrix{Float64}(reduce(hcat, all_vals)')  # (n, n_obj)
    ref_py     = if isnothing(ref_point)
        # ref point: worst value per objective + 10% slack
        np.array(vec(maximum(all_mat; dims=1)) .* 1.1)
    else
        np.array(ref_point)
    end
    ns  = collect(step:step:n)
    hvs = map(enumerate(ns)) do (i, k)
        @info "Hypervolume: trial $k/$n ($i/$(length(ns)))"
        mat_py = np.array(all_mat[1:k, :])
        pyconvert(Float64, hv_fn(mat_py, ref_py))
    end
    return ns, hvs
end

"""
    plot_hypervolume_history(study::Py; reference_point=nothing) -> Figure

Plot hypervolume history for a multi-objective Optuna study.
Reference point defaults to `[1.0, ...]` (one per objective), suitable when
all objectives are normalised to [0, 1].
"""
function plot_hypervolume_history(study::Py; reference_point = nothing)
    n_obj = pyconvert(Int, study.directions.__len__())
    ref   = isnothing(reference_point) ? fill(1.0, n_obj) : reference_point
    plotly_fig = optuna[].visualization.plot_hypervolume_history(study, pylist(ref))
    xs = pyconvert(Vector{Int},     plotly_fig.data[0].x)
    ys = pyconvert(Vector{Float64}, plotly_fig.data[0].y)

    fig = Figure(size = (500, 350))
    ax  = Axis(fig[1, 1];
        xlabel = "Trial",
        ylabel = "Hypervolume",
        title  = "Hypervolume history",
    )
    lines!(ax, xs, ys; color = :steelblue, linewidth = 2)
    return fig
end

"""
    identify_pareto(scores, directions) -> Vector{Int}

Return the indices of non-dominated (Pareto-optimal) solutions.

- `scores`: (n_solutions × n_objectives) matrix
- `directions`: vector of `:min` or `:max` per objective

All objectives are converted to maximisation internally before the dominance check.
"""
function identify_pareto(unsorted_scores, directions)
    scores = Float32.(copy(unsorted_scores))
    for i in axes(scores, 2)
        directions[i] == :min && (scores[:, i] .= -scores[:, i])
    end

    population_size = size(scores, 1)
    pareto_front    = trues(population_size)

    for i in 1:population_size
        for j in 1:population_size
            if all(scores[j, :] .>= scores[i, :]) && any(scores[j, :] .> scores[i, :])
                pareto_front[i] = false
                break
            end
        end
    end
    return findall(pareto_front)
end
