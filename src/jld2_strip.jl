# Python-free helpers of the artifact cleanup: strip the `recordings` key of a JLD2 model file
# without ever leaving the file deleted (needs DrWatson and JLD2 in the enclosing module).

_jld2_file(file) = DrWatson.FileIO.File{DrWatson.FileIO.format"JLD2"}(file)
_save_jld2(file, data) = DrWatson.save(_jld2_file(file), data)

# true if `x` holds a JLD2 placeholder (a type whose layout no longer exists, e.g. a model saved
# with an older SNNModels): such a file is not stripped, it is left untouched.
function _has_jld2_placeholder(x, seen = Base.IdSet{Any}())
    (x isa JLD2.ReconstructedMutable || x isa JLD2.ReconstructedStatic) && return true
    (x === nothing || x isa Union{Number,AbstractString,Symbol,Function,Type,Module}) && return false
    x isa AbstractArray && isbitstype(eltype(x)) && return false
    ismutable(x) && (x in seen ? (return false) : push!(seen, x))
    if x isa AbstractDict
        return any(v -> _has_jld2_placeholder(v, seen), values(x))
    elseif x isa AbstractArray || x isa Tuple
        return any(v -> _has_jld2_placeholder(v, seen), x)
    elseif isstructtype(typeof(x))
        return any(f -> isdefined(x, f) && _has_jld2_placeholder(getfield(x, f), seen), fieldnames(typeof(x)))
    end
    return false
end

"""
    strip_recordings!(file; saver = _save_jld2) -> Bool

Remove the `"recordings"` key of the JLD2 file `file`. The stripped copy is written to a temporary
file in the same directory and moved over `file` only after the save succeeded; if the save (or the
move) fails, `file` is untouched and the error is rethrown. Returns `false` (file untouched) if there
is no `"recordings"` key or the file holds JLD2 placeholders of an older layout (it is then not
rewritten); throws if the file cannot be read.
"""
function strip_recordings!(file::AbstractString; saver = _save_jld2)
    model = DrWatson.load(_jld2_file(file))
    haskey(model, "recordings") || return false
    if _has_jld2_placeholder(model)
        @warn "$file holds objects of an older layout (saved with an older package version, JLD2 placeholders); not stripping its recordings"
        return false
    end
    temp_model = Dict{String,Any}(k => v for (k, v) in pairs(model) if k != "recordings")
    tmp = string(file, ".stripping.tmp")
    try
        saver(tmp, temp_model)
        mv(tmp, file; force = true)
    catch
        rm(tmp; force = true)
        rethrow()
    end
    return true
end
