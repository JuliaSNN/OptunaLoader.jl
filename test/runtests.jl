# Python-free tests of the artifact cleanup helpers (`src/jld2_strip.jl`).
using Test, DrWatson, JLD2

module Strip
    using DrWatson, JLD2
    include(joinpath(@__DIR__, "..", "src", "jld2_strip.jl"))
end

_read(f) = DrWatson.load(Strip._jld2_file(f))

@testset "strip_recordings!" begin
    dir = mktempdir()
    mk(name, d) = (f = joinpath(dir, name); Strip._save_jld2(f, d); f)

    @testset "removes only the recordings key" begin
        f = mk("a.jld2", Dict{String,Any}("model" => [1.0, 2.0], "recordings" => rand(10), "info" => "x"))
        @test Strip.strip_recordings!(f)
        d = _read(f)
        @test !haskey(d, "recordings")
        @test d["model"] == [1.0, 2.0] && d["info"] == "x"
        @test readdir(dir) == ["a.jld2"]                       # no temporary file left
        @test !Strip.strip_recordings!(f)                      # nothing left to strip
    end

    @testset "a failing save keeps the original file" begin
        f = mk("b.jld2", Dict{String,Any}("model" => [3.0], "recordings" => [9.0]))
        before = read(f)
        failing(tmp, data) = (write(tmp, "partial"); error("disk full"))
        @test_throws ErrorException Strip.strip_recordings!(f; saver = failing)
        @test read(f) == before
        @test _read(f)["recordings"] == [9.0]
        @test !any(endswith(".tmp"), readdir(dir))               # the partial file is removed
    end

    @testset "old-layout placeholders are not rewritten" begin
        ph = JLD2.ReconstructedMutable{:OldThing,(:a,),Tuple{Int}}(Any[1])
        @test Strip._has_jld2_placeholder((; x = [ph]))
        @test !Strip._has_jld2_placeholder((; x = [1.0], y = Dict("k" => "v")))
        f = mk("c.jld2", Dict{String,Any}("model" => (; s = ph), "recordings" => [1.0]))
        before = read(f)
        @test_logs (:warn, r"older layout") match_mode = :any !Strip.strip_recordings!(f)
        @test read(f) == before
    end

    @testset "an unreadable file throws and is left alone" begin
        f = joinpath(dir, "d.jld2"); write(f, "not a jld2 file")
        @test_throws Exception Strip.strip_recordings!(f)
        @test read(f, String) == "not a jld2 file"
    end
end
