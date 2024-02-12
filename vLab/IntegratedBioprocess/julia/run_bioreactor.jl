#=
run_bioreactor:
- Julia version: 
- Author: hua.zheng
- Date: 2022-10-28
=#

include("bioreactor.jl")


import Pkg; Pkg.add("Sundials"); Pkg.add("DifferentialEquations"); Pkg.add("BenchmarkTools"); Pkg.add("Distributions"); Pkg.add("ArgParse")
using Sundials
using DifferentialEquations
using BenchmarkTools
using DelimitedFiles
using ArgParse
using Dates
using Distributions, Random


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--noise"
            help = "noise"
            arg_type = Float64
            required = true
            default = 0.0
        "--Xv0"
            help = "initial cell density"
            arg_type = Float64
            required = true
        "--Glc0"
            help = "glucose concentration"
            arg_type = Float64
            required = true
        "--Gln0"
            help = "glutamine concentration"
            arg_type = Float64
            required = true
        "--Lac0"
            help = "lactate concentration"
            arg_type = Float64
            required = true
        "--NH40"
            help = "ammonion concentration"
            arg_type = Float64
            required = true
        "--P10"
            help = "protein concentration"
            arg_type = Float64
            required = true
        "--P20"
            help = "impurity 1 concentration"
            arg_type = Float64
            required = true
        "--P30"
            help = "impurity 2 concentration"
            arg_type = Float64
            required = true
        "--VB0"
            help = "bioreactor volume"
            arg_type = Float64
            required = true
        "--VH0"
            help = "harvest tank volume"
            arg_type = Float64
            required = true
            default = 1e-8
        "--Fin"
            help = "inlet flow rate"
            arg_type = Float64
            required = false
            default = 60.0/1e3
        "--Glcin"
            help = "inlet glucose concentration"
            arg_type = Float64
            required = false
            default = 50.0
        "--Glnin"
            help = "inlet glutamine concentration"
            arg_type = Float64
            required = false
            default = 6.0
        "--path"
            help = "root path to save"
            arg_type = String
            required = true
    end

    return parse_args(s)
end


function run_bioreactor()
    parsed_args = parse_commandline()
    args = []
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
        args = vcat(args, val)
    end
    noise = parsed_args["noise"]

    # Parameters
    mumax   = 0.08052; # [1/h]
    KGln    = 1.322; # [mM]
    YXGlc   = 3.410; # [1e8 cell/mmol]
    mGlc    = 0; # [mmol/1e12 cell/h]
    YXGln   = 11.34; # [1e8 cell/mmol]
    YLacGlc = 1.168; # [mol/mol]
    YNH4Gln = 0.6287; # [mol/mol]
    kdg     = 8.362; # [1e-3/h]
    q_mab = 0.00151; # E-12 g cell^-1 h^-1

    theta_M  = [mumax; KGln; YXGlc; mGlc; YXGln; YLacGlc; YNH4Gln; kdg; q_mab];

    x0 = [parsed_args["Xv0"]; parsed_args["Glc0"]; parsed_args["Gln0"];
          parsed_args["Lac0"]; parsed_args["NH40"];
          parsed_args["P10"]; parsed_args["P20"];
          parsed_args["P30"]; parsed_args["VB0"]]

#     Xv0 = 3.40;
#     Glc0 = 30;
#     Gln0 = 5;
#     Lac0 = 0;
#     NH40 = 0;
#     V0 = 1.5;
#     P1 = 0;
#     P2 = 0;
#     P3 = 0;
#     x0 = [Xv0; Glc0; Gln0; Lac0; NH40; P1; P2; P3; V0];

    tbatch = 56; tperfusion = 30*24; # [h]
    t = [0;tbatch;tperfusion];
    Fin = parsed_args["Fin"] # Fin = 1.5*60/1e3;
    Foutb = Fin*0.2; Foutp = Fin*0.8; # [L/h]
    Glcin = parsed_args["Glcin"]; Glnin = parsed_args["Glnin"]; # Glcin = 50; Glnin = 6;
    Lacin = 0; NH4in = 0; # [mM]
    ubatch = zeros(1,7); uperfusion = [Fin Foutb Foutp Glcin Glnin Lacin NH4in];
    u = [ubatch; uperfusion];

    for (i, value) in enumerate(theta_M)
        rng = MersenneTwister(2022 + i);
        theta_M[i] = value + randn(rng, Float64) * value * noise
    end

    tsim = t[1];
    xsim = x0';
    for ii in 2:length(t)
        p = [theta_M, u[ii-1,1:3], u[ii-1,4:end]];
        prob = ODEProblem(bioreactor_cho, x0, (t[ii-1], t[ii]), p);
        sol = solve(prob, CVODE_BDF(), abstol=1e-6, reltol=1e-6);
        m = mapreduce(permutedims, vcat, sol.u);
        tsim = [tsim;sol.t]; xsim = [xsim; m];
        x0 = xsim[end, :];
    end

    xsim = xsim[2:end, :];
    tsim = tsim[2:end, :];

    # end-to-end process data
    filename_x = string("data_x.csv")
    filename_t = string("data_t.csv")

    writedlm(joinpath(parsed_args["path"], filename_x), xsim)
    writedlm(joinpath(parsed_args["path"], filename_t), tsim)

end

run_bioreactor()
