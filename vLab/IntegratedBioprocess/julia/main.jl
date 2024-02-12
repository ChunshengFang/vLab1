#=
main:
- Julia version:
- Author: Hua
- Date: 2022-09-30
=#


include("bioreactor.jl")
include("Column_Rotavirus.jl")
include("harvest_tank.jl")
include("Plantwise.jl")

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
        "--time"
            help = "current time (hour)"
            arg_type = Float64
            required = false
            default = 0.0
        "--prediction_time"
            help = "future time to predict (hour)"
            arg_type = Float64
            required = false
            default = 720.0
        "--path"
            help = "root path to save"
            arg_type = String
            required = true
    end

    return parse_args(s)
end

function main()
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

    a_g1 = 0.01 * 10; a_m1 = 0.1 * 10;
    a_g2 = 0.05 * 10; a_m2 = 0.001 * 10;
    a_g3 = 0.1 * 10; a_m3 = 0.01 * 10;
    theta_P = [a_g1;a_m1;a_g2;a_m2;a_g3;a_m3];

    theta_C = Dict()

    theta_C[:n]               = 30
    theta_C[:area]            = [0.785;0.785];
    theta_C[:length]          = 6.5;
    theta_C[:D]               = 1e-6*2400;
    theta_C[:epsilontotal]    = 0.75;
    theta_C[:epsilonpore]     = 0.45;

    theta_C[:K]               = [1;1;1]*2e4;
    theta_C[:kads]            = [1;1;1]*1e6;
    theta_C[:qsi]             = [50;50;85];
    theta_C[:elealpha]        = 0.5;
    theta_C[:elebeta]         = 20;

    xC0 = zeros(Float64, 10*theta_C[:n]+3,1)

#     Xv0 = 3.40; # Total cell density [10^5cell/L]
#     Glc0 = 30; # Glucose [mM]
#     Gln0 = 5; # Glutamine [mM]
#     Lac0 = 0; # Lactate [mM]
#     NH40 = 0; # NH4+ [mM]
#     V0 = 1.5; # [L]
#     P1 = 0; # g/L
#     P2 = 0; # g/L
#     P3 = 0; # g/L
#     x0 = [Xv0; Glc0; Gln0; Lac0; NH40; P1; P2; P3; V0; P1; P2; P3; 1.e-08; xC0];

    x0 = [parsed_args["Xv0"]; parsed_args["Glc0"]; parsed_args["Gln0"];
          parsed_args["Lac0"]; parsed_args["NH40"];
          parsed_args["P10"]; parsed_args["P20"];
          parsed_args["P30"]; parsed_args["VB0"];
          parsed_args["P10"]; parsed_args["P20"];
          parsed_args["P30"]; parsed_args["VH0"]; xC0]

    F0 = 0.1*60/1000;
    Sin_g0 = 80;
    Sin_m0 = 40;

    u_1 = [0;0;0;0;0;0;0]; u_Cing1 = [0;0;0];
    u_2 = [F0;F0;0;0;0;0;0]; u_Cing2 = [0;Sin_m0;0];
#     u_Fg1 = [0;0;0;0;0;0;0]; u_Cing1 = [0;0;0];
#     u_Fg2 = [F0;0;F0;0;0;0;0]; u_Cing2 = [Sin_g0;0;0];
#     u_Fm1 = [F0;0;F0;0;0;0;0]; u_Cinm1 = [0;Sin_m0;0];
#     u_Fm2 = [F0;F0;0;0;0;0;0]; u_Cinm2 = [0;Sin_m0;0];
    u_Fl = [F0;F0;0;2*F0;0;0;0]; u_Cinl = [0;Sin_m0;0];
    u_Fw = [F0;F0;0;0;2*F0;0;0]; u_Cinw = [0;Sin_m0;0];
    u_Fe = [F0;F0;0;0;0;2*F0;2*F0]; u_Cine = [0;Sin_m0;1];

    t0 = 0  # initial time
#     tg1 = 22 * 4  # glycerol batch period (h)
#     tg2 = 10 * 4  # glycerol perfusion period (h)
#     tm1 = 8 * 4  # methanol perfusion period (h)
#     tm2 = 20 * 4  # methanol perfusion period (h)
    t1 = 56
    tl = 3  # load period (h)
    tw = 1  # wash period (h)
    te = 6  # elute period (h)
    rep = 3
    t2 = 720 - 56 - (tl + tw + te) * 3
    tran = cumsum([t0 t1 t2 repeat([tl tw te],1,rep)], dims=2)
    u_F = [u_1 u_2 repeat([u_Fl u_Fw u_Fe],1,rep)]
    u_Cin = [u_Cing1 u_Cing2 repeat([u_Cinl u_Cinw u_Cine],1,rep)]


    tbatch = 56; tperfusion = 30*24; # [h]
    t = [0;tbatch;tperfusion];

    Fin = parsed_args["Fin"] # Fin = 1.5*60/1e3;
    Foutb = Fin*0.2; Foutp = Fin*0.8; # [L/h]
    Glcin = parsed_args["Glcin"]; Glnin = parsed_args["Glnin"]; # Glcin = 50; Glnin = 6;
    Lacin = 0; NH4in = 0; # [mM]
    ubatch = zeros(1,7); uperfusion = [Fin Foutb Foutp Glcin Glnin Lacin NH4in];
    u = [ubatch; uperfusion];

#     theta_M = [0.039, 0.01, 1, 0.047, 43, 6.51, 45.8, 6.51, 0.357, 0.974, 0.7, 0.67, 0.0063, 0.0692, 0.0032, 2.1, 0.00151]

    for (i, value) in enumerate(theta_M)
        rng = MersenneTwister(2022 + i);
        theta_M[i] = value + randn(rng, Float64) * value * noise
    end

    p = [theta_M,theta_P,theta_C,u_F,u_Cin,u,tran]
    t_span = (parsed_args["time"], parsed_args["prediction_time"])
    prob = ODEProblem(Plantwide, x0, t_span, p)
    sol = solve(prob, TRBDF2(), abstol=1e-6, reltol=1e-6)

    m = mapreduce(permutedims, vcat, sol.u)

    # end-to-end process data
    filename_x = string("data_x.csv")
    filename_t = string("data_t.csv")

    writedlm(joinpath(parsed_args["path"], filename_x), m)
    writedlm(joinpath(parsed_args["path"], filename_t), sol.t)

    # chromatography data
    filename_yplot = string("data_yplot.csv")
    filename_tC = string("data_tC.csv")
    tC = sol.t[sol.t .>= tran[3]]
    xC = m[sol.t .>= tran[3], 14:end]
    # yplot = xC[:, :(30 * 10)].reshape(nrows, 10, 30, order='F')
    # yplot = reshape(xC[:, 1:(30 * 10)], (length(tC), 10, 30))

    writedlm(joinpath(parsed_args["path"], filename_tC), tC)
    writedlm(joinpath(parsed_args["path"], filename_yplot), xC)

end

main()

"""
using Distributions, Random
Random.seed!(123)
d = Normal(0.16, 0.05)
rand(d, 20)

AFS = 14; LW = 2
KS_g = 0.1; KS_m = 0.1;

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

a_g1 = 0.01 * 10; a_m1 = 0.1 * 10;
a_g2 = 0.05 * 10; a_m2 = 0.001 * 10;
a_g3 = 0.1 * 10; a_m3 = 0.01 * 10;
theta_P = [a_g1;a_m1;a_g2;a_m2;a_g3;a_m3];

theta_C = Dict()

theta_C[:n]               = 30
theta_C[:area]            = [0.785;0.785];
theta_C[:length]          = 5;
theta_C[:D]               = 1e-6*3600;
theta_C[:epsilontotal]    = 0.75;
theta_C[:epsilonpore]     = 0.5;

theta_C[:K]               = [1;1;1]*2e4;
theta_C[:kads]            = [1;1;1]*1e6;
theta_C[:qsi]             = [80;80;10];
theta_C[:elealpha]        = 0.5;
theta_C[:elebeta]         = 20;


F0 = 0.5*60/1000;
Sin_g0 = 80;
Sin_m0 = 40;


u_1 = [0;0;0;0;0;0;0]; u_Cing1 = [0;0;0];
u_2 = [0;F0;0;0;0;0;0]; u_Cing2 = [0;Sin_m0;0];
u_Fl = [F0;F0;0;2*F0;0;0;0]; u_Cinl = [0;Sin_m0;0];
u_Fw = [F0;F0;0;0;2*F0;0;0]; u_Cinw = [0;Sin_m0;0];
u_Fe = [F0;F0;0;0;0;2*F0;2*F0]; u_Cine = [0;Sin_m0;1];



X0 = 3.4
Sg0 = 30
Sm0 = 5
P10 = 0
P20 = 0
P30 = 0
VB0 = 1.5
VH0 = 1e-8
xC0 = zeros(Float64, 10*theta_C[:n]+3,1)
Sl0 = 0
Amm0 = 0

x0 = [X0; Sg0; Sm0; Sl0; Amm0; P10; P20; P30; VB0; P10; P20; P30; VH0; xC0]


# t0 = 0
# tg1 = 2
# tg2 = 10
# tm1 = 8
# tm2 = 20
# tl = 3
# tw = 1
# te = 6
# rep = 3
t0 = 0  # initial time
t1 = 56
t2 = 720 - 56 - 30
tl = 3  # load period (h)
tw = 1  # wash period (h)
te = 6  # elute period (h)
rep = 3
tran = cumsum([t0 t1 t2 repeat([tl tw te],1,rep)], dims=2)
u_F = [u_1 u_2 repeat([u_Fl u_Fw u_Fe],1,rep)]
u_Cin = [u_Cing1 u_Cing2 repeat([u_Cinl u_Cinw u_Cine],1,rep)]


tbatch = 56; tperfusion = 30*24; # [h]
t = [0;tbatch;tperfusion];
Fin = 1.5*60/1e3; Foutb = Fin*0.2; Foutp = Fin*0.8; # [L/h]
Glcin = 50; Glnin = 6; Lacin = 0; NH4in = 0; # [mM]
ubatch = zeros(1,7); uperfusion = [Fin Foutb Foutp Glcin Glnin Lacin NH4in];
u = [ubatch; uperfusion];

#     theta_M = [0.039, 0.01, 1, 0.047, 43, 6.51, 45.8, 6.51, 0.357, 0.974, 0.7, 0.67, 0.0063, 0.0692, 0.0032, 2.1, 0.00151]

p = [theta_M,theta_P,theta_C,u_F,u_Cin,u,tran]
t_span = (0.0,720.0)
prob = ODEProblem(Plantwide, x0, t_span, p)
sol = solve(prob, CVODE_BDF(), abstol=1e-6, reltol=1e-6)
"""