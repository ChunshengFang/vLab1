function Column_Rotavirus(t, x,θ_C,u_F,u_Cin)
    # Written by Moo Sun Hong
    # Modified by Marc D. Berliner on Jun 16, 2022

    # θ_C is a dictionary
    
    ## distributed states
    n = θ_C[:n]; # discretization has to be fixed due to adjoint structural analysis
    distributedstates = reshape(x[1:10n],10,n); # extract distributed states
    c = distributedstates[1:7,:]; # columns liquid phase conc (g/L)
    q = distributedstates[8:10,:]; # columns solid phase conc (g/L)
    
    ## chromatography computed terms
    # unit convention [L/h]
    dz=θ_C[:length]/n; # [cm]
    kdes=θ_C[:kads]./θ_C[:K]; # [1/h]
    u=u_F*1e3./θ_C[:area]; # [cm^2/h]
    u=[
        u[1]*ones(4)
        u[2]*ones(3)
    ]
    
    ## calculation of wall fluxes for 3<=p<=N-2
    # @. is a macro which makes all the functions in this line element-wise
    β03= @. 13/12*(c[:,3:end-2] - 2*c[:,4:end-1] + c[:,5:end-0])^2 +
        1/4*(3*c[:,3:end-2] -4*c[:,4:end-1] + c[:,5:end-0])^2
        
    β13= @. 13/12*(c[:,2:end-3] - 2*c[:,3:end-2] + c[:,4:end-1])^2 +
        1/4*(c[:,2:end-3] -c[:,4:end-1])^2
        
    β23= @. 13/12*(c[:,1:end-4] - 2*c[:,2:end-3] + c[:,3:end-2])^2 +
        1/4*(c[:,1:end-4] -4 *c[:,2:end-3] + 3*c[:,3:end-2])^2
        
    ϵ=1e-16
    
    α0= @. 3/10/(ϵ+β03)^2
    α1= @. 6/10/(ϵ+β13)^2
    α2= @. 1/10/(ϵ+β23)^2
    
    sumα=α0+α1+α2
    
    Ω0=α0./sumα
    Ω1=α1./sumα
    Ω2=α2./sumα
    
    ci0atpplushalf = @.  1/3 * c[:,3:end-2] + 5/6 * c[:,4:end-1] - 1/6  * c[:,5:end-0]
    ci1atpplushalf = @. -1/6 * c[:,2:end-3] + 5/6 * c[:,3:end-2] + 1/3  * c[:,4:end-1]
    ci2atpplushalf = @.  1/3 * c[:,1:end-4] - 7/6 * c[:,2:end-3] + 11/6 * c[:,3:end-2]
    
    ciatpplushalf=Ω0.*ci0atpplushalf .+ Ω1.*ci1atpplushalf .+ Ω2.*ci2atpplushalf
    
    ## calculation of wall fluxes for p==2 and p==N-1
    β02pe2= @. (c[:,3]-c[:,2])^2
    β12pe2= @. (c[:,2]-c[:,1])^2
    
    β02penm1= @. (c[:,end-0]-c[:,end-1])^2
    β12penm1= @. (c[:,end-1]-c[:,end-2])^2
    
    α0pe2= @. 2/3/(ϵ+β02pe2)^2
    α1pe2= @. 1/3/(ϵ+β12pe2)^2
    
    α0penm1= @. 2/3/(ϵ+β02penm1)^2
    α1penm1= @. 1/3/(ϵ+β12penm1)^2
    
    Ω0pe2= @. α0pe2/(α0pe2+α1pe2)
    Ω1pe2= @. α1pe2/(α0pe2+α1pe2)
    
    Ω0penm1= @. α0penm1/(α0penm1+α1penm1)
    Ω1penm1= @. α1penm1/(α0penm1+α1penm1)
    
    ci0at2half=@. +1/2*c[:,2]+1/2*c[:,3]
    ci1at2half=@. -1/2*c[:,1]+3/2*c[:,2]
    
    ci0atnmhalf=@. +1/2*c[:,end-1] + 1/2 *c[:,end-0]
    ci1atnmhalf=@. -1/2*c[:,end-2] + 3/2 *c[:,end-1]
    
    ciat2half=Ω0pe2.*ci0at2half+Ω1pe2.*ci1at2half
    ciatnmhalf=Ω0penm1.*ci0atnmhalf+Ω1penm1.*ci1atnmhalf
    
    cin=[u_Cin;c[1:3,end]]
    ciatpplushalf=[cin c[:,1] ciat2half ciatpplushalf ciatnmhalf c[:,end]]
    
    ## calculation of concentration gradients
    pcpz=(c[:,2:end-0].-c[:,1:end-1])./dz
    pcpz=[zeros(length(c[:,1])) pcpz zeros(length(c[:,end-0]))]
    
    ## column derivative computation
    
    # bilangmuir for A, moderated by a eluant concentration sigmoid on the
    # adsorption rate
    pqpθ1=(θ_C[:kads][1:2].*c[1:2,:]).*(θ_C[:qsi][1:2] .- repeat(sum(q[1:2,:];dims=1),2))./
        repeat((1 .+ exp.(-θ_C[:elebeta].*(θ_C[:elealpha] .- c[4,:])))', 2) .- kdes[1:2].*q[1:2,:]
        
    pqpθ2=θ_C[:kads][3].*c[6,:].*(θ_C[:qsi][3] .- q[3,:]) .-
        kdes[3].*q[3,:]
        
    pcpθ=-u.*(ciatpplushalf[:,2:end]-ciatpplushalf[:,1:end-1])./dz./θ_C[:epsilontotal] .+
        θ_C[:D].*(pcpz[:,2:end] .- pcpz[:,1:end-1])./dz./θ_C[:epsilontotal]
        
    pcpθ[1:2,:] = pcpθ[1:2,:]-pqpθ1.*θ_C[:epsilonpore]./θ_C[:epsilontotal]
    pcpθ[6,:] = pcpθ[6,:]-pqpθ2.*θ_C[:epsilonpore]./θ_C[:epsilontotal]
    
    ## derivative stacking
    dxCdt=[
        pcpθ
        pqpθ1
        pqpθ2'
    ]
    dxCdt=[dxCdt[:];u_F[2].*c[5:7,end]]
    
end