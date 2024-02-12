function bioreactor_julia_cho_cell(t, x, theta_M, theta_P, u_F, u_Cin)
    ## States
    # X:  biomass concentration (g/L)
    # Sg: glucose concentration (g/L)
    # Sm: glutamine concentration (g/L)
    # Sl: lactate concentration (g/L)
    # Amm: ammonium concentration
    # P1: product concentration (g/L)
    # P2: product concentration (g/L)
    # P3: product concentration (g/L)
    # V:  volume (L)
    X, Sg, Sm, Sl, Amm, P1, P2, P3, V = x

    ## Parameters
    mu_max, k_d, k_glc, k_gln, k_llac, k_lamm, k_Dlac, k_Damm, y_xglc, y_xgln, y_lacglc, y_ammgln, r_amm, m_glc, a_1, a_2, q_mab = theta_M

    # Related with product
    a_g1, a_m1, a_g2, a_m2, a_g3, a_m3 = theta_P[1:6]

    ## Inputs
    # Related with feed
    Fin, Fout, Sin_g, Sin_m = u_F[1], u_F[2], u_Cin[1], u_Cin[2]

    ## Cell Metabolism
    # Specific growth rate (1/h)
    mu = mu_max * Sg / (k_glc + Sg) * Sm / (k_gln + Sm) * k_llac / (k_llac + Sl) * k_lamm / (k_lamm + Amm)
    mu_d = k_d * Sl / (k_Dlac + Sl) * Amm / (k_Damm + Amm)
    # Specific rate of substrate consumption
    m_gln = (a_1 * Sm) / (a_2 + Sm)
    qS_g = (Sg > 0) * (X > 0) * ((mu - mu_d) / y_xglc + m_glc)
    qS_m = (Sm > 0) * (X > 0) * ((mu - mu_d) / y_xgln + m_gln)
    # inhibitors
    qS_l = y_lacglc * ((mu - mu_d) / y_xglc + m_glc)
    q_amm = y_ammgln * (mu - mu_d) / y_xgln - r_amm
    # qS_l = max(qS_l, 0)
    q_amm = max(q_amm, 0)

    # Specific product production rate (g/g-h)
    qP1 = a_g1 * Sg / (k_glc + Sg) + a_m1 * Sm / (k_gln + Sm)
    qP2 = a_g2 * Sg / (k_glc + Sg) + a_m2 * Sm / (k_gln + Sm)
    qP3 = a_g3 * Sg / (k_glc + Sg) + a_m3 * Sm / (k_gln + Sm)

    ## States equations
    dXdt = -(Fin - Fout) / V * X + max(mu - mu_d, 0) * X
    dSgdt = Fin / V * (Sin_g - Sg) - qS_g * X
    dSmdt = Fin / V * (Sin_m - Sm) - qS_m * X
    dSldt = -Fin / V * Sl + qS_l * X
    dAmmdt = -Fin / V * Amm + q_amm * X
    dP1dt = -Fin / V * P1 + q_mab * X * qP1 #(1 - mu / mu_max)
    dP2dt = -Fin / V * P2 + q_mab * X * qP2 #(1 - mu / mu_max)
    dP3dt = -Fin / V * P3 + q_mab * X * qP3 #(1 - mu / mu_max)
    dVdt = Fin - Fout

    dxdt = [dXdt; dSgdt; dSmdt; dSldt; dAmmdt; dP1dt; dP2dt; dP3dt; dVdt]
end


function bioreactor_cho(x, p, t)
    Xv = x[1]; # [1e8 cell/L]
    Glc = x[2]; Gln = x[3]; Lac = x[4]; NH4 = x[5]; # [mM]
    P1 = x[6]; P2 = x[7]; P3 = x[8]; # g/L
    V = x[9]; # [L]

    param, u_F, u_Cin = p
    # Parameters
    mumax = param[1]; # [1/h]
    KGln = param[2]; # [mM]
    YXGlc = param[3]; # [1e8 cell/mmol]
    mGlc = param[4]; # [mmol/1e12 cell/h]
    YXGln = param[5];
    YLacGlc = param[6]; # [mol/mol]
    YNH4Gln = param[7];
    kdg = param[8];
    q_mab = param[9]; # E-12 g cell^-1 h^-1

    # Inputs
    Fin = u_F[1]; Foutb = u_F[2]; Foutp = u_F[3]; # [L/h]
    Glcin = u_Cin[1]; Glnin = u_Cin[2];
    Lacin = u_Cin[3]; NH4in = u_Cin[4]; # [mM]

    # Model
    # Growth rate [1/h]
    mu = mumax*Gln/(KGln+Gln);
    # Death rate [1/h]
    mud = 0;
    # Specific Glucose consumption rate [mmol/1e8cell/h]
    qGlc = mu/YXGlc + mGlc*1e-4;
    # Specific Glutamine consumption rate [mmol/1e8cell/h]
    qGln = mu/YXGln;
    # Specific lactate production rate [mmol/1e8cell/h]
    qLac = YLacGlc*qGlc;
    # Specific ammonia production rate [mmol/1e8cell/h]
    qNH4 = YNH4Gln*qGln;

    dXvdt  = (Fin-Foutp)/V*(-Xv) + (mu-mud)*Xv;
    dGlcdt = Fin/V*(Glcin-Glc) - qGlc*Xv;
    dGlndt = Fin/V*(Glnin-Gln) - qGln*Xv - kdg*1e-3*Gln;
    dLacdt = Fin/V*(Lacin-Lac) + qLac*Xv;
    dNH4dt = Fin/V*(NH4in-NH4) + qNH4*Xv + kdg*1e-3*Gln;
    dP1dt = -Fin / V * P1 + q_mab * Xv * (1 - mu / mumax) / 6
    dP2dt = -Fin / V * P2 + q_mab / 1.5 * Xv * mu / mumax
    dP3dt = -Fin / V * P3 + q_mab / 3 * Xv * mu / mumax
    dVdt   = Fin - Foutb - Foutp;

    dxdt = [dXvdt; dGlcdt; dGlndt; dLacdt; dNH4dt; dP1dt; dP2dt; dP3dt; dVdt];
end