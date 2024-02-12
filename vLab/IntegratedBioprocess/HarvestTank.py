def harvest_tank(t, x, u_F, u_Cin):
    """ Compute the derivatives of harvest simulator

    :param t: time
    :type t: float
    :param x:  process state at time t
    :type x: array
    :param u_F: flow rates of bioreactor, harvest tank and chromatograph (load/wash/elute) at time t
    :type u_F: array
    :param u_Cin: inlet concentrations at time t
    :type u_Cin: array
    :return: dxdt
    :rtype: array
    """
    ## States
    # P1: product concentration (g/L)
    # P2: product concentration (g/L)
    # P3: product concentration (g/L)
    # V:  Hold tank volume (L)
    P1, P2, P3, V = x
    ## Inputs
    Fin, Fout = u_F  # inlet/outlet flow rate from bioreactor (L/h)
    P1in = u_Cin[0]
    P2in = u_Cin[1]
    P3in = u_Cin[2]

    ## States equations
    dP1dt = Fin / V * (P1in - P1)
    dP2dt = Fin / V * (P2in - P2)
    dP3dt = Fin / V * (P3in - P3)
    dVdt = Fin - Fout

    dxdt = [dP1dt, dP2dt, dP3dt, dVdt]

    return dxdt
