def cell_culture_derivative(z, y, p, fp, feed):
    """ Solve the dynamic glycosylation model between two samplings

    :param array z: time interval between two samplings
    :param array y: perfusion cell culture state
    :param GlycosylationModelParamClass p: N-linked glycosylation model parameters
    :param GlycosylationNetwork fp: N-linked glycosylation network
    :param array feed: feeding conditions between two samplings

    :returns ndarray: updated cell culture states [volume, VCD, Mn, Amm, mAb, Galactose] and 33 oligsaccharides
    """

    # states are volume, VCD*V, manganese*V, ammonia*V, mab*V, Gal*V,
    # all the oligosaccharides
    timescale = 24 * 60  # hour/day
    # feed is feedin, perf, bleed, mn, gal
    Fin, Fharvest, Fbleed, Mn_in, Gal_in = feed[:5]
    Fin = Fin / timescale  # L/day to L/hour
    Fharvest = Fharvest / timescale  # L/day to L/hour
    Fbleed = Fbleed / timescale  # L/day to L/hour
    Fout = Fharvest + Fbleed
    feed_copy = feed.copy()

    V = y[0]
    VCD = y[1] / V
    Mn = y[2] / V
    Amm = y[3] / V
    mAb = y[4] / V
    Gal = y[5] / V
    mu = p.mumax * (p.k_amm / (p.k_amm + Amm)) / timescale  # 1/day to 1/min
    q_mab = p.mabtiter / 2 / 1e-12 * 1.12 * timescale / 1e15 * 150 * 1e3 / 1e6  # umol/L to pg/cell/day
    q_mab = q_mab / 1e12 * 1e6 * 1e3 / timescale  # pg/cell/day to g/(million cells per L)/hour

    feed_copy[5:(6 + fp.nos - 1)] = feed_copy[5:(
                6 + fp.nos - 1)] / 2 / 1e-12 * 1.12 * 60 * timescale / 1e15 * 150 * 1e3 / 1e6  # umol/L to pg/cell/day
    feed_copy[5:(6 + fp.nos - 1)] = feed_copy[5:(
                6 + fp.nos - 1)] / 1e12 * 1e6 * 1e3 / timescale  # pg/cell/day to g/(million cells per L)/hour

    # macroscopic derivatives
    deriv_macro = [0] * (7 + fp.nos - 1)
    deriv_macro[0] = Fin - Fout  # culture volume is feeds minus  outlet
    deriv_macro[1] = mu * V * VCD - Fbleed * VCD  # VCD bal
    deriv_macro[2] = Fin * Mn_in - Fout * Mn  # mn bal
    deriv_macro[3] = p.q_amm / timescale * V * VCD - Fout * Amm  # amm bal
    deriv_macro[4] = q_mab * V * VCD - Fout * mAb  # mab bal
    deriv_macro[5] = Fin * Gal_in - Fout * Gal  # gal
    deriv_macro[6:7 + fp.nos - 1] = feed_copy[5:6 + fp.nos - 1] * V * VCD - Fout * (y[6:7 + fp.nos - 1] / V)
    return deriv_macro