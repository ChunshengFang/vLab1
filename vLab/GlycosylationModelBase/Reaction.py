import numpy as np


def any_column(data):
    any_by_col = np.sum(data > 0, axis=0) > 0
    return np.array(any_by_col).flatten()


def compute_stoichiometric_coefficient_reaction(z, y, x, p, fp):
    """ Compute stoichiometric coefficient and reaction rates of N-linked glycosylation models. It includes Equation
    19-29 from Villiger et. al.

    **Reference**: Villiger, T. K., Scibona, E., Stettler, M., Broly, H., Morbidelli, M., & Soos, M. (2016). Controlling
    the time evolution of mAb N‐linked glycosylation‐Part II: Model‐based predictions. Biotechnology progress, 32(5),
    1135-1148.

    :param array y: state (Oligsaccharides, nucleotide sugars in Golgi apparatus and nucleotides in Golgi apparatus)
    :param array z: Golgi dimensionless length
    :param array x: cell culture related data (ammonium, Manganese chloride and Nucleotide sugar in cytosol)
    :param GlycosylationModelParamClass p: N-linked glycosylation model parameters
    :param GlycosylationNetwork fp: N-linked glycosylation network
    :return list: flowrate of Nucleotide sugars into the Golgi, stoichiometric coefficient and reaction rates of N-linked glycosylation models
    """
    os = y[:fp.nos]  # oligosaccharides
    ns = y[fp.nos:(fp.nos + fp.nns)]  # nucleotide sugars
    n = y[(fp.nos + fp.nns):(fp.nos + fp.nns + fp.nn)]  # nucleotides

    # Cell internal calculation
    ph = p.pka + np.log10(x.amm / (p.na - x.amm))  # equation 15 from karst supplement
    kf = p.kfmax * np.exp(
        -0.5 * ((ph - p.phopt) / p.omegaf) ** 2)  # equation 19 from karst supplement; units are per minute

    e = p.emax * np.exp(-0.5 * ((z - p.zmaxj) / p.omegaj) ** 2)  # equation 22 from karst supplement; units are umol/L
    tp = p.tpmax * np.exp(
        -0.5 * ((z - p.zmaxk) / p.omegak) ** 2)  # equation 21 from karst supplement; units are umol / dm ^ 2

    man1sum = sum(os[any_column(fp.network.Man1)] / p.kdman1)
    man2sum = sum(os[any_column(fp.network.Man2)] / p.kdman2)
    gnt1sum = sum(os[any_column(fp.network.GnT1)] / p.kdgnt1)
    gnt2sum = sum(os[any_column(fp.network.GnT2)] / p.kdgnt2)

    fuctsum = sum(os[any_column(fp.network.FucT)] / p.kdfuct)
    galtsum = os[any_column(fp.network.GalT)]
    galtsum = sum(galtsum[2:4]) / p.kdgalt_f + (sum(galtsum[:2]) + sum(galtsum[4:])) / p.kdgalt

    siatsum = sum(os[any_column(fp.network.SiaT)] / p.kdsiat)

    vos = np.zeros((fp.nos, fp.nr))  # number of species by number of reactions
    vns = np.zeros((fp.nns, fp.nr))  # number of nucelotide sugars by number of reactions
    vn = np.zeros((fp.nn, fp.nr))  # number of nucleotides by number of reactions

    n_list = np.array([n[k] for k in [0, 1, 0, 2]])
    denominator_ft = n_list / (p.kgolgin + n_list)
    ft = p.a / p.v * 1e5 * p.ktk * tp * (
            x.nscyt / (p.kcytns + x.nscyt)) * denominator_ft  # um^2 / um^3 * um/dm * 1/min * umol/dm^2 = umol/L/min

    # Reactions
    r = np.zeros(fp.nr)
    for i in range(fp.nr):
        enzyme_idx = fp.networkidx[i, 2]
        if enzyme_idx == 0:
            man1sum2 = os[fp.networkidx[i, 1]] / p.kdman1[i]
            if fp.networkidx[i, 1] < 3:
                man1sum2 = man1sum2 + os[fp.networkidx[i, 0]] / p.kdman1[i + 1]
            r[i] = fp.eqkmm(kf[0],
                            e[0],
                            os[fp.networkidx[i, 1]],
                            p.kdman1[i],
                            man1sum - os[fp.networkidx[i, 1]] / p.kdman1[i],
                            man1sum2)

        if enzyme_idx == 1:
            man2sum2 = os[fp.networkidx[i, 1]] / p.kdman2[i - 4]
            if i == 4 or i == 6:
                man2sum2 = man2sum2 + os[fp.networkidx[i, 0]] / p.kdman1[i - 3]

            r[i] = fp.eqkmm(kf[1],
                            e[1],
                            os[fp.networkidx[i, 1]],
                            p.kdman2[i - 4],
                            man2sum - os[fp.networkidx[i, 1]] / p.kdman2[i - 4],
                            man2sum2)

        if enzyme_idx == 2:  # gnt1
            r[i] = fp.eqksobb(kf[2],
                              e[2],
                              x.mn,
                              ns[0],
                              os[fp.networkidx[i, 1]],
                              p.kdmngnt,
                              p.kdkgnt1,
                              p.kdgnt1,
                              gnt1sum - os[fp.networkidx[i, 1]] / p.kdgnt1,
                              os[fp.networkidx[i, 0]],
                              1e6,
                              n[0],
                              p.kdkgnt1)
            vns[0, i] = -1
            vn[0, i] = 1
        if enzyme_idx == 3:  # gnt2
            r[i] = fp.eqksobb(kf[3],
                              e[3],
                              x.mn,
                              ns[0],
                              os[fp.networkidx[i, 1]],
                              p.kdmngnt,
                              p.kdkgnt2,
                              p.kdgnt2,
                              gnt2sum - os[fp.networkidx[i, 1]] / p.kdgnt2,
                              os[fp.networkidx[i, 0]],
                              1e6,
                              n[0],
                              p.kdkgnt1)
            vns[0, i] = -1
            vn[0, i] = 1
        if enzyme_idx == 4:  # fuct
            r[i] = fp.eqkrobb(kf[4],
                              e[4],
                              ns[1],
                              os[fp.networkidx[i, 1]],
                              p.kdkfuct,
                              p.kdfuct,
                              fuctsum - os[fp.networkidx[i, 1]] / p.kdfuct,
                              os[fp.networkidx[i, 0]],
                              1e6,
                              n[1],
                              p.kdkfuct)
            vns[1, i] = -1
            vn[1, i] = 1
        if enzyme_idx == 5:  # galt
            if 17 <= i <= 20:
                r[i] = fp.eqksobb(kf[5],
                                  e[5],
                                  x.mn,
                                  ns[2],
                                  os[fp.networkidx[i, 1]],
                                  p.kdmngalt,
                                  p.kdkgalt,
                                  p.kdgalt_f,
                                  galtsum - os[fp.networkidx[i, 1]] / p.kdgalt_f,
                                  os[fp.networkidx[i, 0]],
                                  p.kdgalt,
                                  n[0],
                                  p.kdkgalt)
                r[i] = r[i] / 2  # dividing this because these reactions have same substrate
            else:
                r[i] = fp.eqksobb(kf[5],
                                  e[5],
                                  x.mn,
                                  ns[2],
                                  os[fp.networkidx[i, 1]],
                                  p.kdmngalt,
                                  p.kdkgalt,
                                  p.kdgalt,
                                  galtsum - os[fp.networkidx[i, 1]] / p.kdgalt,
                                  os[fp.networkidx[i, 0]],
                                  1e6,
                                  n[0],
                                  p.kdkgalt)

            if 15 <= i <= 16:
                r[i] = fp.eqksobb(kf[5],
                                  e[5],
                                  x.mn,
                                  ns[2],
                                  os[fp.networkidx[i, 1]],
                                  p.kdmngalt,
                                  p.kdkgalt,
                                  p.kdgalt_f,
                                  galtsum - os[fp.networkidx[i, 1]] / p.kdgalt_f,
                                  os[fp.networkidx[i, 0]],
                                  1e6,
                                  n[0],
                                  p.kdkgalt)
            vns[2, i] = -1
            vn[0, i] = 1
        if enzyme_idx == 6:  # siat
            if 35 <= i <= 38:
                r[i] = fp.eqkrobb(kf[6],
                                  e[6],
                                  ns[3],
                                  os[fp.networkidx[i, 1]],
                                  p.kdksiat,
                                  p.kdsiat,
                                  siatsum - os[fp.networkidx[i, 1]] / p.kdsiat,
                                  os[fp.networkidx[i, 0]],
                                  p.kdsiat,
                                  n[2],
                                  p.kdksiat)
                r[i] = r[i] / 2
            else:
                r[i] = fp.eqkrobb(kf[6],
                                  e[6],
                                  ns[3],
                                  os[fp.networkidx[i, 1]],
                                  p.kdksiat,
                                  p.kdsiat,
                                  siatsum - os[fp.networkidx[i, 1]] / p.kdsiat,
                                  os[fp.networkidx[i, 0]],
                                  1e6,
                                  n[2],
                                  p.kdksiat)
            vns[3, i] = -1
            vn[2, i] = 1

        vos[fp.networkidx[i, 1], i] = -1
        vos[fp.networkidx[i, 0], i] = 1
    return ft, vos, vns, vn, r
