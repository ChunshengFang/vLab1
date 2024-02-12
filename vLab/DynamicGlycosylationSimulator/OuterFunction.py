from vLab.DynamicGlycosylationSimulator.InnerFunction import innerFunction
from vLab.GlycosylationModelBase.GlycosylationModelParams import CellCultureVariables


def outerfunction(mult, p, fp, t_span, feed_cond, ic_states, ic_macro_rest):
    """ Solve the dynamic glycosylation model between two samplings

    :param array mult: updated parameters for the perfusion cell culture and the N-glycosylation model from MIT
    :param GlycosylationModelParamClass p: N-linked glycosylation model parameters
    :param GlycosylationNetwork fp: N-linked glycosylation network
    :param int t_span: sampling time [min]
    :param ndarray feed_cond: feeding conditions between two samplings
    :param array ic_states: initial state including volume, VCD*V, manganese*V, ammonia*V, mab*V, Gal*V
    :param array ic_macro_rest: 33 oligsaccharides

    :returns ndarray: updated cell culture states (i.e. volume, VCD, Mn, Amm, mAb, Galactose) and 33 oligsaccharides
    """

    p.ncyt[0] = mult[0]
    p.ncyt[2] = mult[1]
    p.ncyt[1] = mult[2]
    p.nscyt[0] = mult[3]
    p.nscyt[1] = mult[4]
    p.nscyt[3] = mult[5]
    p.kdgnt1 = mult[6]
    p.kdfuct = mult[7]
    p.kdgalt_f = mult[8]
    p.kdgalt = mult[9]
    p.kdmngalt = mult[10]
    p.mumax = mult[11]
    p.kgaludpgal = mult[12]
    p.q_amm = mult[13]
    p.mabtiter = mult[14]
    x = CellCultureVariables(0, 0, p.mudpgal / p.kudpgal + 0 / p.kgaludpgal, mult[14], p.ncyt, p.nscyt)
    return innerFunction(x, p, fp, t_span, feed_cond, ic_states, ic_macro_rest)[-1, :]
