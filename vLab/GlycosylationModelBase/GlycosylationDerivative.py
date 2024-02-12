import numpy as np

from vLab.GlycosylationModelBase.Reaction import compute_stoichiometric_coefficient_reaction


def steady_state_inner_derivative(z, y, x, p, fp, is_dynamic=False):
    """ Compute the derivative of steady state mechanistic Glycosylation model dy/dz

    :param float z: Golgi dimensionless length
    :param array y: state (Oligsaccharides, nucleotide sugars in Golgi apparatus and nucleotides in Golgi apparatus)
    :param array x: perfusion cell culture related data (ammonium, Manganese chloride and Nucleotide sugar in cytosol)
    :param GlycosylationModelParamClass p: N-linked glycosylation model parameters
    :param GlycosylationNetwork fp: N-linked glycosylation network
    :return array deriv: derivatives of N-linked glycosylation models
    """
    ft, vos, vns, vn, r = compute_stoichiometric_coefficient_reaction(z, y, x, p, fp)
    # steady state derivative w.r.t z
    deriv = np.zeros((fp.nos + fp.nns + fp.nn))
    deriv[:fp.nos] = vos @ r
    deriv[fp.nos:(fp.nos + fp.nns)] = ft + vns @ r  # golgi nucleotide sugars
    deriv[(fp.nos + fp.nns):(fp.nos + fp.nns + fp.nn)] = -np.array([ft[0] + ft[2], ft[1], ft[3]]) + vn @ r
    deriv = deriv * np.pi * (p.d ** 2) / 4 / p.q
    if is_dynamic:
        deriv *= p.l
    return deriv

