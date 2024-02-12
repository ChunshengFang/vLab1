import numpy as np

SIA_INDEX = [17, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32]


def compute_species_distribution(os):
    """ Summary of Oligsaccharides during the experiment: the distributions of glycan forms over time

    :param ndarray os: oligosaccharide
    :returns list: HM, FA1G1, FA2G0, FA2G1, FA2G2, SIA: percentage of different glycan forms
    """
    total = np.sum(os, axis=0)
    HM = os[np.array(list(range(8)) + [9]),].sum(axis=0) / total * 100
    FA1G1 = os[14,] / total * 100
    FA2G0 = os[13,] / total * 100
    FA2G1 = (os[18,] + os[19,]) / total * 100
    FA2G2 = os[24,] / total * 100
    SIA = os[np.array(SIA_INDEX),].sum(axis=0) / total * 100
    return HM, FA1G1, FA2G0, FA2G1, FA2G2, SIA
