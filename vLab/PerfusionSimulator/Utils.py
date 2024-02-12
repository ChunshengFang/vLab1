SIA_INDEX = [17, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32]


def compute_species_distribution(os):
    """ Summary of Oligsaccharides: the distributions of glycan forms

    :param array os: oligosaccharide
    :returns HM, FA1G1, FA2G0, FA2G1, FA2G2, SIA: percentage of different glycan forms
    """
    total = sum(os[:, -1])
    HM = sum([os[i, -1] for i in (list(range(8)) + [9])]) / total * 100
    FA1G1 = os[14, -1] / total * 100
    FA2G0 = os[13, -1] / total * 100
    FA2G1 = (os[18, -1] + os[19, -1]) / total * 100
    FA2G2 = os[24, -1] / total * 100
    SIA = sum([os[i, -1] for i in SIA_INDEX]) / total * 100

    return HM, FA1G1, FA2G0, FA2G1, FA2G2, SIA
