import numpy as np
import pandas as pd
from vLab.GlycosylationModelBase.GlycosylationModelParams import Network
from vLab.PerfusionSimulator.Utils import SIA_INDEX


class GlycosylationNetwork:
    """ Class for Glycosylation network. This class will build the network from a network description file

        :param int nn: number of nucleotides in the Golgi apparatus
        :param int nns: number of nucelotide sugars in the Golgi apparatus
        :param int nos: number of oligsaccharides
        :param str network_data_path: data path for glycosylation network
    """
    def __init__(self, nn=3, nns=4, nos=33, network_data_path='data/Network Description.csv'):
        """ Construct the Glycosylation network. """
        self.nn = nn  # number of species
        self.nns = nns  # number of nucelotide sugars
        self.nos = nos  # number of golgi species
        self.elist = ['Man1', 'Man2', 'GnT1', 'GnT2', 'FucT', 'GalT', 'SiaT']  # List the enzymes
        self._build_network(network_data_path)

    def _build_network(self, network_data_path):
        """Build the network from the provided data"""
        data = pd.read_csv(network_data_path, index_col=0, skiprows=0, na_values=0, dtype=str)
        data.fillna(0, inplace=True)
        data = data.to_numpy()

        network_for_individual_enzyme = [(enzyme, (data == enzyme).astype(int)) for enzyme in self.elist]
        network_for_individual_enzyme = dict(network_for_individual_enzyme)

        ''' build the network'''
        self.network = Network(network_for_individual_enzyme['Man1'],
                               network_for_individual_enzyme['Man2'],
                               network_for_individual_enzyme['GnT1'],
                               network_for_individual_enzyme['GnT2'],
                               network_for_individual_enzyme['FucT'],
                               network_for_individual_enzyme['GalT'],
                               network_for_individual_enzyme['SiaT'])
        enzyme_idx = zip(range(len(self.elist)), self.elist)

        '''build the network index ndarray'''
        self.networkidx = []
        for i, enzyme in enzyme_idx:
            rows, cols = np.nonzero(network_for_individual_enzyme[enzyme])
            self.networkidx = self.networkidx + [[row, col, i] for row, col in zip(rows, cols)]
        self.nr = len(self.networkidx)
        self.networkidx = np.array(self.networkidx)

        '''build the reaction labels'''
        self.reaction_label = ['R{0} OS{1} OS{2} {3}'.format(str(i+1), self.networkidx[i, 1], self.networkidx[i, 0],
                                                             self.networkidx[i, 2]) for i in range(self.nr)]

        '''create the labels for species'''
        self.labels = ['OS{}'.format(str(i+1)) for i in range(self.nos)]

        # nucleotidesugars
        self.labels.append('UDP-GlcNAc')
        self.labels.append('GDP-Fuc')
        self.labels.append('UDP-Gal')
        self.labels.append('CMP-Neu5Ac')

        # nucleotides
        self.labels.append('UDP')
        self.labels.append('GDP')
        self.labels.append('CMP')

        # add labels for reactions
        self.labels = self.labels + self.reaction_label
        # add labels for enzymes
        self.labels = self.labels + self.elist

        # add labels for transport proteins between nucleotide sugars and nucleotides
        self.labels = self.labels + ['TP GDP-GlcNAc/UMP', 'TP GDP-Fuc/GMP', 'TP UDP-Gal/UMP', 'TP CMP-Neu5Ac/CMP']

        # Labeled as high mannose structures
        self.labels[9] = self.labels[9] + ' HM'
        for i in range(8):
            self.labels[i] = self.labels[i] + ' HM'

        # label structure 13 as fa2g0
        self.labels[13] = self.labels[13] + ' FA2G0'

        # label structure 14 and 15 as fa2g1
        self.labels[14] = self.labels[14] + ' FA2G1'
        self.labels[15] = self.labels[15] + ' FA2G1'

        # label structure 24 as FA2G2
        self.labels[24] = self.labels[24] + ' FA2G2'

        # label structure 17,20,22,23,25:33 as SIA
        for i in SIA_INDEX:
            self.labels[i] = self.labels[i] + ' SIA'

    def eqkmm(self, kfj, ej, osi, kdi, sumterm1, sumterm2):
        """Michealis–Menten kinetics (ManI and ManII) Eq. 25 of Villiger et. al."""
        return kfj * ej * osi / kdi / (1 + sumterm1 + sumterm2)

    def eqksobb(self, kfj, ej, mn, nsgolgik, osi, kdmn, kdk, kdi, sumterm, osi1, kdi1, ngolgik, kdnk):
        """Sequential-order bi-bi kinetcis with Mn as cofactor (GnTI, GnTII, and GalT). Eq. 26 of Villiger et. al. """
        return kfj * ej * mn * nsgolgik * osi / kdmn / kdk / kdi / \
               (1 + mn / kdmn + mn / kdmn * nsgolgik / kdk + mn / kdmn * nsgolgik / kdk * osi / kdi +
                mn / kdmn * nsgolgik / kdk * sumterm + osi1 / kdi1 * ngolgik / kdnk + ngolgik / kdnk)

    def eqkrobb(self, kfj, ej, nsgolgik, osi, kdk, kdi, sumterm, osi1, kdi1, ngolgik, kdnk):
        """Random-order bi-bi kinetics (FucT and SiaT). Eq. 27 of Villiger et. al. """
        return kfj * ej * nsgolgik * osi / kdk / kdi / (1 + nsgolgik / kdk + osi / kdi +
                                                        sumterm + nsgolgik / kdk * osi / kdi + nsgolgik / kdk * sumterm +
                                                        ngolgik / kdnk * osi1 / kdi1 + ngolgik / kdnk + osi1 / kdi1)

