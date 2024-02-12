import pandas as pd
import random
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Network:
    """Dataa class for enzyme based Glycosylation newtork"""
    Man1: np.ndarray  #: alpha-1,2 mannosidase I
    Man2: np.ndarray  #: alpha-1,3/alpha-1,6 mannosidase II
    GnT1: np.ndarray  #: alpha-1,3 N-acetylglucosaminyl transferase I
    GnT2: np.ndarray  #: alpha-1,6 N-acetylglucosaminyl transferase II
    FucT: np.ndarray  #: alpha-1,6 fucosyltransferase
    GalT: np.ndarray  #: beta-1,4 galactosyltransferase
    SiaT: np.ndarray  #: alpha-1,6 sialyltransferase


@dataclass()
class CellCultureVariables:
    """Data (static) class for cell culture state of perfusion bioreactor"""
    amm: float  #: Ammonia concentration
    mn: float  #: Mn concentration
    udpgalcyt: float  #: UDP-Gal balance with external addition of galactosea
    mabtiter: float  #: mAb concentratio
    ncyt: np.ndarray  #: Nucleotides in cytosol
    nscyt: np.ndarray  #: Nucleotide sugars in cytosol


class GlycosylationModelParamClass:
    """ Static Class for Glycosylation parameters """

    def __init__(self, is_dynamic=False, noise=0):
        self.a = 99  #: Surface area of the Golgi  (SI table 1. um^2)
        self.v = 25  #: Volume of the Golgi apparatus (SI table 1. um^3)
        self.d = 7.82  #: Golgi internal diameter (SI table 1. um)
        self.l = 0.52  # from delVal (not used)
        self.q = 1.12  #: Flow rate through the Golgi (SI table 1 um^3/min)

        self.phopt = 6.6  #: Optimal Golgi pH (SI table 5)
        self.pka = 7.5  #: Apparent Golgi pKA (SI table 5)

        '''SI table 6'''
        self.emax = np.array(
            [2.32, 1.41, 1.14, .822, 1.83, 4.00,
             4.26]) * 1e-1  #: peak concentration of enzyme (SI Table 7. umol/L; values divided by 10 from delVal).
        self.zmaxj = np.array([2.55, 3.88, 3.63, 4.95, 5.25, 7.76,
                               7.82]) * 1e-1  #: localization of peak concentration of each enzyme (SI Table 7 ).
        self.omegaj = np.array([7.85, 5.75, 7.83, 7.81, 7.53, 4.50,
                                3.79]) * 1e-2  #: width of concentration profile of enzyme (SI Table 6)
        if is_dynamic:
            self.omegaj = self.omegaj * 2
        #  UDP - GlcNAc / UMP, GDP - Fuc / GMP UDP - Gal / UMP CMP - Neu5Ac / CMP
        self.tpmax = np.array(
            [7.60, 3.65, 6.94, 8.23]) * 1e-7  #: peak concentration of transport protein (SI Table 7. umol/dm^2)
        self.zmaxk = np.array(
            [3.69, 4.96, 7.34, 7.91]) * 1e-1  #: localization of peak concentration of transport protein (SI Table 7 )
        self.omegak = np.array(
            [6.51, 8.43, 4.26, 5.16]) * 1e-2  #: width of concentration profile of  transport protein (SI Table 7 )

        self.kfmax = np.array([888, 1924, 1022, 1406, 291, 872, 491])
        ''' maximum turnover rate for reaction (SI table 8 enzyme kinetic parameters. min^-1)'''

        self.omegaf = np.array([1.72, 1.39, 1.08, 0.96, 2.01, 0.78, 0.60])
        ''' Enzymatic activity parameters determined from literature data (SI table 10)'''

        self.kcytns = np.array([7.13, 7.5, 2.4, 1.3])
        '''Table 3 from delVal. uM'''

        self.kgolgin = np.array([0.135, 0.140, 0.124, 0.133])
        '''Table in Figure 3 from delVal. uM'''

        self.ktk = np.array([1084, 130, 689, 397])
        '''Table 3 from delVal. Transport kinetic parameters. 1/min'''

        # other constants
        self.mudpgal = 6.16e-2  #: Maintenance coefficient of UDP-Gal. (Table 3. mmol/L/day)
        self.kudpgal = 5.14e-1  #: Sugar mediated nucleotide sugar synthesis. (Table 3. 1/day)
        self.na = 1.45e1  #: Ammonia associated Golgi pH constant (villiger Table 4. mmol/L)
        self.kgaludpgal = 5.69 * 1e1  #: Equilibrium constant Gal/UDP-Gal (Table 3. mmol/L)
        '''Cell Growth'''
        self.mumax = 0.909
        self.k_amm = 1.86

        '''to change if we want to change ammonia specific productivity'''
        self.q_amm = 9.36e-2

        self.kdman1 = np.array([60.5, 110.0, 30.8, 74.1])  # M9,8,7,6
        '''SI table 9 - dissociation constants for golgi resident enzymes'''
        self.kdman2 = np.array([200, 100, 200, 100])  # M5,4,5,4
        '''SI table 9 - dissociation constants for golgi resident enzymes'''

        self.kdgnt1 = 115  # M5. and matches with villiger. 260 for del val
        '''SI table 9 - dissociation constants for golgi resident enzymes'''
        self.kdkgnt1 = 170  # checked. and matches with villiger.
        '''SI table 9 - dissociation constants for golgi resident enzymes'''

        self.kdgnt2 = 97  # checked
        '''SI table 9 - dissociation constants for golgi resident enzymes'''
        self.kdkgnt2 = 960  # checked. and matches with del val and villiger.
        '''SI table 9 - dissociation constants for golgi resident enzymes'''

        self.kdfuct = 43.4  # checked. and matches with villiger
        '''SI table 9 - dissociation constants for golgi resident enzymes'''
        self.kdkfuct = 46  # checked. and matches with del val and villiger.
        '''SI table 9 - dissociation constants for golgi resident enzymes'''

        self.kdkgalt = 65  # checked. and matches with del val and villiger.
        '''SI table 9 - dissociation constants for golgi resident enzymes'''
        self.kdgalt = 1.16e4
        '''SI table 9 - dissociation constants for golgi resident enzymes'''
        self.kdgalt_f = 1.73e3
        '''SI table 9 - dissociation constants for golgi resident enzymes'''

        self.kdksiat = 50  # checked. and matches with del val and villiger.
        '''SI table 9 - dissociation constants for golgi resident enzymes'''
        self.kdsiat = 3.81e4
        '''SI table 9 - dissociation constants for golgi resident enzymes'''

        # mn dependence
        self.kdmngnt = 5.47e-3  #: Manganese dissociation constant for GnT (Table 3. umol/L)
        self.kdmngalt = 3.82e-2  #: Manganese dissociation constant for GalT (Table 3. umol/L)

        '''Concentrations of UDP-GlcNAc, GDP-Fuc, UDP-Gal, CMP-Neu5Ac'''
        self.nscyt = np.array([1.62, 0.043, 0.1158, 0.040]) * 1e3  # uM

        '''Concentrations of UMP + UDP, GMP+GDP, CMP+CDP'''
        self.ncyt = np.array([0.490 + 1.452, 0.117 + 0.379, 0.058 + 0.190]) * 1e3  # uM

        '''titer'''
        self.mabtiter = 2 * 8.03 * 10 ** -12 / 1.12 / 60 / 24 * 10 ** 15 / 150 / 10 ** 3 * 10 ** 6  # umol/L

        if noise > 0:
            self.a = random.normalvariate(self.a, self.a * noise)
            self.v = random.normalvariate(self.v, self.v * noise)
            self.d = random.normalvariate(self.d, self.d * noise)
            self.l = random.normalvariate(self.l, self.l * noise)
            self.q = random.normalvariate(self.q, self.q * noise)
            self.phopt = random.normalvariate(self.phopt, self.phopt * noise)
            self.pka = random.normalvariate(self.pka, self.pka * noise)
            emax = np.random.multivariate_normal(self.emax, np.power(np.diag(self.emax) * noise, 2))
            if np.sum(emax < 0) == 0:
                self.emax = emax
            zmaxj = np.random.multivariate_normal(self.zmaxj, np.power(np.diag(self.zmaxj) * noise, 2))
            if np.sum(zmaxj < 0) == 0:
                self.zmaxj = zmaxj
            omegaj = np.random.multivariate_normal(self.omegaj, np.power(np.diag(self.omegaj) * noise, 2))
            if np.sum(omegaj < 0) == 0:
                self.omegaj = omegaj
            tpmax = np.random.multivariate_normal(self.tpmax, np.power(np.diag(self.tpmax) * noise, 2))
            if np.sum(tpmax < 0) == 0:
                self.tpmax = tpmax
            self.zmaxk = np.random.multivariate_normal(self.zmaxk, np.power(np.diag(self.zmaxk) * noise, 2))
            self.omegak = np.random.multivariate_normal(self.omegak, np.power(np.diag(self.omegak) * noise, 2))
            self.kfmax = np.random.multivariate_normal(self.kfmax, np.power(np.diag(self.kfmax) * noise, 2))
            self.omegaf = np.random.multivariate_normal(self.omegaf, np.power(np.diag(self.omegaf) * noise, 2))
            self.kcytns = np.random.multivariate_normal(self.kcytns, np.power(np.diag(self.kcytns) * noise, 2))
            kgolgin = np.random.multivariate_normal(self.kgolgin, np.power(np.diag(self.kgolgin) * noise, 2))
            if np.sum(kgolgin < 0) == 0:
                self.kgolgin = kgolgin
            self.ktk = np.random.multivariate_normal(self.ktk, np.power(np.diag(self.ktk) * noise, 2))
            mudpgal = random.normalvariate(self.mudpgal, self.mudpgal * noise)
            if np.sum(mudpgal < 0) == 0:
                self.mudpgal = mudpgal
            kudpgal = random.normalvariate(self.kudpgal, self.kudpgal * noise)
            if np.sum(kudpgal < 0) == 0:
                self.kudpgal = kudpgal
            self.na = random.normalvariate(self.na, self.na * noise)
            self.kgaludpgal = random.normalvariate(self.kgaludpgal, self.kgaludpgal * noise)

            self.mumax = random.normalvariate(self.mumax, self.mumax * noise)
            self.k_amm = random.normalvariate(self.k_amm, self.k_amm * noise)
            q_amm = random.normalvariate(self.q_amm, self.q_amm * noise)
            if np.sum(q_amm < 0) == 0:
                self.q_amm = q_amm
            self.kdman1 = np.random.multivariate_normal(self.kdman1, np.power(np.diag(self.kdman1) * noise, 2))
            self.kdman2 = np.random.multivariate_normal(self.kdman2, np.power(np.diag(self.kdman2) * noise, 2))
            self.kdgnt1 = random.normalvariate(self.kdgnt1, self.kdgnt1 * noise)
            self.kdkgnt1 = random.normalvariate(self.kdkgnt1, self.kdkgnt1 * noise)
            self.kdgnt2 = random.normalvariate(self.kdgnt2, self.kdgnt2 * noise)
            self.kdkgnt2 = random.normalvariate(self.kdkgnt2, self.kdkgnt2 * noise)
            self.kdfuct = random.normalvariate(self.kdfuct, self.kdfuct * noise)
            self.kdkfuct = random.normalvariate(self.kdkfuct, self.kdkfuct * noise)

            self.kdkgalt = random.normalvariate(self.kdkgalt, self.kdkgalt * noise)
            self.kdgalt = random.normalvariate(self.kdgalt, self.kdgalt * noise)
            self.kdgalt_f = random.normalvariate(self.kdgalt_f, self.kdgalt_f * noise)
            self.kdksiat = random.normalvariate(self.kdksiat, self.kdksiat * noise)
            self.kdsiat = random.normalvariate(self.kdsiat, self.kdsiat * noise)
            kdmngnt = random.normalvariate(self.kdmngnt, self.kdmngnt * noise)
            if np.sum(kdmngnt < 0) == 0:
                self.kdmngnt = kdmngnt
            kdmngalt = random.normalvariate(self.kdmngalt, self.kdmngalt * noise)
            if np.sum(kdmngalt < 0) == 0:
                self.kdmngalt = kdmngalt
            self.nscyt = np.random.multivariate_normal(self.nscyt, np.power(np.diag(self.nscyt) * noise, 2))
            self.ncyt = np.random.multivariate_normal(self.ncyt, np.power(np.diag(self.ncyt) * noise, 2))
            self.mabtiter = random.normalvariate(self.mabtiter, self.mabtiter * noise)


label_matrix_ = np.array([['UMP_UDP_Conc', 1942, 0.1000],
                          ['CMP_CDP_Conc', 248, 0.1000],
                          ['GMP_GDP_Conc', 496, 0.1000],
                          ['UDP_Glc_Conc', 1620, 0.1000],
                          ['GDP_Fuc_Conc', 43, 0.1000],
                          ['CMP_Neu_Conc', 40, 0.1000],
                          ['GnTI_Kd', 96, 0.1840],
                          ['FucT_Kd', 78.6000, 0.1610],
                          ['GalT_FA2G0_Kd', 1500, 0.0610],
                          ['GalT_FA2G1_Kd', 4400, 0.0240],
                          ['GalT_K_Mn', 0.1181, 0.0820],
                          ['Mn_Conc', 0.0100, 0.1000],
                          ['Gal_Conc', 0, 0.1000],
                          ['Amm_Conc ', 1.5000, 0.1000],
                          ['qmab', 66.4000, 0.1000]])

label_matrix = np.array([['UMP_UDP_Conc', 1942, 0.1000],
                         ['CMP_CDP_Conc', 248, 0.1000],
                         ['GMP_GDP_Conc', 496, 0.1000],
                         ['UDP_Glc_Conc', 1620, 0.1000],
                         ['GDP_Fuc_Conc', 43, 0.1000],
                         ['CMP_Neu_Conc', 40, 0.1000],
                         ['GnTI_Kd', 96, 0.1840],
                         ['FucT_Kd', 78.6000, 0.1610],
                         ['GalT_FA2G0_Kd', 1500, 0.0610],
                         ['GalT_FA2G1_Kd', 4400, 0.0240],
                         ['GalT_K_Mn', 0.1181, 0.0820],
                         ['mu_max', 0.9090, 0.1000],
                         ['kgal_udpgal', 56.9000, 0.1000],
                         ['Amm_prod', 0.0936, 0.1000],
                         ['qmab', 66.4000, 0.1000]
                         ])

default_initial_states = [1.80000000000000,
                          63.6663908285356,
                          0.161999999999992,
                          4.76727639704635,
                          0.408987494319253,
                          8.99999999999949,
                          3.59297567587514e-09,
                          0.000218577287977678,
                          8.50037601725363e-05,
                          0.000621135859993949,
                          0.00869962639240848,
                          0.00985212908037032,
                          0.00499724032254945,
                          0.0886110715709328,
                          0.000628533413373264,
                          0.0749560938463601,
                          0.0131822876661247,
                          0.275027412331113,
                          0.00741195243683822,
                          8.78026225919547,
                          0.0527983257868376,
                          0.296699346037757,
                          0.296699346037757,
                          0.000132939737207326,
                          5.92149328027054,
                          5.92149328027054,
                          0.000819874801904915,
                          0.149964716124047,
                          0.00403812322356789,
                          0.00403812322356785,
                          2.45770781499769,
                          0.0676981380786785,
                          0.0676981380786783,
                          0.00122047961684700,
                          0.00122047961684701,
                          0.0180467299836550,
                          0.0180467299836551,
                          1.44854746452927e-05,
                          0.000196725070420910]
