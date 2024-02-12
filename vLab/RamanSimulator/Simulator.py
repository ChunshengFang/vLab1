import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


class Raman_Simulator:
    """
    Performs Raman Spectra data simulation with background error and measurement error based on concentrations of molecules.
    """
    def __init__(self):
        self.concen_base = pd.Series(np.array([6.190, 0.000, 6.120, 0.200, 0.341 ]),index=['Glucose', 'Lactate', 'Glutamine', 'Ammonia', 'Glutamate'])
        self.Raman_base = pd.read_csv('data/RamanBaseline', index_col=(0), header=None).squeeze()
        self.Raman_mean = pd.read_csv('data/mean.txt', header=None,sep=" ").squeeze()
        self.Raman_cov = pd.read_csv('data/covar.txt', header=None,sep=" ")


    def simulate(self, Glc, Lac, Gln, NH3, Glu = -100.0):

        if Glu > 0:
            concen = pd.Series(np.array([Glc, Lac, Gln, NH3, Glu ]),index=['Glucose', 'Lactate', 'Glutamine', 'Ammonia', 'Glutamate'])
            reg_model = pickle.load(open('vLab/RamanSimulator/simulator_wi_Glu.sav', 'rb'))
            predicted_diff = pd.DataFrame(reg_model.predict((np.array(concen)-np.array(self.concen_base)).reshape(1, -1)))
            predicted_diff.columns = self.Raman_base.index
            predicted = predicted_diff +  self.Raman_base

        else:
            concen = pd.Series(np.array([Glc, Lac, Gln, NH3 ]),index=['Glucose', 'Lactate', 'Glutamine', 'Ammonia'])
            reg_model = pickle.load(open('vLab/RamanSimulator/simulator_wo_Glu.sav', 'rb'))
            predicted_diff = pd.DataFrame(reg_model.predict((np.array(concen)-np.array(self.concen_base)[0:4]).reshape(1, -1)))
            predicted_diff.columns = self.Raman_base.index
            predicted = predicted_diff +  self.Raman_base
            
        predicted = predicted + np.random.multivariate_normal(self.Raman_mean,self.Raman_cov)
        plt.figure()
        np.transpose(predicted)[0:1401].plot(legend=False, title = "Simulated Raman Spectra Data")
        plt.xlabel('RamanShift(cm-1)')
        plt.ylabel('Intensity')
        plt.show()
          
        simulated_Raman = predicted.squeeze()[0:1401]
        return simulated_Raman






