import numpy as np
from scipy.integrate import solve_ivp

from vLab.DynamicGlycosylationSimulator.CellCultureDerivative import cell_culture_derivative
from vLab.GlycosylationModelBase.GlycosylationDerivative import steady_state_inner_derivative


def innerFunction(x, p, fp, t_span, feed_cond, ic_states, ic_macro_rest):
    """ Solve the dynamic glycosylation model between two samplings

    :param CellCultureVariables x: perfusion cell culture related data (ammonium, Manganese chloride and Nucleotide sugar in cytosol)
    :param GlycosylationModelParamClass p: N-linked glycosylation model parameters
    :param GlycosylationNetwork fp: N-linked glycosylation network
    :param int t_span: sampling time [min]
    :param ndarray feed_cond: feeding conditions between two samplings
    :param array ic_states: initial state including volume, VCD*V, manganese*V, ammonia*V, mab*V, Gal*V
    :param array ic_macro_rest: 33 oligsaccharides

    :returns ndarray: updated cell culture states [volume, VCD, Mn, Amm, mAb, Galactose] and 33 oligsaccharides
    """
    ic_macro = np.zeros((len(ic_states) + fp.nos))
    ic_macro[:len(ic_states)] = ic_states
    ic_macro[len(ic_states):(len(ic_states) + fp.nos)] = ic_macro_rest
    # initialized ic_macro with ic_macro_rest to include initial conditions
    # for the remaining 33 elements of the output
    yout_macro = ic_macro.reshape((1, len(ic_macro)))
    # yout_macro(5) = 0; % 12/23/2020 Anastasia: comment this out to allow total
    # productivity to accumulate during the MPC
    # ic_macro(5)= 0; % 12/23/2020 Anastasia: comment this out to allow total
    # productivity to accumulate during the MPC
    tplot = [0]
    num_macro_states = len(ic_states)
    convert_2_hour = 60

    ## initial conditions for golgi
    ic_golgi = np.zeros((fp.nos + fp.nns + fp.nn))
    ic_golgi[0] = x.mabtiter  # umol/L
    ic_golgi[fp.nos:fp.nos + fp.nns] = x.nscyt * 40  # nucleotide sugar concentrations in umol/L. third entry is mystery
    ic_golgi[fp.nos + 2] = (ic_macro[5] * p.kudpgal / p.kgaludpgal + p.mudpgal) / p.kudpgal * 1e3 * 40  # updating with correct UDP-Gal concentration
    ic_golgi[fp.nos + fp.nns:fp.nos + fp.nns + fp.nn] = x.ncyt  # sum of nucleotide concentrations in umol/L

    offset = 0
    feed = np.zeros(num_macro_states + fp.nos - 1)
    for j in range(len(feed_cond[:, 0]) - 1):
        feed[0] = feed_cond[j, 1] * ic_macro[0]  # perf rate * volume
        feed[1] = feed[0] * feed_cond[j, 2]  # vol in * bleed ratio
        feed[2] = feed[0] * (1 - feed_cond[j, 2])  # vol in * (1-bleed ratio)
        feed[3] = feed_cond[j, 3]  # mn
        feed[4] = feed_cond[j, 4]  # galactose
        feed[5] = ic_macro[0]  # volume
        # change voluem as needed

        for k in range(int((feed_cond[j + 1, 0] - feed_cond[j, 0]) / t_span)):
            # get feed states
            x.mn = yout_macro[max(0, yout_macro.shape[0] - 1 - offset), 2] / yout_macro[
                max(0, yout_macro.shape[0] - 1 - offset), 0]  # total manganese over total volume
            x.amm = yout_macro[max(0, yout_macro.shape[0] - 1 - offset), 3] / yout_macro[
                max(0, yout_macro.shape[0] - 1 - offset), 0]  # shouldnt be feed, should be in BR
            x.udpgalcyt = (yout_macro[max(0, yout_macro.shape[
                0] - 1 - offset), 5] * p.kudpgal / p.kgaludpgal + p.mudpgal) / p.kudpgal  # shouldnt be feed should be BR
            x.nscyt[2] = x.udpgalcyt * 1e3
            ic_golgi[fp.nos + 2] = x.udpgalcyt * 1e3 * 40
            # yout_golgi = odeint(steady_state_inner_derivative, ic_golgi, t, args=(x, p, fp,), rtol=1e-6, atol=1e-7)
            yout_golgi = solve_ivp(steady_state_inner_derivative, [0, 1], ic_golgi, args=(x, p, fp, True), method='RK45',
                             rtol=1e-6, atol=1e-9)
            # [tout,yout_golgi]=ode15s(@(z,y) innerderiv_golgi(z,y,x,p,fp),[0 1],ic_golgi,options)
            feed[(num_macro_states-1):(num_macro_states + fp.nos - 1)] = yout_golgi.y.transpose()[-1, :fp.nos]
            yout = solve_ivp(cell_culture_derivative, [tplot[-1], t_span + tplot[-1]], ic_macro, args=(p, fp, feed),
                             method='RK45', rtol=1e-6, atol=1e-9)
            yout_macro = np.vstack([yout_macro, yout.y.transpose()[-1, :]])
            tplot.append(yout.t[-1])
            ic_macro = yout.y[-1, :]

    return yout_macro
