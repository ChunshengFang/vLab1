import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from scipy.optimize import minimize

from vLab import bioreactor_control


class MPC:
    """ Class for model predictive control (MPC). Model predictive control (MPC) is a control scheme where a model is used for predicting the future behavior of
        the system over finite time window, the horizon. Based on these predictions and the current measured/estimated
        state of the system, the optimal control inputs with respect to a defined control objective and subject to system
        constraints is computed. After a certain time interval, the measurement, estimation and computation process is
        repeated with a shifted horizon.

        :param int P: Prediction Horizon
        :param int M: Control Horizon
    """
    def __init__(self, P=10, M=3):
        self.P = P  # Prediction Horizon
        self.M = M  # Control Horizon
        self.maxmove = 0.1

    def objective(self, u_hat, cur_index, sp, u, y0, yp_hat, delta_t_hat):
        """ Objective function for MPC
        """
        # Declare the variables in fuctions
        se = np.zeros(2 * self.P + 1)
        sp_hat = np.zeros(2 * self.P + 1)

        # Prediction
        for k in range(1, 2 * self.P + 1):
            if k == 1:
                y_hat0 = y0

            if k <= self.P:
                if cur_index - self.P + k < 0:
                    u_hat[k] = 0

                else:
                    u_hat[k] = u[cur_index - self.P + k]

            elif k > self.P + self.M:
                u_hat[k] = u_hat[self.P + self.M]

            ts_hat = [delta_t_hat * (k - 1), delta_t_hat * (k)]

            y_hat = odeint(bioreactor_control, y_hat0, ts_hat, args=(u_hat[k],))
            y_hat0 = y_hat[-1, :].copy()
            yp_hat[k] = y_hat[-1, 1]

            # Squared Error calculation
            sp_hat[k] = sp[cur_index]
            delta_u_hat = np.zeros(2 * self.P + 1)

            if k > self.P:
                delta_u_hat[k] = u_hat[k] - u_hat[k - 1]
                se[k] = (sp_hat[k] - yp_hat[k]) ** 2  # + 0.01 * (delta_u_hat[k])**2

        # Sum of Squared Error calculation
        obj = np.sum(se[self.P + 1:])
        return obj

    def control(self, set_point, x0, total_time=48, sample_time=1, lower_limit=0, upper_limit=0.2):
        """ Compute MPC control

        :param float set_point: setpoint for glucose concentration
        :param array x0: initial state
        :param int total_time: operation time (hr)
        :param int sample_time: sample/measurement time
        :param float lower_limit: lower controller output limits
        :param float upper_limit: upper controller output limits

        :return time, controls, predicted states, predicted controls, process states
        """
        t = np.linspace(0, total_time, int(total_time / sample_time) + 1)
        ns = len(t)
        u = np.zeros(ns + 1)

        # setpoint
        sp = np.ones(ns + 1 + 2 * self.P) * set_point
        delta_t = t[1] - t[0]
        yp_hats = []
        u_hats = []

        Xv_ss, Glc_ss, Gln_ss, Lac_ss, NH4_ss, P1_ss, P2_ss, P3_ss, V_ss = x0
        Xv_s, Glc_s, Gln_s, Lac_s, NH4_s, P1_s, P2_s, P3_s, V_s = [Xv_ss], [Glc_ss], [Gln_ss], [Lac_ss], [NH4_ss], [
            P1_ss], [P2_ss], [P3_ss], [V_ss]

        for i in range(1, ns + 1):
            # if i == 1:
            #     yp[i] = x0[1]
            ts = [delta_t * (i - 1), delta_t * i]
            y = odeint(bioreactor_control, x0, ts, args=(u[i],))
            x0 = y[-1, :].copy()
            # yp[i] = y[-1, 1]

            Xv_s.append(y[-1, 0])
            Glc_s.append(y[-1, 1])
            Gln_s.append(y[-1, 2])
            Lac_s.append(y[-1, 3])
            NH4_s.append(y[-1, 4])
            P1_s.append(y[-1, 5])
            P2_s.append(y[-1, 6])
            P3_s.append(y[-1, 7])
            V_s.append(y[-1, 8])

            u_hat0 = np.ones(2 * self.P + 1) * 0.01
            yp_hat = np.zeros(2 * self.P + 1)
            t_hat = np.linspace(i - self.P, i + self.P, 2 * self.P + 1)
            delta_t_hat = t_hat[1] - t_hat[0]
            # initial guesses
            for k in range(1, 2 * self.P + 1):

                if k <= self.P:
                    if i - self.P + k < 0:
                        u_hat0[k] = 0

                    else:
                        u_hat0[k] = u[i - self.P + k]

                elif k > self.P:
                    u_hat0[k] = u[i]

            # show initial objective
            print('Initial SSE Objective: ' + str(self.objective(u_hat0, i, sp, u, x0, yp_hat, delta_t_hat)))

            # MPC calculation
            start = time.time()

            solution = minimize(self.objective, u_hat0, args=(i, sp, u, x0, yp_hat, delta_t_hat), options={'disp': False})  # , method='SLSQP'
            u_hat = solution.x

            end = time.time()
            elapsed = end - start

            print('Final SSE Objective: ' + str(self.objective(u_hat, i, sp, u, x0, yp_hat, delta_t_hat)))
            # print('Elapsed time: ' + str(elapsed) )
            yp_hats.append(yp_hat)
            u_hats.append(u_hat)
            delta = np.diff(u_hat)

            if i < ns:
                if np.abs(delta[self.P]) >= self.maxmove:
                    if delta[self.P] > 0:
                        u[i + 1] = u[i] + self.maxmove
                    else:
                        u[i + 1] = u[i] - self.maxmove

                else:
                    u[i + 1] = u[i] + delta[self.P]
                u[i + 1] = max(min(u[i + 1], upper_limit), lower_limit)

        return t, u[1:-1], np.array(yp_hats[1:]), np.array(u_hats[1:]), Xv_s[:-1], Glc_s[:-1], Gln_s[:-1], Lac_s[:-1], NH4_s[:-1], P1_s[:-1], P2_s[:-1], P3_s[:-1], V_s[:-1]

if __name__ == '__main__':
    x0 = [100, 40, 50, 0., 0., 0, 0.e+00, 0, 5.e-01]
    setpoint = 40  # 0 - 80
    total_time = 48  # 24 - 72
    sample_time = 1 # 0.1-10
    mpc = MPC(P=10, M=3)
    t, u, yp_hats, u_hats, Xv_s, Glc_s, Gln_s, Lac_s, NH4_s, P1_s, P2_s, P3_s, V_s = \
        mpc.control(setpoint, x0, total_time, sample_time, lower_limit=0, upper_limit=0.2)
    ns = len(t)
   # plotting for forced prediction
    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.subplot(2,1,1)
    plt.plot(t[:-1], np.ones(len(t[:-1])) * setpoint, 'r-',linewidth=2,label='Setpoint')
    # plt.plot(t_hat[P:],sp_hat[P:],'r--',linewidth=2)
    plt.plot(t[:-1], Glc_s[1:], 'k-', linewidth=2, label='Measured GLC')
    # plt.plot(t_hat[P:],yp_hat[P:],'k--',linewidth=2,label='Predicted GLC')
    # plt.axvline(x=i,color='gray',alpha=0.5)
    plt.axis([0, ns+mpc.P, 0, 50])
    plt.ylabel('y(t)')
    plt.legend()
    plt.subplot(2,1,2)
    plt.step(t[:-1], u,'b-',linewidth=2,label='Measured Feed Rate')
    # plt.plot(t_hat[P:],u_hat[P:],'b.-',linewidth=2,label='Predicted Feed Rate')
    # plt.axvline(x=i,color='gray',alpha=0.5)
    plt.ylabel('u(t)')
    plt.xlabel('time')
    plt.axis([0, ns+mpc.P, 0, 0.5])
    plt.legend()
    plt.show()