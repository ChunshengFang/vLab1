import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from vLab import bioreactor_control
from simple_pid import PID

def PID_controller(set_point,
                   x0,
                   u0=0.09,
                   total_time=48,
                   sample_time=1,
                   lower_limit=0,
                   upper_limit=0.2,
                   Kc=1.0,  # Controller gain
                   Ki=1.0,  # Controller integral parameter
                   Kd=0.5  # Controller derivative parameter
                   ):
    """ Compute PID control

    :param float set_point: setpoint for glucose concentration
    :param array x0: initial state
    :param floa u0: initial feed rate
    :param int total_time: operation time (hr)
    :param int sample_time: sample/measurement time
    :param float lower_limit: lower controller output limits
    :param float upper_limit: upper controller output limits
    :param float Kc: Controller gain
    :param float Ki: Controller integral parameter
    :param float Kd: Controller derivative parameter
    :return process state, controls, time and set points
    """
    t = np.linspace(0, total_time, int(total_time / sample_time) + 1)

    Xv_ss, Glc_ss, Gln_ss, Lac_ss, NH4_ss, P1_ss, P2_ss, P3_ss, V_ss = x0
    Xv_s, Glc_s, Gln_s, Lac_s, NH4_s, P1_s, P2_s, P3_s, V_s = [Xv_ss], [Glc_ss], [Gln_ss], [Lac_ss], [NH4_ss], [
        P1_ss], [P2_ss], [P3_ss], [V_ss]
    u = [u0]

    # setpoint
    if type(set_point) == np.ndarray:
        sp = set_point
    elif type(set_point) == float or type(set_point) == int:
        sp = np.ones(len(t) + 1) * float(set_point)
    else:
        print('Set point was not unspecified. Use default setpoint instead.')
        sp = np.ones(len(t) + 1) * float(set_point)

    # create PID controller
    # op = op_bias + Kc * e + Ki * ei + Kd * ed
    #      with ei = error integral
    #      with ed = error derivative

    pid = PID(Kc, Ki, Kd)
    # lower and upper controller output limits
    pid.output_limits = (lower_limit, upper_limit)
    # PID sample time
    pid.sample_time = sample_time

    ini_state = x0.copy()
    # Simulate CSTR
    for i in range(len(t) - 1):
        # PID control
        pid.setpoint = sp[i]
        action = pid(Glc_s[-1], dt=sample_time)
        u.append(action)
        print(i, pid.setpoint, Glc_s[-1], action)
        # cntrl = K @ np.array([Xv_s[i], Glc_s[i], Gln_s[i], P1_s[i]])
        # u[i+1] = cntrl[0]

        ts = [t[i], t[i + 1]]
        y = odeint(bioreactor_control, x0, ts, args=(u[i + 1],))
        Xv_s.append(y[-1, 0])
        Glc_s.append(y[-1, 1])
        Gln_s.append(y[-1, 2])
        Lac_s.append(y[-1, 3])
        NH4_s.append(y[-1, 4])
        P1_s.append(y[-1, 5])
        P2_s.append(y[-1, 6])
        P3_s.append(y[-1, 7])
        V_s.append(y[-1, 8])
        x0 = y[-1, :].copy()

    return Xv_s, Glc_s, Gln_s, Lac_s, NH4_s, P1_s, P2_s, P3_s, V_s, u, t, sp


if __name__ == '__main__':
    # Steady State Initial Conditions for the States
    x0 = [100, 40, 10, 0., 0., 0, 0.e+00, 0, 5.e-01]
    Xv, Glc, Gln, Lac, NH4, P1, P2, P3, V, u, t, sp = PID_controller(40, x0, u0=1.5 * 60 / 1e3, total_time=96,
                                                                     sample_time=1, lower_limit=0, upper_limit=0.2)

    plt.figure(figsize=(15, 8))
    # Plot the results
    plt.subplot(4, 1, 1)
    plt.plot(t[0:-1], u[0:-1], 'b--', linewidth=3)
    plt.ylabel('Feed Rate')
    plt.legend(['Jacket Temperature'], loc='best')

    plt.subplot(4, 1, 2)
    plt.plot(t[0:-1], Glc[0:-1], 'r-', linewidth=3)
    plt.ylabel('Glucose (mM)')
    plt.legend(['Glucose Concentration'], loc='best')

    plt.subplot(4, 1, 3)
    plt.plot(t[0:-1], Gln[0:-1], 'r-', linewidth=3)
    plt.ylabel('Glutamine (mM)')
    plt.legend(['Glutamine Concentration'], loc='best')

    plt.subplot(4, 1, 4)
    plt.plot([t[0], t[-1]], [50.0, 50.0], 'r-', linewidth=2)
    plt.plot(t[0:-1], Glc[0:-1], 'b.-', linewidth=3)
    plt.plot(t[0:-1], sp[0:-2], 'k:', linewidth=3)
    plt.ylabel('Concentration (mM)')
    plt.xlabel('Time (h)')
    plt.legend(['Upper Limit', 'Glucose', 'Set Point'], loc='best')
    plt.show()
