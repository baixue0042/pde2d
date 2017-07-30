'''
Created on Jul 19, 2017

@author: baixue
'''
import numpy as np

def derivative_value(u, t, k):
    RT, RD, A = u
    dRTdt = (1 - k[0] + k[0] * RT ** 2 / (RT ** 2 + k[1] ** 2)) * RD - (1 + k[2] * A) * RT
    dRDdt = k[4] * (1 - RD) - k[5] * dRTdt
    dAdt = k[3] * (RT - A)
    return [dRTdt, dRDdt, dAdt]

def derivative_value_simplified(u, t, k):
    RT, A = u
    dRTdt = (1 - k[0] + k[0] * RT ** 2 / (RT ** 2 + k[1] ** 2)) - (1 + k[2] * A) * RT
    dAdt = k[3] * (RT - A)
    return [dRTdt, dAdt]

def nc_RT(x, k):
    return ((1 - k[0]) / x + k[0] * x / (x ** 2 + k[1] ** 2) - 1) / k[2]

def nc_RT_derivative(x, k):
    return (-(1 - k[0]) / x ** 2 + k[0] / (x ** 2 + k[1] ** 2) + k[0] * x / (x ** 2 + k[1] ** 2) ** 2 * (-2 * x)) / k[2]

def nc_RT_A_diff(x, k):
    return ((1 - k[0]) / x + k[0] * x / (x ** 2 + k[1] ** 2) - 1) / k[2] - x
    
def solve_nc_RT_peaks(k):
    RT = np.linspace(0.00001, 1, 10000)
    index = np.where(np.diff(np.sign(nc_RT_derivative(RT, k))))[0]
    return [RT[i] for i in index]

def solve_ss(k):
    RT = np.linspace(0.00001, 1, 10000)
    index = np.where(np.diff(np.sign(nc_RT_A_diff(RT, k))))[0]
    return [RT[i] for i in index]

def quiverplot(ax, k, RTmin, RTmax, RTn, Amin, Amax, An):
    RTgrid, Agrid = np.meshgrid(np.linspace(RTmin, RTmax, RTn), np.linspace(Amin, Amax, An))
    dRT_dA_grid = np.array([derivative_value_simplified([RTgrid.ravel()[i], Agrid.ravel()[i]], 0, k) for i in range(len(RTgrid.ravel()))])
    dRT_grid, dA_grid = dRT_dA_grid[:, 0], dRT_dA_grid[:, 1]
    amp_grid = np.sqrt(dRT_grid ** 2 + dA_grid ** 2)
    ax.quiver(RTgrid.ravel(), Agrid.ravel(), dRT_grid / amp_grid, dA_grid / amp_grid, amp_grid)

def random_choice():
    k0 = np.random.uniform(10 ** -6, 1)
    k1 = np.random.uniform(10 ** -6, 5)
    k2 = np.random.uniform(10 ** -6, 1000)
    k3 = 0.1
    k4 = 0
    k5 = 0
    return [k0, k1, k2, k3, k4, k5]

def params_bistable(k):
    from scipy.integrate import odeint
    decision = False
    ss = solve_ss(k)
    if (len(ss) == 3):
        ss2 = [ss[2], 1, ss[2]]
        peaks = solve_nc_RT_peaks(k)
        peaks_A = [nc_RT(p, k) for p in peaks]
        if (min(peaks_A) > 0):  # all peaks of A nullcline is above x axis
            t_total = 300
            c = odeint(derivative_value, [ss[2] * 1.1, 1, ss[2]], np.arange(0, t_total), args=tuple([k]))
            if ((np.abs(c[-1] - ss2)).max() < 10 ** -4) & ((np.abs(c[-1] - c[-t_total / 3])).max() < 10 ** -4):  # the third ss is stable
                c = odeint(derivative_value, [ss[0] * 2.0, 1, ss[0]], np.arange(0, t_total), args=tuple([k]))
                if ((np.abs(c[-1] - ss2)).max() < 10 ** -4):  # increase RT level from ss[0] causes system to end up in ss[2]
                    decision = True
    return decision

def find_params_bistable(N):
    params_list = []
    n = 0
    while n < N:
        k = random_choice()
        result = params_bistable(k)
        if result:
            params_list.append(k)
            n += 1
            print n
    params_list = np.array(params_list)
    np.save('params_list', params_list)

if __name__ == '__main__':
    find_params_bistable(100)
