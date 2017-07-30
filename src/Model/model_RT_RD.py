import numpy as np

def derivative_value(u, t, k):
    # k = [k0, k1, 0, 0, k4, 1]
    RT, RD = u
    dRTdt = (1 - k[0] + k[0] * RT ** 2 / (RT ** 2 + k[1] ** 2)) * RD - RT
    dRDdt = k[4] * (1 - RD) - k[5] * dRTdt
    return [dRTdt, dRDdt]

def nc_RT(x, k):
    return x / (1 - k[0] + k[0] * x ** 2 / (x ** 2 + k[1] ** 2))

def nc_RD(x, k):
    return 1

def nc_RT_RD_diff(x, k):
    return nc_RT(x, k) - nc_RD(x, k)
    
def solve_ss(RT, k):
    index = np.where(np.diff(np.sign(nc_RT_RD_diff(RT, k))))[0]
    return [RT[i] for i in index]

def params_bistable(k):
    decision = False
    RT = np.linspace(0.00001, 1, 10000)
    ss = solve_ss(RT, k)
    if (len(ss) == 3):
        decision = True
    return decision
