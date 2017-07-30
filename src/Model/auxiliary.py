import numpy as np
def plot_blue_red(zero, data, datadir):
    check_directory(datadir)
    import matplotlib.pyplot as plt
    import matplotlib
    span_time, n_space, n_chemical = data.shape
    n_space = int(np.sqrt(n_space))
    plotrange = np.zeros((n_chemical, 2))
    cmap = []
    for s in range(n_chemical):
        plotrange[s, 0] = min(zero[s] - 0.1, data[:, :, s].min())
        plotrange[s, 1] = max(zero[s] + 0.1, data[:, :, s].max())
        zero_norm = (zero[s] - plotrange[s, 0]) / (plotrange[s, 1] - plotrange[s, 0])
        # print plotrange[s], zero_norm*(plotrange[s,1]-plotrange[s,0])+plotrange[s,0]
        cdict = {'red':   ((0.0, 0.0, 0.0), (zero_norm, 0.0, 0.0), (1.0, 1.0, 1.0)),
             'green': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
             'blue':  ((0.0, 0.0, 1.0), (zero_norm, 0.0, 0.0), (1.0, 0.0, 0.0))}
        cmap.append(matplotlib.colors.LinearSegmentedColormap('BlueRed', cdict))
    plt.style.use('dark_background')
    for frm in range(span_time):
        fig, axarr = plt.subplots(1, n_chemical, figsize=(n_chemical * 2, 2))
        for s in range(n_chemical):
            axarr[s].imshow(data[frm, :, s].reshape((n_space, n_space)), norm=matplotlib.colors.Normalize(plotrange[s, 0], plotrange[s, 1]), interpolation='nearest', cmap=cmap[s])
            axarr[s].xaxis.set(visible=False)
            axarr[s].yaxis.set(visible=False)
        plt.tight_layout()
        fig.savefig(datadir + '/{:0>2d}'.format(frm) + '.png', format='png')
        plt.close()
    

def stimulus_square(span_space, n_space, loc):
    cx, cy, dx, dy = loc
    icx, icy, idx, idy = [int(float(v) / span_space * n_space) for v in [cx, cy, dx, dy]]
    arr = np.zeros((n_space, n_space))
    arr[icx - idx:icx + idx, icy - idy:icy + idy] = 1
    return arr

@  np.vectorize
def round_sig(x, sig=4, small_value=1.0e-9):
    from math import log10, floor
    if type(x) is int:
        return str(x)
    else:
        return str(round(x, sig - int(floor(log10(max(abs(x), abs(small_value))))) - 1))

def check_directory(filedir):
        import os
        if not os.path.exists(filedir):
            os.makedirs(filedir)

def RK45(u, h_time, param_reaction, f_R):
    A = np.array([ [ 0, 0, 0, 0, 0, 0, 0],
        [ 1, 0, 0, 0, 0, 0, 0],
        [1.0 / 4, 3.0 / 4, 0, 0, 0, 0, 0],
        [11.0 / 9, -14.0 / 3, 40.0 / 9, 0, 0, 0, 0],
        [4843.0 / 1458, -3170.0 / 243, 8056.0 / 729, -53.0 / 162, 0, 0, 0],
        [9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656, 0, 0],
        [35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84, 0]]);
    B4 = np.array([5179.0 / 57600, 0, 7571.0 / 16695, 393.0 / 640, -92097.0 / 339200, 187.0 / 2100, 1.0 / 40]);
    B5 = np.array([  35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84, 0]);
    C = np.array([0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1, 1]);
    n_K = 7;
    K = np.zeros((n_K, len(u)))
    for m in range(n_K):
        K[m, :] = f_R(u + h_time * C[m] * A[m, :].dot(K), 0, param_reaction);
    u = u + h_time * B5.dot(K)
    return u

def diffusion_matrix_2d(alpha, n_space):
    from scipy import sparse
    M_I = sparse.lil_matrix((n_space, n_space))
    M_I.setdiag(1)
    M_I = M_I.tocoo()

    M_T1 = sparse.lil_matrix((n_space, n_space))
    M_T1.setdiag(-1, k=-1)
    M_T1.setdiag(-1, k=1)
    M_T1.setdiag(alpha + 4)
    M_T1[0, -1] = -1
    M_T1[-1, 0] = -1
    M_T1 = M_T1.tocoo()

    M_T2 = sparse.lil_matrix((n_space, n_space))
    M_T2.setdiag(1, k=-1)
    M_T2.setdiag(1, k=1)
    M_T2.setdiag(alpha - 4)
    M_T2[0, -1] = 1
    M_T2[-1, 0] = 1
    M_T2 = M_T2.tocoo()

    D, I, J = [], [], []
    for ii in range(n_space):
        D += M_T1.data.tolist()
        I += (ii * n_space + M_T1.row).tolist()
        J += (ii * n_space + M_T1.col).tolist()
    D += (-1 * M_I.data).tolist()
    I += (M_I.row).tolist()
    J += ((n_space - 1) * n_space + M_I.col).tolist()
    for ii in range(1, n_space):
        D += (-1 * M_I.data).tolist()
        I += (ii * n_space + M_I.row).tolist()
        J += ((ii - 1) * n_space + M_I.col).tolist()
    D += (-1 * M_I.data).tolist()
    I += ((n_space - 1) * n_space + M_I.row).tolist()
    J += (M_I.col).tolist()
    for ii in range(1, n_space):
        D += (-1 * M_I.data).tolist()
        I += ((ii - 1) * n_space + M_I.row).tolist()
        J += (ii * n_space + M_I.col).tolist()
    array_left = sparse.coo_matrix((D, (I, J)), shape=(n_space * n_space, n_space * n_space)).tocsr()

    D, I, J = [], [], []
    for ii in range(n_space):
        D += M_T2.data.tolist()
        I += (ii * n_space + M_T2.row).tolist()
        J += (ii * n_space + M_T2.col).tolist()
    D += (M_I.data).tolist()
    I += (M_I.row).tolist()
    J += ((n_space - 1) * n_space + M_I.col).tolist()
    for ii in range(1, n_space):
        D += (M_I.data).tolist()
        I += (ii * n_space + M_I.row).tolist()
        J += ((ii - 1) * n_space + M_I.col).tolist()
    D += (M_I.data).tolist()
    I += ((n_space - 1) * n_space + M_I.row).tolist()
    J += (M_I.col).tolist()
    for ii in range(1, n_space):
        D += (M_I.data).tolist()
        I += ((ii - 1) * n_space + M_I.row).tolist()
        J += (ii * n_space + M_I.col).tolist()
    array_right = sparse.coo_matrix((D, (I, J)), shape=(n_space * n_space, n_space * n_space)).tocsr()
    return array_left, array_right

