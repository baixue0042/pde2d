from auxiliary import *

class Skeleton2d(object):
    def __init__(self, k_R, f_R, c0, span_time, span_space, stimulus_list, k_D, basedir):
        # set reaction parameters
        self.n_chemical = 3  # number of chemical species
        self.k_R = k_R
        self.f_R = f_R
        # set initial state and perturbation
        self.c0 = c0
        self.stimulus_list = stimulus_list
        # set spatial and temporal grid
        self.span_time = span_time  # unit: seconds
        self.span_space = span_space  # unit: micrometers
        self.n_space = 50  # number of grid points
        self.k_D = k_D  # unit: micrometers**2/seconds
        self.h_space = float(self.span_space) / self.n_space;
        v = 0.5 * (self.h_space ** 2) / max(self.k_D)  # characteristic time step based on diffusion
        self.groupsize = int(np.ceil(1.0 / v))
        self.h_time = 1.0 / self.groupsize
        # directory to save result
        self.dir = basedir + "_".join([",".join(round_sig(self.k_R)), ",".join(round_sig(self.k_D))])

    def stimulus_t(self, t):
        z = np.zeros((self.n_space ** 2, 3))
        for info in self.stimulus_list:
            chemical, start, lifetime, amp, loc = info
            if ((t >= start) & (t <= start + lifetime)):
                tmid = start + lifetime / 2.0
                z[:, chemical] += amp * (1 - np.abs(t - tmid) / (lifetime / 2.0)) * stimulus_square(self.span_space, self.n_space, loc).ravel()
                # print t, start, lifetime, z.max()
        return z
    
    def stimulus(self):
        data = np.zeros((self.span_time + 1, self.n_space ** 2, self.n_chemical))
        for t in range(self.span_time):
            data[t] = self.stimulus_t(t)
        return data
    
    def integrate(self):
        from scipy.sparse.linalg import spsolve
        # initial condition
        self.ic = np.zeros((self.n_space ** 2, self.n_chemical))
        for s in range(self.n_chemical):
            self.ic[:, s] = (np.ones((self.n_space, self.n_space)) * self.c0[s]).ravel()
        # diffusion matrix
        DM = np.empty((self.n_chemical, 2), dtype=object)  
        for s in range(self.n_chemical):
            DM[s, 0], DM[s, 1] = diffusion_matrix_2d(2 * (self.h_space ** 2) / (self.h_time * self.k_D[s]) , self.n_space);
        # time integration
        self.data = np.zeros((self.span_time + 1, self.n_space ** 2, self.n_chemical))  # dimentions: time, space, chemical
        c = self.ic + self.stimulus_t(0)
        self.data[0] = np.copy(c)
        for k in range(1, self.groupsize * self.span_time + 1):
            for i in range(self.n_space ** 2):
                c[i, :] = RK45(c[i, :], self.h_time, self.k_R, self.f_R)
            c = c + self.stimulus_t(k / float(self.groupsize))
            for s in range(self.n_chemical):
                c[:, s] = spsolve(DM[s, 0], DM[s, 1].dot(c[:, s]))
            if k % self.groupsize == 0:
                self. data[k / self.groupsize] = np.copy(c)
        # save result
        check_directory(self.dir)
        np.save(self.dir, self)
