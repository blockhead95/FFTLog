import numpy as np
from scipy.special import gamma
from scipy.fft import rfft, irfft

def log_extrap(x, N_extrap_low, N_extrap_high,nome):

    low_x = high_x = []
    #fig,ax2=plt.subplots()
    if(N_extrap_low):
        dlnx_low = np.log(x[1]) - np.log(x[0])
        low_x = x[0] * np.exp(dlnx_low * np.arange(-N_extrap_low, 0))
        #ax2.plot(low_x, label=nome+'_low')
        #ax2.legend()
    if(N_extrap_high):
        dlnx_high= np.log(x[-1]) - np.log(x[-2])
        high_x = x[-1] * np.exp(dlnx_high * np.arange(1, N_extrap_high+1))
        #ax2.plot(high_x, label=nome+'_high')
        #ax2.legend()
    x_extrap = np.hstack((low_x, x, high_x))
    return x_extrap

def c_window(x, x_cut):
    """
    One-side window function of c_m,
    Adapted from Eq.(C1) in McEwen et al. (2016), arXiv:1603.04826
    """
    x_right = x[-1] - x_cut
    #x_left = x[0] + x_cut
    x_r = x[x[:] > x_right]
    #x_l = x[x[:] < x_left]
    beta_right = (x[-1] - x_r) / (x[-1] - x_right)
    #beta_left = (x_l - x[0]) / (x_left - x[0])
    W = np.ones(x.size)
    W[x[:] > x_right] = beta_right - 1 / (2 * np.pi) * np.sin(2 * np.pi * beta_right)
    #W[x[:] < x_left] = beta_left - 1 / (2 * np.pi) * np.sin(2 * np.pi * beta_left)
    return W


def g_m_vals(mu, q):
    '''
    g_m_vals function is adapted from FAST-PT
    g_m_vals(mu,q) = gamma( (mu+1+q)/2 ) / gamma( (mu+1-q)/2 ) = gamma(alpha+)/gamma(alpha-)
    mu = (alpha+) + (alpha-) - 1
    q = (alpha+) - (alpha-)
    switching to asymptotic form when |Im(q)| + |mu| > cut = 200
    '''
    if (mu + 1 + q.real[0] == 0):
        print("gamma(0) encountered. Please change another nu value! Try nu=1.1 .")
        exit()
    imag_q = np.imag(q)
    g_m = np.zeros(q.size, dtype=complex)
    cut = 200
    asym_q = q[np.absolute(imag_q) + np.absolute(mu) > cut]
    asym_plus = (mu + 1 + asym_q) / 2.
    asym_minus = (mu + 1 - asym_q) / 2.

    q_good = q[(np.absolute(imag_q) + np.absolute(mu) <= cut) & (q != mu + 1 + 0.0j)]

    alpha_plus = (mu + 1 + q_good) / 2.
    alpha_minus = (mu + 1 - q_good) / 2.

    g_m[(np.absolute(imag_q) + np.absolute(mu) <= cut) & (q != mu + 1 + 0.0j)] = gamma(alpha_plus) / gamma(alpha_minus)

    # asymptotic form
    g_m[np.absolute(imag_q) + np.absolute(mu) > cut] = np.exp(
            (asym_plus - 0.5) * np.log(asym_plus) - (asym_minus - 0.5) * np.log(asym_minus) - asym_q \
                + 1. / 12 * (1. / asym_plus - 1. / asym_minus) + 1. / 360. * (
                            1. / asym_minus ** 3 - 1. / asym_plus ** 3) + 1. / 1260 * (
                            1. / asym_plus ** 5 - 1. / asym_minus ** 5))

    g_m[np.where(q == mu + 1 + 0.0j)[0]] = 0. + 0.0j
    return g_m

def g_l(l, z_array):
    '''
    gl = 2^z_array * gamma((l+z_array)/2.) / gamma((3.+l-z_array)/2.)
    alpha+ = (l+z_array)/2.
    alpha- = (3.+l-z_array)/2.
    mu = (alpha+) + (alpha-) - 1 = l+0.5
    q = (alpha+) - (alpha-) = z_array - 1.5
    '''
    g_l = 2. ** z_array * g_m_vals(l + 0.5, z_array - 1.5)
    return g_l


class fftlog():

    def __init__(self, k, f_k, nu, c_w_w=0.25, N_extrap_low=0, N_extrap_high=0, N_pad=0):
        self.k_o = k
        self.f_k_o = f_k
        self.nu = nu
        self.c_w_w = c_w_w
        self.delta_l = np.log(k[1]) - np.log(k[0])
        self.N_extrap_low = N_extrap_low
        self.N_extrap_high = N_extrap_high

        # extrapolate k and f(k) linearly in log(k), and log(f(k))
        self.k = log_extrap(k, N_extrap_low, N_extrap_high, 'k')
        self.f_k = log_extrap(f_k, N_extrap_low, N_extrap_high, 'F(k)')
        # fig,ax= plt.subplots(figsize=(4,4))
        # ax.plot(self.k,self.f_k)
        # ax.set_title("log_extrap_grafico")
        self.N = self.k.size

        # zero-padding
        self.N_pad = N_pad
        if (N_pad):
            pad = np.zeros(N_pad)
            self.k = log_extrap(self.k, N_pad, N_pad, 'k_pad')
            # fig,ax4=plt.subplots()
            # ax4.plot(self.f_k, label='before padding')
            self.f_k = np.hstack((pad, self.f_k, pad))
            # ax4.plot(self.f_k, label='after padding')
            # ax4.legend()
            # fig,ax1= plt.subplots(figsize=(4,4))
            # ax1.plot(self.k,self.f_k)
            # ax1.set_title("log_extrap_grafico_pad")
            self.N += 2 * N_pad
            self.N_extrap_high += N_pad
            self.N_extrap_low += N_pad

        if (self.N % 2 == 1):  # Make sure the array sizes are even
            self.k = self.k[:-1]
            self.f_k = self.f_k[:-1]
            self.N -= 1
            if (N_pad):
                self.N_extrap_high -= 1

        self.m, self.c_m = self.get_c_m()
        self.eta_m = (2 * np.pi) / (self.N * self.delta_l) * self.m

    def get_c_m(self):
        f_b = self.f_k / (self.k ** self.nu)
        c_m = rfft(f_b)
        m = np.arange(0, self.N // 2 + 1)
        c_m = c_m * c_window(m, self.c_w_w * self.N // 2.)
        return m, c_m

    def fftlog(self, l):
        z = self.nu + 1j * self.eta_m
        r = (l + 1) / self.k[::-1]
        u_m = self.c_m * (self.k[0] * r[0]) ** (-1j * self.eta_m) * g_l(l, z)
        print(self.N_extrap_high, self.N, self.N_extrap_low)
        F_r = irfft(np.conj(u_m)) * r ** (-self.nu) * np.sqrt(np.pi) / 4
        return r[self.N_extrap_high:self.N - self.N_extrap_low], F_r[self.N_extrap_high:self.N - self.N_extrap_low]


class hankel():
    def __init__(self, k, f_k, nu, c_w_w=0.25, N_extrap_low=0, N_extrap_high=0, N_pad=0):
        self.myfftlog = fftlog(k, (k ** (5 / 2)) * f_k, nu, c_w_w, N_extrap_low, N_extrap_high)

    def hankel(self, n):
        r, F_r = self.myfftlog.fftlog(n - 0.5)
        F_r *= np.sqrt(2 * r / np.pi)
        return r, F_r
