import numpy as np
import math as mt
from scipy.special import gamma
from scipy.fft import fft, ifft, fftshift

# sample size
N = 100

#indice sommatoria
m = np.arange(mt.ceil(-N/2), mt.ceil(N/2)+1)

#spaziatura logaritmica
r = np.logspace((10)**(-5), 10., N, endpoint=False)
L = np.log(r[1]) - np.log(r[0])

#valori costanti
a = 1; mu = 0; q = 1; kr = 1;

#valore esponente
f = (2j+np.pi*m)/L
z = q + f
#print(z[24], z[25], z[75], z[76])

#funzione f(k)
f_r=r

#fft per trovare c_m
c_m= fft(f_r)/N

#la shifto per avere ordinati c_-N/2 fino a c_N/2
c_m_sh = fftshift(f_r)

#rapporto delle gamma
num = (mu + 1 + z)/2.0

den = (mu + 1 - z)/2.0

gamma_fraction = gamma(num)/gamma(den)

#calcolo u_m

u_m = (kr)**(-f)*(2**z)*gamma_fraction

#moltiplico i due pezzi

b_m = c_m_sh*u_m

#fft back

a_m = ifft(b_m)
