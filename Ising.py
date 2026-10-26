import math
import numpy as np 
import matplotlib.pyplot as plt 
import random
from numba import njit


@njit
def ising_sim(Lat, J, h, kb, T, steps):
    M, N = Lat.shape
    beta = 1.0 / (kb * T)
    mag = np.sum(Lat)
    Mag_list = np.empty(steps + 1)
    Mag_list[0] = abs(mag)
    for k in range(steps):
        for _ in range(M * N):
            i = np.random.randint(0, M)
            j = np.random.randint(0, N)
            s = Lat[i, j]
            nn = (
                Lat[(i - 1) % M, j] +
                Lat[(i + 1) % M, j] +
                Lat[i, (j - 1) % N] +
                Lat[i, (j + 1) % N]
            )
            dE = 2 * s * (J * nn + h)

            if dE < 0 or np.random.random() < np.exp(-beta * dE):
                Lat[i, j] = -s
                mag -= 2 * s

        Mag_list[k + 1] = abs(mag)

    return Mag_list

L = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(50, 50))

T_list = np.linspace(1, 3, 50)
avg_mag = np.zeros_like(T_list)


for i, T in enumerate(T_list):
    mags = ising_sim(L.copy(), 1, 0, 1, T, 1000)

    eq = mags[500:]
    avg_mag[i] = np.mean(eq) / (100 * 100)

plt.plot(T_list, avg_mag)
#yes, the code now detects a phase change at around 2-2.25


