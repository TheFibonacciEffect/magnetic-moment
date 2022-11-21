# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.linalg import lstsq
# %%
# Read data
df_front = pd.read_csv('Magnetfeld-front.csv')
df_side = pd.read_csv('Magnetfeld-front.csv')
df_front
# clean data
df_front["cube flux density [T]"] = -df_front["cube flux density [T]"]
df_side["cube flux density [T]"] = -df_side["cube flux density [T]"]

df_front
# %%
# $\vec{B}(r)=\frac{\mu_0}{4 \pi r^3}[3(\vec{m} \cdot \hat{r}) \hat{r}-\vec{m}]$

# plot data
plt.title("Magneticfield in front of the magnet")
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].plot(df_front['distance[mm]'], df_front['cube flux density [T]'], label='cube flux density [T]')

for label in df_front.columns[1:]:
    ax[0].plot(df_front['distance[mm]'], df_front[label], label=label)
ax[0].set_xlabel('distance [mm]')
ax[0].set_ylabel('flux density [T]')
ax[0].legend()
ax[0].set_title("Magneticfield in front of the magnet")
ax[0].grid()

for label in df_side.columns[1:]:
    ax[1].plot(df_side['distance[mm]'], df_side[label], label=label)
ax[1].set_xlabel('distance [mm]')
ax[1].set_ylabel('flux density [T]')
ax[1].legend()
ax[1].set_title("Magneticfield on the side of the magnet")
ax[1].grid()
# %%

# fit the data using least squares
def Bx(r, m):
    return const.mu_0/(4*np.pi)*2*m*1/r**3

def Bz(r, m):
    return -const.mu_0/(4*np.pi*r**3)*m

def fit_params(x,y):
    M = np.atleast_2d(1/x[1:]**3).T
    weigth = np.diag(1/y[1:]**2).T
    p, res, rnk, s = lstsq(M, y[1:])
    print(res)
    plt.plot(x, y, 'o', label='data')
    plt.plot(x, p*x**(-3.), label='fit')
    plt.show()
    return p

fit_params(df_front['distance[mm]'], df_front['cube flux density [T]'])
# %%
