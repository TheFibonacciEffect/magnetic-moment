# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
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
# fit data
from scipy.optimize import curve_fit

# def B(r, m):
#     return const.mu_0/(4*np.pi*r**3)*(3*(m/np.linalg.norm(m))*np.array([0, 0, 1]) - m)

# def B_fit(r, m1, m2, m3):
#     return B(r, np.array([m1, m2, m3]))

# def Bz(r, m):
#     return B(r, np.array([0, 0, m]))[2]

def Bx(r, m):
    return const.mu_0/(4*np.pi*r**3)*2*m

def Bz(r, m):
    return -const.mu_0/(4*np.pi*r**3)*m

popt_cube, pcov_cube = curve_fit(Bz, df_front['distance[mm]'], df_front['cube flux density [T]'], p0=[1])
# popt_small_cylinder, pcov_small_cylinder = curve_fit(Bz, df_front['distance[mm]'], df_front['small cylinder flux density [T]'], p0=[1])
popt_big_cylinder, pcov_big_cylinder = curve_fit(Bz, df_front['distance[mm]'], df_front['big cylinder flux density [T]'], p0=[1])

# %%
# fit inverse polynomial
# np.polyfit
plt.plot(1/df_front['distance[mm]'],df_front['cube flux density [T]'],"o")

coefficients_cube = np.polyfit(1/df_front['distance[mm]'],df_front['cube flux density [T]'],3)
coefficients_big_cylinder = np.polyfit(1/df_front['distance[mm]'],df_front['big cylinder flux density [T]'],3)
# coefficients_small_cylinder = np.polyfit(1/df_front['distance[mm]'],df_front['small cylinder flux density [T]'],3)

plt.plot(1/df_front['distance[mm]'],np.polyval(coefficients_cube,1/df_front['distance[mm]']))
plt.plot(1/df_front['distance[mm]'],np.polyval(coefficients_big_cylinder,1/df_front['distance[mm]']))
# plt.plot(1/df_front['distance[mm]'],np.polyval(coefficients_small_cylinder,1/df_front['distance[mm]']))
plt.title("plot fit in inverse space")
plt.xlabel("distance [mm]")
plt.ylabel("flux density [T]")
plt.legend(["data","cube","big cylinder"])
plt.grid()
# %%
# plot fit
plt.plot(df_front['distance[mm]'],df_front['cube flux density [T]'],"o")
plt.plot(df_front['distance[mm]'],coefficients_cube[0]*df_front['distance[mm]']**(-3.)+coefficients_cube[1]*df_front['distance[mm]']**(-2.)+coefficients_cube[2]*df_front['distance[mm]']**(-1.)+coefficients_cube[3])
# %%
# plot fit
plt.title("Magneticfield in front of the magnet")
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(df_front['distance[mm]'], df_front['cube flux density [T]'],"o", label='cube flux density [T]')
ax[0].plot(df_front['distance[mm]'], Bz(df_front['distance[mm]'], *popt_cube), label='fit')
ax[0].set_xlabel('distance [mm]')
ax[0].set_ylabel('flux density [T]')
ax[0].legend()

