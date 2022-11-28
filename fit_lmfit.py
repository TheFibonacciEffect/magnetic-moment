# %%
import lmfit
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import pandas as pd
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
# fit using lmfit
def Bx_dp(r, m,r0):
    return const.mu_0/(4*np.pi)*2*m*1/(r-r0)**3 #+ Q/(r-r0)**4

def Bx_with_qpole(r, m,Q,r0):
    return const.mu_0/(4*np.pi)*2*(m*1/(r-r0)**3 + const.mu_0/(4*np.pi)*Q/(r-r0)**4)


def Bz(r, m):
    return -const.mu_0/(4*np.pi*r**3)*m

# %%
# withput Qpole
from lmfit import Model
model_dp = Model(Bx_dp)
params = model_dp.make_params(m=4.6611e+08,r0=-0.05)
result = model_dp.fit(df_front['cube flux density [T]'], params, r=df_front['distance[mm]'])
print(result.fit_report())

# %%

# with Qpole
model_qpole = Model(Bx_with_qpole)
params_qp = model_qpole.make_params(m=4.6611e+08,Q=0,r0=-0.05)
result_qp = model_qpole.fit(df_front['cube flux density [T]'], params_qp, r=df_front['distance[mm]'])
print(result_qp.fit_report())
# %%
# plot fit
plt.plot(df_front['distance[mm]'], df_front['cube flux density [T]'], 'bo')
plt.plot(df_front['distance[mm]'], result.best_fit, 'r--', label='without Qpole')
plt.plot(df_front['distance[mm]'], result_qp.best_fit, "--",c="orange", label='with Qpole')

plt.xlabel('distance [mm]')
plt.ylabel('flux density [T]')
plt.title("Magneticfield in front of the magnet")
plt.grid()
plt.legend()
# %%
# evaluate fit at r=50mm
r = 50
print(f"without Qpole: {result.eval(r=r)*1e4}")
print(f"with Qpole: {result_qp.eval(r=r)*1e4}")

# %%
x = np.linspace(0, 50, 100) #mm
plt.plot(x, result.eval(r=x)*1e4, 'r--', label='without Qpole in Gauss')
plt.plot(x, result_qp.eval(r=x)*1e4, "--",c="orange", label='with Qpole in Gauss')
plt.xlabel('distance [mm]')
plt.ylabel('flux density [Gauss]')
plt.title("Magneticfield in front of the magnet")
plt.grid()
plt.legend()