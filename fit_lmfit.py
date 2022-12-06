# %%
import lmfit
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import pandas as pd
from lmfit import Model

# %%
# Read data
df_front = pd.read_csv('Magnetfeld-front.csv')
df_side = pd.read_csv('Magnetfeld-side.csv')
# clean data
df_front["cube flux density [T]"] = -df_front["cube flux density [T]"]
# df_front["distance [mm]"] = df_front["distance [mm]"]
df_side["cube flux density [T]"] = df_side["cube flux density [T]"]

# convert to si
df_front['distance[m]'] = df_front['distance[mm]']*1e-3
df_side['distance[m]'] = df_side['distance[mm]']*1e-3

df_side
# %%
# $\vec{B}(r)=\frac{\mu_0}{4 \pi r^3}[3(\vec{m} \cdot \hat{r}) \hat{r}-\vec{m}]$

# plot data
plt.title("Magneticfield in front of the magnet")
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].plot(df_front['distance[mm]'], df_front['cube flux density [T]'], label='cube flux density [T]')

def plot_data(df_front, df_side, ax):
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

plot_data(df_front, df_side, ax)

# %%
# fit using lmfit
def Bx_dp(r, m,r0):
    return const.mu_0/(4*np.pi)*2*m*1/(r-r0)**3 #+ Q/(r-r0)**4

def Bx_with_qpole(r, m,Q,r0):
    return const.mu_0/(4*np.pi)*2*(m*1/(r-r0)**3 + const.mu_0/(4*np.pi)*Q/(r-r0)**4)


def Bz(r, m,r0):
    return -const.mu_0/(4*np.pi*(r-r0)**3)*m

# %%
# withput Qpole
def fit_cube_dp(df_front, Bx_dp):
    model_dp = Model(Bx_dp)
    params = model_dp.make_params(m=4.6611e+08,r0=-0.05)
    result = model_dp.fit(df_front['cube flux density [T]'], params, r=df_front['distance[mm]'])
    print(result.fit_report())
    return result

result = fit_cube_dp(df_front, Bx_dp)

# %%

# with Qpole
def fit_cube_qp(df_front, Bx_with_qpole):
    model_qpole = Model(Bx_with_qpole)
    params_qp = model_qpole.make_params(m=1000,Q=-10,r0=-0.05)
    result_qp = model_qpole.fit(df_front['cube flux density [T]'], params_qp, r=df_front['distance[mm]'])
    print(result_qp.fit_report())
    return result_qp

result_qp = fit_cube_qp(df_front, Bx_with_qpole)
# %%
# plot fit
def plot_results_dipole_quadrupole(df_front, result, result_qp,parameter):
    df_front = df_front.copy()
    # if it contains NaN values, remove them
    if df_front[parameter].isnull().values.any():
        print("removing NaN values")
        df_front = df_front.dropna()
    plt.plot(df_front['distance[mm]'], df_front[parameter], 'bo')
    plt.plot(df_front['distance[mm]'], result.best_fit, 'r--', label='without Qpole')
    plt.plot(df_front['distance[mm]'], result_qp.best_fit, "--",c="orange", label='with Qpole')

    plt.xlabel('distance [mm]')
    plt.ylabel('flux density [T]')
    plt.title("Magneticfield in front of the magnet")
    plt.grid()
    plt.legend()

plot_results_dipole_quadrupole(df_front, result, result_qp,'cube flux density [T]')
plt.savefig("Magnetfeld-front.png")
# %%
# evaluate fit at r=50mm
r = 50
print(f"without Qpole: {result.eval(r=r)}")
print(f"with Qpole: {result_qp.eval(r=r)}")

# %%
def compare_results(result, result_qp):
    x = np.linspace(0, 50, 100) #mm
    plt.plot(x, result.eval(r=x)*1e4, 'r--', label='without Qpole in Gauss')
    plt.plot(x, result_qp.eval(r=x)*1e4, "--",c="orange", label='with Qpole in Gauss')
    plt.xlabel('distance [mm]')
    plt.ylabel('flux density [Gauss]')
    plt.title("Magneticfield in front of the magnet")
    plt.grid()
    plt.legend()

compare_results(result, result_qp)
# %%
def fit_Bx(x,y):
    model_dp = Model(Bx_dp)
    params = model_dp.make_params(m=4.6611e+08,r0=-0.05)
    result = model_dp.fit(y, params, r=x)
    print(result.fit_report())
    return result

# %%
def fit_large_cylinder(df_front, Bx_dp):
    model_dp = Model(Bx_dp)
    params = model_dp.make_params(m=4.6611e+08,r0=-0.05)
    result = model_dp.fit(df_front['big cylinder flux density [T]'], params, r=df_front['distance[mm]'])
    print(result.fit_report())

fit_large_cylinder(df_front, Bx_dp)

#    m:   6.9439e+08 +/- 1.1934e+08 (17.19%) (init = 4.6611e+08)
#    r0: -6.99293535 +/- 0.44472250 (6.36%) (init = -0.05)

# %%
def plot_results_dipole(df_front, result,parameter,fig,ax):
    plt.figure(fig)
    plt.sca(ax)
    df_front = df_front.copy()
    # if it contains NaN values, remove them
    if df_front[parameter].isnull().values.any():
        print("removing NaN values")
        df_front = df_front.dropna()
    plt.plot(df_front['distance[mm]'], df_front[parameter], 'bo')
    plt.plot(df_front['distance[mm]'], result.best_fit, 'r--', label='best fit')
    plt.xlabel('distance [mm]')
    plt.ylabel('flux density [T]')
    plt.title(parameter)
    plt.grid()
    plt.legend()

def fit_dipole(df_front, Bx_dp,y,x):
    model_dp = Model(Bx_dp)
    params = model_dp.make_params(m=4.6611e+08,r0=-0.05,Q=0)
    result = model_dp.fit(df_front[y], params, r=df_front[x],nan_policy='omit')
    print(result.fit_report())
    return result

fig,ax = plt.subplots(1,3,figsize=(15,5))

result_dp_cube_front = fit_dipole(df_front, Bx_dp,'cube flux density [T]','distance[m]')
result_qp_cube_front = fit_dipole(df_front, Bx_with_qpole,'cube flux density [T]','distance[m]')

df_results = pd.DataFrame(index=["cube","small cylinder","big cylinder"],columns=["front","side"])

for df,direction,f in zip([df_front, df_side],["front","side"],[Bx_dp]):
    for magnet_index,y,i in zip(["cube","small cylinder","big cylinder"],['cube flux density [T]','small cylinder flux density [T]','big cylinder flux density [T]'],[0,1,2]):
        for x in ['distance[m]']:
            result_dp = fit_dipole(df,f ,y,x)
            result_qp = fit_dipole(df, Bx_with_qpole,y,x)
            df_results.loc[magnet_index,direction] = f"{result_dp.params['m'].value} +/- {result_dp.params['m'].stderr}"
            plot_results_dipole(df, result_dp,y,fig,ax[i])

plt.savefig("fitting three panels.png")
# %%
df_results
