# %%
# import conda
# conda_file_dir = conda.__file__
# conda_dir = conda_file_dir.split('lib')[0]
# proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
# os.environ["PROJ_LIB"] = proj_lib
from netCDF4 import Dataset
from wrf import getvar, to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,cartopy_ylim, latlon_coords, get_basemap, interplevel
import matplotlib.pyplot as plt
import matplotlib.colors
import xarray as xr
import numpy as np
import math
# import imageio
import glob 
import os
# from goes2go import GOES
# %%

def calc_LI_TT_SWI_Q(z,P,t,T2,LCL,Td,td_2,w,wspd,wdir,qi,qs,qg,lat):
    # Lifted Index calculation
    lapse_d = 0.0098 # °K /m
    g = 9.8 # m s-2
    Lv = 2.25e6 # J kg-1
    Rsd = 287.0528 # J kg-1 K-1
    Rsw = 461.5 # J kg-1 K-1
    Cp = 1004 # J kg-1 K-1
    e_0 = 611.3 # Pa
    b = 5423 # K
    T0 = 273.15 # K
    epsilon = 0.622 #kg kg-1

    t_500 = interplevel(t,P,50000)
    t_700 = interplevel(t,P,70000)
    t_850 = t[0,:,:]

    t_700 = interplevel(t,P,85000)
    td_850 = Td[1,:,:]

    delta_t = t_850 - t_500
    delta_td = t_850 - td_850
    t_prime_500 = t_500 - ((T2 - td_2)/4)
    t_prime_700 = t_700 - ((T2 - td_2)/4)
    t_prime_850 = t_850 - ((T2 - td_2)/4)

    K = (t_prime_850 - t_prime_500) + (td_850 - t_prime_700)

    t_0 = t[0,:,:]

    z_500 = interplevel(z,P,50000)
    t_500 = interplevel(t,P,50000)

    t_diff_LCL = - lapse_d * LCL

    es = e_0 * np.exp(b*(1/T0 - 1/t)) # saturated vapour pressure (Pa)
    r = epsilon * es / (P - es) # mixing ratio

    # wet adiabatic lapse rate in °K/m
    lapse_w = g * (1 + Lv*r/(Rsd*t)) / (Cp + Lv**2*r/(Rsw*t**2))

    # makes 3D array of wet lapse rate where LCL < z < z_500 
    lapse_3d_1 = np.where(z >= LCL, lapse_w, 0)
    lapse_3d_2 = np.where(P > 50000,lapse_3d_1,0)

    # 2D array of zeros 
    t_diff_w = np.zeros_like(np.sum(lapse_w,axis=0))

    # itterates over the vertical levels
    for ii in range(lapse_w[:,0,0].shape[0]-1):
        t_diff_w -= lapse_3d_2[ii,:,:] * (z[ii+1,:,:] - z[ii,:,:])

    # temp of air parcel lifted to 50000 Pa
    t_parcel_500 = t_0 + t_diff_LCL + t_diff_w 

    # Lifted index
    LI = t_parcel_500 - t_500


    # Total Totals index
    VT = t_850 - t_500
    CT = td_850 - t_500
    TT = VT + CT

    # Severe weather index (SWI)
    wspd_850 = wspd[0,:,:]
    wspd_500 = interplevel(wspd,P,50000)
    wdir_850 = wdir[0,:,:]
    wdir_500 = interplevel(wdir,P,50000)

    # first term
    term_11 = 20*(TT - 49)
    term_1 = np.where(term_11<0,0,term_11)

    # second term
    term_22 = 12*(td_850-273)
    term_2 = np.where(term_22<0,0,term_22)

    # Last term
    wdir8 = np.where(wdir_850<130,0,wdir_850)
    wdir85 = np.where(wdir_850>250,0,wdir8)
    wdir5 = np.where(wdir_500<210,0,wdir_500)
    wdir50 = np.where(wdir_500>310,0,wdir5)
    term = np.where((wdir50 - wdir85)<0,0,(wdir50 - wdir85))
    conv = 1.94584 # kts / m -s
    boolwspd = np.logical_and(wspd_850 > 15*conv, wspd_500 > 15*conv)
    wspd_condition = np.where(boolwspd,0,1)

    term_55 = 125*(np.sin(term)+ 0.2)*wspd_condition
    term_5 = np.where(term_55<0,0,term_55)

    SWI = term_1 + term_2 + 2*wspd_850 + wspd_500 + term_5

    # Experimental Q quantifying interactions within the chargin zone
    lat_flat = lat.to_numpy().flatten()
    lat_norm = 1-np.reshape((lat_flat - np.sort(lat_flat)[0]) / (-np.sort(-lat_flat)[0] - np.sort(lat_flat)[0]),t_0.shape)

    qi1 = (np.abs(qi)*1000)**(1/4)*lat_norm # ice mixing ratio (g/kg)
    qs1 = (np.abs(qs)*1000)**(1/7)*lat_norm # snow mixing ratio (g/kg)
    qg1 = np.log(np.abs(qg)*1000+1)**(1/100)*lat_norm # graupel mixing ratio (g/kg)

    # getting 0°C isotherm geopotential height (km)
    ht_0C = interplevel(z,t,273)
    # getting -20°C isotherm geopotential height (km)
    ht_20C = interplevel(z,t,253)

    qs_lis = [qi1,qs1,qg1]
    q_new = []
    for q in qs_lis:
        # charging zone
        Q1 = np.where(z < ht_0C, 0, q)
        Q2 = np.where(z > ht_20C, 0, Q1)
        q2d = np.sum(Q2,axis=0)

        q1d = q2d.flatten()
        norm = (q1d - np.sort(q1d)[0]) / (-np.sort(-q1d)[0] - np.sort(q1d)[0])
        q_new.append(np.reshape(norm,t_0.shape))
        
    Q = q_new[0]+q_new[1]+0.3*q_new[2]

    return K, LI, TT, SWI, Q
