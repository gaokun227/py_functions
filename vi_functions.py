import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from netCDF4 import Dataset
import glob
import os

def read_tcvital(filename):
    tc_dict = {}
    f = open(filename, "r")
    for line in f:
        tc_tmp = {}
        L = line.split()
        tc_id = L[1]
        lon = float(L[6][:-1])/10
        lat = float(L[5][:-1])/10
        vmax = float(L[12])
        tc_tmp['lon'] = lon
        tc_tmp['lat'] = lat
        tc_tmp['vmax'] = vmax
        tc_dict[tc_id] = tc_tmp
    return tc_dict 

def find_center(var, lon, lat, tc_lon, tc_lat):
    dlon = lon - (360-tc_lon)
    dlat = lat - tc_lat

    dist = (dlon**2+dlat**2)**0.5
    center = np.where(dist == np.nanmin(dist))
    ic, jc = center[0][0], center[1][0]

    box_half_width = 0.5 # deg
    res = 1./33 # res in deg
    scope = np.int(box_half_width/res)

    var_sel = var[ic-scope:ic+scope, jc-scope:jc+scope]
    lon_sel = lon[ic-scope:ic+scope, jc-scope:jc+scope]
    lat_sel = lat[ic-scope:ic+scope, jc-scope:jc+scope]

    detected_center =  np.where(var_sel == np.nanmin(var_sel))

    ic_new, jc_new = detected_center[0][0], detected_center[1][0]

    tc_lon_new = 360-lon_sel[ic_new,jc_new]
    tc_lat_new = lat_sel[ic_new,jc_new]

    return tc_lon_new, tc_lat_new


def detect_tc_center_from_ic(ic_dir, tc_lon, tc_lat, opt=0):

    file1 = 'gfs_data.tile7.nc'
    file2 = 'sfc_data.tile7.nc'
    f1 = Dataset(ic_dir+'/'+file1, 'r')
    f2 = Dataset(ic_dir+'/'+file2, 'r')

    if opt == 0:
       lon = f1.variables['geolon'][:]
       lat = f1.variables['geolat'][:]
       var = np.squeeze(f1.variables['ps'][:])
       slmsk = np.squeeze(f2.variables['slmsk'][:])
       slmsk[slmsk==1]=np.nan
       slmsk[slmsk==0]=1
       var = var*slmsk
       tc_lon_new, tc_lat_new = find_center(var, lon, lat, tc_lon, tc_lat)

    elif opt == 1:
       lon1 = f1.variables['geolon_s'][:]
       lat1 = f1.variables['geolat_s'][:]
       u1 = np.squeeze(f1.variables['u_s'][-1,:,:])
       v1 = np.squeeze(f1.variables['v_s'][-1,:,:])
       wind1 = np.sqrt(u1**2+v1**2)

       lon2 = f1.variables['geolon_w'][:]
       lat2 = f1.variables['geolat_w'][:]
       u2 = np.squeeze(f1.variables['u_w'][-1,:,:])
       v2 = np.squeeze(f1.variables['v_w'][-1,:,:])
       wind2 = np.sqrt(u2**2+v2**2)

       tc_lon_new1, tc_lat_new1 = find_center(wind1, lon1, lat1, tc_lon, tc_lat)
       tc_lon_new2, tc_lat_new2 = find_center(wind2, lon2, lat2, tc_lon, tc_lat)
       tc_lon_new = 0.5*(tc_lon_new1+tc_lon_new2)
       tc_lat_new = 0.5*(tc_lat_new1+tc_lat_new2)

    return np.round(tc_lon_new,1), np.round(tc_lat_new,1)
