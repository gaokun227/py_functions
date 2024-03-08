import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from netCDF4 import Dataset
import glob
import os

def map_dvar_to_cgrid(dvar):

    # map increment field (dvar) on a-grid to c-grid
    nz, ny, nx = np.shape(dvar)
    dvar_w = np.zeros((nz, ny,   nx+1))
    dvar_s = np.zeros((nz, ny+1, nx))

    dvar_w[:, :, 1:nx] = 0.5 * ( dvar[:, :, 0:nx-1] + dvar[:, :, 1:nx] )
    dvar_w[:, :, 0]    = 0.5 *   dvar[:, :, 0]
    dvar_w[:, :, nx]   = 0.5 *   dvar[:, :, nx-1]

    dvar_s[:, 1:ny, :] = 0.5 * ( dvar[:, 0:ny-1, :] + dvar[:, 1:ny, :] )
    dvar_s[:, 0,    :] = 0.5 *   dvar[:, 0,      :]
    dvar_s[:, ny,   :] = 0.5 *   dvar[:, ny-1,   :]

    return dvar_w, dvar_s 

def selected_data_wind(xm, ym, u, v, deg_sel):
    scope = deg_sel*33
    dist = (xm**2 + ym**2)**0.5
    detected_center =  np.where(dist == np.min(dist))
    ic, jc = detected_center[0][0], detected_center[1][0]
    istr = ic - scope
    iend = ic + scope
    jstr = jc - scope
    jend = jc + scope

    xm = xm[istr:iend,jstr:jend]
    ym = ym[istr:iend,jstr:jend]
    u = u[istr:iend,jstr:jend]
    v = v[istr:iend,jstr:jend]

    return xm, ym, u, v

def find_min_dist_tc_center_and_domain_edges(tc_lon, tc_lat, grid_lont, grid_latt):

  ny, nx = np.shape(grid_lont)
  dist = (grid_latt-tc_lat)**2 + (grid_lont-tc_lon)**2
  detected_center =  np.where(dist == np.min(dist))
  jc, ic = detected_center[0][0], detected_center[1][0]

  dist_e = ic
  dist_w = nx-ic
  dist_s = ny-jc
  dist_n = jc

  return dist_e, dist_w, dist_s, dist_n

def read_tcvitals(filename):
    # Note TCs in input file are sorted based on intensity
    tc_dict = {}
    f = open(filename, "r")
    counter = 0
    for line in f:
        tc_tmp = {}
        L = line.split()
        tc_id = str(counter)+'_'+L[1]

        tc_tmp['lat'] = float(L[5][:-1])/10
        tc_tmp['lon'] = float(L[6][:-1])/10
        tc_tmp['roci'] = float(L[11])
        tc_tmp['vmax'] = float(L[12])
        tc_tmp['rmw'] = float(L[13])

        r1,r2,r3,r4 = float(L[14]),float(L[15]),float(L[16]),float(L[17])

        #print r1,r2,r3,r4

        r34 = 0.
        count = 0.
        for r_ in [r1, r2, r3, r4]:
            if r_ > 0 and r_ < 900:
               r34 += r_
               count += 1

        if count >= 1:
           r34 = r34/count

        tc_tmp['r34_list'] = [r1,r2,r3,r4]
        tc_tmp['r34_mean'] = r34

        tc_dict[tc_id] = tc_tmp
        counter += 1
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
       #slmsk = np.squeeze(f2.variables['slmsk'][:])
       #slmsk[slmsk==1]=np.nan
       #slmsk[slmsk==0]=1
       #var = var*slmsk
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
