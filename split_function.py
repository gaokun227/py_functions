import numpy as np
import math

# =====================================================
#  function to obatin the azimuthal-mean profile for 
#  a 2d var based on slected theta range 
# =====================================================

def azimuthal_mean(xm, ym, var_2d, theta_str, theta_end, radius_list, r_bin):

     # --- define const.
     dr = radius_list[1]-radius_list[0]
     #print(dx, dr)

     # --- get dims
     ny, nx = np.shape(xm)
     nr = len(radius_list)

     # --- get radius and theta for input grid
     radius_sel = np.sqrt(xm**2 + ym**2)
     theta_sel = np.zeros(np.shape(xm))
     for j in np.arange(ny):
         for i in np.arange(nx):
             theta_sel[j,i] = math.degrees(math.atan2(ym[j,i],xm[j,i]))
             if (theta_sel[j,i] < 0):
                 theta_sel[j,i] = 360 + theta_sel[j,i]

     # --- derive the mean radial prof 

     var_mean = np.zeros(nr)

     for j in np.arange(nr):

         theta = 0.5*(theta_str + theta_end)
         theta_bin = 0.5*(theta_end - theta_str) 
         theta_diff = abs(theta_sel-theta)
         theta_diff [theta_diff >= 180. ] = 360 - theta_diff [theta_diff >= 180. ]
         mask1 = theta_diff <= theta_bin

         radius = radius_list[j]

         mask2 = abs(radius_sel-radius) <= r_bin
         var_masked = np.ma.MaskedArray(var_2d,mask=~mask1)
         var_masked = np.ma.MaskedArray(var_masked,mask=~mask2)

         var_mean[j] = np.nanmean(np.nanmean(var_masked))

     return var_mean

# =====================================================
#  function to split total 2d var to tc and eddy comp
# =====================================================


def split_tc_eddy(xm, ym, var_2d, dx, theta_list, radius_list, theta_bin_ref):

# === algorithm:
# 1) obtain mean prof along pre-defined azimuthal angles (theta_list) via bin averaging
# 2) remap the mean prof to xm, ym as mean var
# 3) subtract mean from raw 2d field as pert.

# === input:
# - xm, ym, var_2d: 2D x, y, var
# - theta_list, radius_list: 1D theta and radius where the mean var will be defined on
# - dx: dx for 2D var
# - theta_bin_ref: theta bin size for azimuthal averaging

# === output:
# 2D mean and pert defined on xm and ym
# for wind analysis, vr and vt can be projected after this call based on the 2D mean and pert components

     # --- define const.
     dr = radius_list[1]-radius_list[0]
     #print(dx, dr)
 
     # --- get dims
     ny, nx = np.shape(xm)
     ntheta = len(theta_list)
     nr = len(radius_list)

     # --- get radius and theta for input grid
     radius_sel = np.sqrt(xm**2 + ym**2)
     theta_sel = np.zeros(np.shape(xm))
     for j in np.arange(ny):
         for i in np.arange(nx):
             theta_sel[j,i] = math.degrees(math.atan2(ym[j,i],xm[j,i]))
             if (theta_sel[j,i] < 0):
                 theta_sel[j,i] = 360 + theta_sel[j,i]

     # --- derive the mean var on r-theta mesh (created based on theta_list, radius_list)

     var_mean = np.zeros((ntheta+1, nr))
     #rmw_list = []
     for i in np.arange(ntheta):
        for j in np.arange(nr):
           theta = theta_list[i]
           radius = radius_list[j]

           r_bin = dx
           theta_bin = theta_bin_ref
           if (radius <= 5 and theta_bin_ref < 10.):
               theta_bin = theta_bin_ref * 2.

           theta_diff = abs(theta_sel-theta)
           theta_diff [theta_diff >= 180. ] = 360 - theta_diff [theta_diff >= 180. ] 
           mask1 = theta_diff <= theta_bin
           mask2 = abs(radius_sel-radius) <= r_bin
           var_masked = np.ma.MaskedArray(var_2d,mask=~mask1)
           var_masked = np.ma.MaskedArray(var_masked,mask=~mask2)

           var_mean[i,j] = np.nanmean(np.nanmean(var_masked))

        #rind = np.argmax(var_mean[i,:])
        #rmw_list.append(radius_list[rind])

        # pad data with the profile along the first theta 
        theta_list = list(theta_list)
        theta_list.append(theta_list[0]+360.)
        var_mean[-1,:] = var_mean[0,:] 

     # --- get the x and y for location of max wind along each theta
     #rmw_x = []
     #rmw_y = []
     #for theta, rmw in zip(theta_list, rmw_list):
     #    rmw_x.append(rmw*np.cos(np.radians(theta)))
     #    rmw_y.append(rmw*np.sin(np.radians(theta)))

     # --- get the mean var and pert at each grid point
     var_mean_2d = np.zeros(np.shape(var_2d))
     var_pert_2d = np.zeros(np.shape(var_2d)) 
     for j in np.arange(ny):
         for i in np.arange(nx):
             theta = theta_sel[j,i]
             radius = radius_sel[j,i]
             if (radius >= radius_list[2] and radius < radius_list[-1]-dr):
                 for tind in np.arange(ntheta):
                     if (theta_list[tind] <= theta and theta_list[tind+1] > theta):
                         t1 = tind
                         t2 = tind+1
                         theta1 = theta_list[t1]
                         theta2 = theta_list[t2]
                         break
                 for rind in np.arange(nr-1):
                     if (radius_list[rind] <= radius and radius_list[rind+1] > radius):
                         r1 = rind
                         r2 = rind+1
                         radius1 = radius_list[r1]
                         radius2 = radius_list[r2]
                         break

                 # - grid layout:
                 #  (t2,r2)         (t1,r2)
                 #     p_b    p     p_a
                 #    (t2,r1)    (t1,r1)

                 # - get mean-var value at p_a and p_b first
                 # - then get mean_var value at p

                 weight_r1 = np.abs(radius-radius2)/(radius2-radius1)
                 weight_r2 = np.abs(radius-radius1)/(radius2-radius1)
                 weight_t1 = np.abs(theta-theta2)/(theta2-theta1)
                 weight_t2 = np.abs(theta-theta1)/(theta2-theta1)
                 #print(weight_r1, weight_r2, weight_t1, weight_t2)
                 var_a = weight_r1*var_mean[t1,r1] + weight_r2*var_mean[t1,r2]
                 var_b = weight_r1*var_mean[t2,r1] + weight_r2*var_mean[t2,r2]
                 var_mean_2d[j,i] = weight_t1 * var_a + weight_t2 * var_b
                 var_pert_2d[j,i] = var_2d[j,i] - var_mean_2d[j,i] 
 
     return var_mean_2d, var_pert_2d # rmw_x, rmw_y
