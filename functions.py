import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as	cm
#from mpl_toolkits.basemap import Basemap
import datetime as dt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset 
#from scipy import signal
#from scipy.signal import butter, lfilter, freqz, filtfilt
import math
from math import factorial
#import scipy.io
#from matplotlib.mlab import griddata
from scipy.interpolate import griddata
#import scipy.stats as st
import matplotlib.path as mpath
from matplotlib.patches import Polygon
import matplotlib as mpl
import matplotlib.colors as mcolors

########################################################################
# functions - calculations (interp, remap, filter ...)
########################################################################

#-----------------------------------------------------------------------
# function to pad 2D data
#-----------------------------------------------------------------------

def pad_var_2d(var, pad_value=np.nan):
   NX, _ = np.shape(var)
   var_new = np.zeros((2*NX,2*NX)) + pad_value
   ind1 = NX/2
   ind2 = NX/2+NX
   var_new[ind1:ind2,ind1:ind2] = var
   return var_new

#-----------------------------------------------------------------------
# function to mask 3D data out of selected box
#-----------------------------------------------------------------------

def mask_3dvar(var, lon_str, lon_end, lat_str, lat_end, grid_file='/work/kng/FV3_INPUT_DATA/GRID/C768r10n4_atl_new/grid_spec.nest02.tile7.nc'):

  lon_name = 'grid_lont'
  lat_name = 'grid_latt'
  fg = Dataset(grid_file, 'r')
  lat = fg.variables[lat_name][:]
  lon = fg.variables[lon_name][:]

  mask1 = lon < lon_str
  mask2 = lon > lon_end
  mask3 = lat < lat_str
  mask4 = lat > lat_end

  nz, _, _ = np.shape(var)

  var_masked = var

  for k in range(nz): 
     var_masked[k,:,:] = np.ma.MaskedArray(var[k,:,:],mask=mask1)
     var_masked[k,:,:] = np.ma.MaskedArray(var_masked[k,:,:],mask=mask2)
     var_masked[k,:,:] = np.ma.MaskedArray(var_masked[k,:,:],mask=mask3)
     var_masked[k,:,:] = np.ma.MaskedArray(var_masked[k,:,:],mask=mask4)

  return var_masked

#-----------------------------------------------------------------------
# function to cal. f
#-----------------------------------------------------------------------

def cal_f(lat):
   omega = 7.2921e-5 # rad/s
   f = 2 * omega * np.sin(np.radians(lat)) # coriolis frequency, s^-1
   return f

#-----------------------------------------------------------------------
# function to smooth 2d array
#-----------------------------------------------------------------------

def smooth_2d(var,scope):

   nx,ny = np.shape(var)
   var_s = var 

   sel_i=np.arange(scope,nx-scope)
   sel_j=np.arange(scope,ny-scope)

   for i in sel_i:
     for j in sel_j:
        var_sel = var[i-scope:i+scope+1,j-scope:j+scope+1]
        var_s[i][j] = np.mean(np.mean(var_sel,axis=0),axis=0)

   return var_s

#-----------------------------------------------------------------------
# function to smooth 1d array
#-----------------------------------------------------------------------

def smooth_1d(var,scope):

   nx = len(var)
   var_s = var

   sel_i=np.arange(scope,nx-scope)

   for i in sel_i:
        var_sel = var[i-scope:i+scope+1]
        var_s[i] = np.mean(var_sel,axis=0)

   return var_s

#-----------------------------------------------------------------------
# function to do vertical interp for 3D var
#-----------------------------------------------------------------------
 
def remap_z(var,z,zm):

  nz,ny,nx = np.shape(var)
  varm = np.zeros((len(zm),ny,nx))

  for i in np.arange(nx):
     for j in np.arange(ny):
        var1  = np.squeeze(var[:,j,i])
        z1    = np.squeeze(z[:,j,i])
        varm1 = np.interp(zm,z1[::-1],var1[::-1])
        varm[:,j,i] = varm1

  return varm

#-----------------------------------------------------------------------
# function to do griddata for 2D var
#-----------------------------------------------------------------------
 
def remap_2d(var_ori, lon_ori, lat_ori, lon_new, lat_new, skip=1):

    nt,nx,ny = np.shape(var_ori)

    nx1,ny1 = np.shape(lon_new)

    var_new = np.zeros((nt,nx1,ny1))

    for t in np.arange(nt):
        #var_new[t,:,:] = griddata(lon_ori[::skip,::skip].ravel(), lat_ori[::skip,::skip].ravel(), var_ori[t,::skip,::skip].ravel(), lon_new, lat_new, method='linear')
        var_new[t,:,:] = griddata((lon_ori[::skip,::skip].ravel(), lat_ori[::skip,::skip].ravel()), var_ori[t,::skip,::skip].ravel(), (lon_new, lat_new), method='linear')

    return var_new

#------------------------------------------------------------------------------
# function to derive z (m) and p (pa) values for native model vertical levels  
#------------------------------------------------------------------------------

def cal_zp(z0,p0,delz,delp):

  nz,ny,nx = np.shape(delp)

  pe = np.zeros((nz+1,ny,nx))
  pm = np.zeros((nz,ny,nx))

  ze = np.zeros((nz+1,ny,nx))
  zm = np.zeros((nz,ny,nx))

# calculate p 
  pe[0,:,:]=p0

  for k in np.arange(nz):
    pe[k+1,:,:]=pe[k,:,:]+delp[k,:,:]

  for k in np.arange(nz):
    pm[k,:,:]=(pe[k,:,:]*pe[k+1,:,:])**0.5

# calculate z
  ze[-1,:,:]=z0

  for k in np.arange(nz)[::-1]:
    ze[k,:,:]=ze[k+1,:,:]+delz[k,:,:]

  for k in np.arange(nz):
    zm[k,:,:]=0.5*(ze[k+1,:,:]+ze[k,:,:])

  return zm, pm

# --- simple version to get z only

def cal_z_only(z0,delz):

  nz,ny,nx = np.shape(delz)

  ze = np.zeros((nz+1,ny,nx))
  zm = np.zeros((nz,ny,nx))

# calculate z
  ze[-1,:,:]=z0

  for k in np.arange(nz)[::-1]:
    ze[k,:,:]=ze[k+1,:,:]+delz[k,:,:]

  for k in np.arange(nz):
    zm[k,:,:]=0.5*(ze[k+1,:,:]+ze[k,:,:])

  return zm

def cal_z_1d(z0,delz):

  nz = len(delz)

  ze = np.zeros(nz+1)
  zm = np.zeros(nz)

# calculate z
  ze[-1]=z0

  for k in np.arange(nz)[::-1]:
    ze[k]=ze[k+1]+delz[k]

  for k in np.arange(nz):
    zm[k]=0.5*(ze[k+1]+ze[k])

  return zm

#-----------------------------------------------------------------------
# function to calculate distance
#-----------------------------------------------------------------------

def cal_dist_2d(lonc,latc,lon,lat):

   nx,ny = np.shape(lon)
   x = np.zeros((nx,ny))
   y = np.zeros((nx,ny))
   flag_x = np.zeros((nx,ny))+1.
   flag_y = np.zeros((nx,ny))+1.

   R = 6371  

#  for x 
   lon1 = lonc
   lat1 = latc
   lon2 = lon
   lat2 = latc

   dlon = np.radians(lon2-lon1)
   dlat = np.radians(lat2-lat1)

   a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) \
     * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
   c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
   x = R * c

#  for y
   lon1 = lonc
   lat1 = latc
   lon2 = lonc
   lat2 = lat

   dlon = np.radians(lon2-lon1)
   dlat = np.radians(lat2-lat1)

   a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) \
     * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
   c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
   y = R * c

# change sign
   flag_x[lon < lonc] = -1.
   flag_y[lat < latc] = -1.
   
   x = x*flag_x
   y = y*flag_y

   return x,y

def cal_dist_2p(lon1,lat1,lon2,lat2):

   R = 6371  

   dlat = np.radians(lat2-lat1)
   dlon = np.radians(lon2-lon1)

   a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) \
     * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
   c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
   d = R * c

   return d

#-----------------------------------------------------------------------
# function to get spectrum of the 2d field
#-----------------------------------------------------------------------

def get_spectrum(var_2d, dx):

    da = 1

    # Detrend along x and y
    #phi = signal.detrend(signal.detrend(var[t,:,:],axis=1),axis=0)
    phi = np.squeeze(var_2d)

    # Apply hamming window
    xwindow = np.tile(np.hamming(phi.shape[1]),(phi.shape[0],1) )
    ywindow = np.tile((np.hamming(phi.shape[0]),),(phi.shape[1],1)).T
    phi = phi*xwindow*ywindow

    PHI = np.fft.rfft2(phi)

    nx = int(PHI.shape[0]/2.)
    ny = PHI.shape[1]
    nn = np.min(phi.shape)
    PHI = PHI[:nx,:ny].T.flatten()

    # Implementing algorithm of Denis, Cote and Laprise (MWR, 2002)
    k = np.arange(nx)/float(nx)
    l = np.arange(ny)/float(ny)
    kk, ll = np.meshgrid(k,l)
    waveno = np.floor(np.sqrt(kk**2 + ll**2)*np.min((nx,ny))/da).flatten()
    nl = np.min(( len(k), len(l) ))/da
    nl = int(nl)

    X = np.zeros(int(nl))
    
    for i in np.arange(nl):
       A = np.where(waveno == i)
       X[i] = np.sum( np.abs(PHI[A])**2 )  / da #Same amount of energy in each band, normalized with respect to one-wavenumber-wide bands

    vv = (np.arange(nl)+0.5)*da
    LL = nx*dx/vv #convert to wavelength

    #Clip wavenumber 1, which has been removed by detrending
    LL = LL[1:]
    vv = vv[1:]
    X  = X[1:]

    #plt.loglog(LL,X,color=color,alpha=0.15,zorder=-1)

    return LL, vv,  X

#-----------------------------------------------------------------------
# function to do savitzky golay filter
# Note - needs to be replaced by scipy.signal.savgol_filter
#-----------------------------------------------------------------------

'''
def savitzky_golay(y, window_size, order, deriv=0, rate=1):

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2

    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)

    # pad the signal at the extremes with values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))

    return np.convolve( m[::-1], y, mode='valid')
'''

#-----------------------------------------------------------------------
# function to do 1-2-1 filter
#-----------------------------------------------------------------------

def apply_121_filter(data, ntimes):

    data_new = data

    for i in np.arange(ntimes): 

        y = data_new
        y[1:-1] = 0.25*data_new[:-2] + 0.5*data_new[1:-1] + 0.25*data_new[2:]
        data_new = y # filtered data

    return y

#-----------------------------------------------------------------------
# function to do butterworth bandpass filter
#-----------------------------------------------------------------------

# --- design a bandpass filter
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# --- apply the filter to data
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    #y = lfilter(b, a, data)
    y = filtfilt(b, a, data)
    return y

#-----------------------------------------------------------------------
# function to get annual cycle - mean + harmonics
# Note - m specifies the first few harmonics
#-----------------------------------------------------------------------

def get_annual_cycle(var_1d, m):
 
    nn = len(var_1d) 

    ave = np.mean(var_1d)
    var_1d_a = var_1d - ave

# get coeff for the harmonics
    a,b = harm(var_1d_a,m)

# reconstruct the first few harmonics
    pm = np.zeros(nn)
    for ii in np.arange(m): 
       for jj in np.arange(nn):
           pm[jj] = pm[jj] + a[ii]*np.cos(2*np.pi*(ii+1)*(jj+1)/float(nn)) + \
                             b[ii]*np.sin(2*np.pi*(ii+1)*(jj+1)/float(nn))

# annual cycle: mean + harmonics
    datm = ave + pm 

    return datm

# --- get harmonic coeff 

def harm(t,m):

  a=np.zeros(m)
  b=np.zeros(m)

  nn=len(t)

  for i in np.arange(m):     
     for j in np.arange(nn):
        a[i] = a[i] + t[j]*np.cos(2*np.pi*(i+1)*(j+1)/float(nn))
        b[i] = b[i] + t[j]*np.sin(2*np.pi*(i+1)*(j+1)/float(nn))

     a[i]=a[i]*2/float(nn)
     b[i]=b[i]*2/float(nn)

  return a,b	

#-----------------------------------------------------------------------
# function to find index
#-----------------------------------------------------------------------

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

########################################################################
# functions - read and write
########################################################################

#-----------------------------------------------------------------------
# function to write data in a selected box (nz, ny, nx) 
#-----------------------------------------------------------------------

def write_boxed_data(filename, x, y, z, var_name_list, var_list):

  fnc = Dataset(filename, 'w',format='NETCDF4_CLASSIC')

  u = var_list[0]
  nz, ny, nx = np.shape(u)

  # create dim
  X = fnc.createDimension('X', nx)
  Y = fnc.createDimension('Y', ny)
  Z = fnc.createDimension('Z', nz)

  # create var
  var_w = fnc.createVariable('x', np.float32, ('Y', 'X'))
  var_w[:,:] = x

  var_w = fnc.createVariable('y', np.float32, ('Y', 'X'))
  var_w[:,:] = y

  var_w = fnc.createVariable('z', np.float32, ('Z'))
  var_w[:] = z 

  for var_name, var in zip(var_name_list, var_list):
    var_w = fnc.createVariable(var_name, np.float32, ('Z', 'Y', 'X'))
    var_w[:,:,:] = var

  fnc.close()

def write_azi_mean_data(filename, r, z, t, var_name_list, var_list):

  fnc = Dataset(filename, 'w',format='NETCDF4_CLASSIC')

  u = var_list[0]
  nt, nz, nr = np.shape(u)

  # create dim
  X = fnc.createDimension('X', nr)
  Z = fnc.createDimension('Z', nz)
  T = fnc.createDimension('T', nt)

  # create var
  var_w = fnc.createVariable('r', np.float32, ('X'))
  var_w[:] = r 

  var_w = fnc.createVariable('z', np.float32, ('Z'))
  var_w[:] = z 

  for var_name, var in zip(var_name_list, var_list):
    var_w = fnc.createVariable(var_name, np.float32, ('T', 'Z', 'X'))
    var_w[:,:,:] = var

  fnc.close()

#-----------------------------------------------------------------------
# function to write nc 
#-----------------------------------------------------------------------

def write_nc(var, var_name, file_name):

 fnc = Dataset(file_name, 'w',format='NETCDF4_CLASSIC')

 # create dim
 nx,ny = np.shape(var)
 X = fnc.createDimension('X', nx)
 Y = fnc.createDimension('Y', ny)

 # create var 
 var_w = fnc.createVariable(var_name, np.float32, ('X', 'Y'))
 var_w[:,:] = var
 
 fnc.close()

def write_nc_1d(var, var_name, file_name):

 fnc = Dataset(file_name, 'w',format='NETCDF4_CLASSIC')

 # create dim
 nz = len(var)
 Z = fnc.createDimension('z', nz)

 # create var
 var_w = fnc.createVariable(var_name, np.float32, ('Z'))
 var_w[:] = var

 fnc.close()

#-----------------------------------------------------------------------
# function to read nc 
#-----------------------------------------------------------------------

def read_nc(file, var_name):

    f1 = Dataset(file, 'r')

    var  = f1.variables[var_name][:]
    var  = np.squeeze(var)

    return var

def read_nc_2d(file, var_name, idx1, idx2, idy1, idy2, is_grid=False):

    f1 = Dataset(file, 'r')
    if is_grid:
        var = f1.variables[var_name][idx1:idx2+1, idy1:idy2+1]
    else:
         var = f1.variables[var_name][:,idx1:idx2+1, idy1:idy2+1]
    return np.array(var)

#-----------------------------------------------------------------------
# function to read mat file
#-----------------------------------------------------------------------

def read_mat(file_name,var_name):

    mat = scipy.io.loadmat(file_name)
    var = np.squeeze(mat.get(var_name))

    return var

#-----------------------------------------------------------------------
# function to write txt 
#-----------------------------------------------------------------------

def write_txt(data, outfile = './var.txt' ):
    output =  open(outfile,"w")
    nn = len(data)
    recfmt = '%12.6f\n'
    for i in np.arange(nn):
        recstr = recfmt % (data[i])
        output.write(recstr)
    output.close()

#-----------------------------------------------------------------------
# function to read txt
#-----------------------------------------------------------------------

def read_txt(filename):
    data = []
    f = open(filename, "r")
    for line in f:
        L = line.split()
        data += [float(L[0])]
    return np.array(data)

#-----------------------------------------------------------------------
# function to read txt with 4 var (pc1/2 from obs and model)
#-----------------------------------------------------------------------

def read_txt_4var(file):

  a=[]
  b=[]
  c=[]
  d=[]

  frmm = open(file, "r")
  for line in frmm:
        L = line.split()
        a.append(float(L[0]))
        b.append(float(L[1]))
        c.append(float(L[2]))
        d.append(float(L[3]))
  return np.array(a), np.array(b), np.array(c), np.array(d)

#-----------------------------------------------------------------------
# function to read txt with 4 var (pc1/2 from obs and model)
#-----------------------------------------------------------------------

def read_rmm_txt2(rmm_file):

  a=[]
  b=[]
  c=[]
  d=[]

  frmm = open(rmm_file, "r")
  for line in frmm:
        L = line.split()
        a.append(float(L[1]))
        b.append(float(L[2]))
        c.append(float(L[3]))
        d.append(float(L[4]))
  return np.array(a), np.array(b), np.array(c), np.array(d)

#-----------------------------------------------------------------------
# function to read txt with 4 var (pc1, pc2, amp, phase)
#-----------------------------------------------------------------------

def read_rmm_txt(rmm_file):

  a=[]
  b=[]
  c=[]
  d=[]

  frmm = open(rmm_file, "r")
  for line in frmm:
        L = line.split()
        a.append(float(L[0]))
        b.append(float(L[1]))
        c.append(float(L[2]))
        d.append(int(L[3]))
  return np.array(a), np.array(b), np.array(c), np.array(d)
     
#-----------------------------------------------------------------------
# function to write out txt with 4 var
#-----------------------------------------------------------------------

def write_rmm_txt(a, b, c, d, outfile):
	  recfmt = '%12.6f   %12.6f   %12.6f   %4d\n'
	  output =  open(outfile,"w")
	  nn = len(a)
	  for i in np.arange(nn):
	      recstr = recfmt % (a[i], b[i], c[i], d[i])
	      output.write(recstr)
	  output.close()

########################################################################
# functions - model/pred evaluation
########################################################################

#-----------------------------------------------------------------------
# function to cal ACC and RMSE for mjo indices
#-----------------------------------------------------------------------

def cal_mjo_bias(o1,o2,p1,p2):
    nrec = len(o1)
    bias = 0
    for i in np.arange(nrec):
       bias += (p1[i]**2+p2[i]**2)**0.5-(o1[i]**2+o2[i]**2)**0.5
    return bias/nrec 

def cal_mjo_acc(a1,a2,b1,b2):
    temp1=np.sum(a1*b1+a2*b2)
    temp2=np.sqrt(np.sum(a1*a1+a2*a2))
    temp3=np.sqrt(np.sum(b1*b1+b2*b2))
    acc=temp1/(temp2*temp3)
    return acc

def cal_mjo_rmse(a1,a2,b1,b2):
    temp1 = (a1-b1)**2+(a2-b2)**2
    rmse = np.sqrt(np.mean(temp1))
    return rmse

def cal_mjo_acc_rmse(a1,a2,b1,b2):
    acc = cal_mjo_acc(a1,a2,b1,b2)
    rmse = cal_mjo_rmse(a1,a2,b1,b2)
    return acc, rmse

#-----------------------------------------------------------------------
# function to cal rmse
#-----------------------------------------------------------------------

def cal_rmse(predictions, targets):
    rmse2 = np.mean ((predictions - targets)**2)
    rmse = np.sqrt(rmse2)
    return rmse

#-----------------------------------------------------------------------
# function to cal cor 
#-----------------------------------------------------------------------
 
def cal_cor(a,b):

  cor1 = st.pearsonr(a,b)[0]
  cor2 = st.spearmanr(a,b)[0]

  return round(cor1,2), round(cor2,2)

#-----------------------------------------------------------------------
# function to cal ACC
#-----------------------------------------------------------------------

def cal_acc(vara1, vara2):

    nt,nx,ny = np.shape(vara1)

    acc = np.zeros(nt)

    for t in np.arange(nt):
        var1 = np.squeeze(vara1[t,:,:])
        var2 = np.squeeze(vara2[t,:,:])
        
        acc[t] = np.sum(var1*var2)/np.sqrt(np.sum(var1*var1)*np.sum(var2*var2))
     
    return acc

def cal_acc1(var1, var2):

    nx,ny = np.shape(var1)
        
    acc = np.sum(var1*var2)/np.sqrt(np.sum(var1*var1)*np.sum(var2*var2))
     
    return acc

#-----------------------------------------------------------------------
# function to cal. prediction score - KG defination
#-----------------------------------------------------------------------

# Option 1:
# if abs(pred-obs) <= abs(clim-obs) => score = 1 (0 if not)

def cal_score1(pred, clim, obs):

    score = np.zeros(len(obs))
    for i in np.arange(len(obs)):
       if abs(pred[i]-obs[i]) <= abs(clim[i]-obs[i]):
          score[i] = 1
     
    return score

# Option 2:
# if pred_a and obs_a has same sign => score = 1 (0 if not)

def cal_score2(pred, obs):

    score = np.zeros(len(obs))#-1
    for i in np.arange(len(obs)):
       if (pred[i]*obs[i]) > 0. :
          score[i] = 1
     
    return score


#-----------------------------------------------------------------------
# function to cal. Heidke skill score - 2 cat
#-----------------------------------------------------------------------

def cal_hss_2cat(pre, obs, pre_th=0.5, obs_th=1):

    # two answers only - yes or no

    # if pre >= pre_th : yes case
    # if obs >= obs_th : yes case 

    N = len(obs)

    N_cat1 = 0

    a = 0.
    b = 0.
    c = 0.
    d = 0.

    pre_new = np.zeros(N)
    obs_new = np.zeros(N)

    for i in np.arange(N):

       if pre[i]>=pre_th:
          pre_new[i]=1
       else:
          pre_new[i]=0 

       if obs[i]>=obs_th:
          obs_new[i]=1
       else: 
          obs_new[i]=0

       if obs_new[i] == 0 and pre_new[i] == 0:
          d = d+1
          N_cat1 = N_cat1+1 
       elif obs_new[i] == 1 and pre_new[i] == 1:
          a = a+1
       elif obs_new[i] == 1 and pre_new[i] == 0:
          c = c+1
       elif obs_new[i] == 0 and pre_new[i] == 1:
          b = b+1
          N_cat1 = N_cat1+1

    N_cat2 = N - N_cat1

    pc = (a+d)/N

    term = (a+c)*(c+d)+(a+b)*(b+d)
  
    if term == 0.:
       hss = np.nan
    else:
       hss = 2*(a*d-b*c)/term

    return round(hss,2), round(pc,2), N_cat1, N_cat2

#-----------------------------------------------------------------------
# function to cal. Heidke skill score - 2 cat new 
# HSS = min(HSS_E, HSS_1, HSS_0)
#-----------------------------------------------------------------------

def cal_hss_2cat_new(pre, obs, pre_th=0.5, obs_th=1):

    # two answers only - yes or no

    # if pre >= pre_th : yes case
    # if obs >= obs_th : yes case 

    N = len(obs)

    N_cat1 = 0

    a = 0.
    b = 0.
    c = 0.
    d = 0.

    pre_new = np.zeros(N)
    obs_new = np.zeros(N)

    for i in np.arange(N):

       if pre[i]>=pre_th:
          pre_new[i]=1
       else:
          pre_new[i]=0 

       if obs[i]>=obs_th:
          obs_new[i]=1
       else: 
          obs_new[i]=0

       if obs_new[i] == 0 and pre_new[i] == 0:
          d = d+1
          N_cat1 = N_cat1+1 
       elif obs_new[i] == 1 and pre_new[i] == 1:
          a = a+1
       elif obs_new[i] == 1 and pre_new[i] == 0:
          c = c+1
       elif obs_new[i] == 0 and pre_new[i] == 1:
          b = b+1
          N_cat1 = N_cat1+1

    N_cat2 = N - N_cat1

    # hss relative to random forecasts
    term = (a+c)*(c+d)+(a+b)*(b+d)
  
    if term == 0.:
       hss_e = 0.
    else:
       hss_e = 2*(a*d-b*c)/term

    # hss relative to a constant forecast (alway yes or no)
    pc = (a+d)/N
    pc0 = float(N_cat1)/float(N)
    pc1 = float(N_cat2)/float(N)
    hss0 = (pc - pc0)/(1. - pc0)
    hss1 = (pc - pc1)/(1. - pc1)

    hss = min(hss_e, hss1, hss0)

    return round(hss,2), round(pc,2), N_cat1, N_cat2

#-----------------------------------------------------------------------
# function to cal. Heidke skill score - 3 cat
# HSS = (C-E)/(T-E)
# T - total cases
# C - correct cases by model
# E - correct cases by random prediction
#-----------------------------------------------------------------------

def cal_hss(pred_a, obs_a, th):

    score = np.zeros(len(obs_a))
 
    std1 = np.std(pred_a)
    std2 = np.std(obs_a)

# Get C

    for i in np.arange(len(obs_a)):

       # normal case
       if np.abs(obs_a[i]) < th*std2:
          if np.abs(pred_a[i]) < th*std1:   
             score[i] = 1
 
       # extreme cases
       if np.abs(obs_a[i]) >= th*std2:
          if np.abs(pred_a[i]) >= th*std1 : 
             if pred_a[i]*obs_a[i] > 0. : 
                score[i] = 1

    score_mean = np.mean(score) # percent of correct cases (C/T)
    T = len(obs_a)
    C = score_mean*T

# Get E

    cat_1o = 0.
    cat_2o = 0.
    cat_3o = 0.

    cat_1m = 0.
    cat_2m = 0.
    cat_3m = 0.

    for i in np.arange(len(obs_a)):

       if np.abs(pred_a[i]) < th*std1:  
             cat_1m = cat_1m+1

       if pred_a[i] >= th*std1:  
             cat_2m = cat_2m+1

       if pred_a[i] <= -th*std1:  
             cat_3m = cat_3m+1

       if np.abs(obs_a[i]) < th*std2:  
             cat_1o = cat_1o+1

       if obs_a[i] >= th*std2:  
             cat_2o = cat_2o+1

       if obs_a[i] <= -th*std2:  
             cat_3o = cat_3o+1

    E = cat_1m*cat_1o/T + cat_2m*cat_2o/T + cat_3m*cat_3o/T
    hss = (C-E)/(T-E)

    return round(hss,2), round(score_mean,2)

# by month

def cal_hss_new(pred_a, obs_a, th):

    score = np.zeros(len(obs_a))
 
    std1 = np.std(pred_a)
    std2 = np.std(obs_a)

# Get C

    for i in np.arange(len(obs_a)):

       # normal case
       if np.abs(obs_a[i]) < th*std2:
          if np.abs(pred_a[i]) < th*std1:   
             score[i] = 1
 
       # extreme cases
       if np.abs(obs_a[i]) >= th*std2:
          if np.abs(pred_a[i]) >= th*std1 : 
             if pred_a[i]*obs_a[i] > 0. : 
                score[i] = 1

    score_mean = np.mean(score) # percent of correct cases (C/T)
    T = len(obs_a)
    C = score_mean*T

# Get E

    cat_1o = 0.
    cat_2o = 0.
    cat_3o = 0.

    cat_1m = 0.
    cat_2m = 0.
    cat_3m = 0.

    for i in np.arange(len(obs_a)):

       if np.abs(pred_a[i]) < th*std1:  
             cat_1m = cat_1m+1

       if pred_a[i] >= th*std1:  
             cat_2m = cat_2m+1

       if pred_a[i] <= -th*std1:  
             cat_3m = cat_3m+1

       if np.abs(obs_a[i]) < th*std2:  
             cat_1o = cat_1o+1

       if obs_a[i] >= th*std2:  
             cat_2o = cat_2o+1

       if obs_a[i] <= -th*std2:  
             cat_3o = cat_3o+1

    cat_list = [cat_1o, cat_2o, cat_3o, cat_1m, cat_2m, cat_3m]

    return T, C, cat_list

#-----------------------------------------------------------------------
# function to cal. msss - determinstic
#-----------------------------------------------------------------------

# e.g. pred = 1.5, clim = 0.5, obs = 2

def cal_msss(pred, clim, obs):

    pred = np.array(pred)
    clim = np.array(clim)
    obs = np.array(obs)

    score = np.zeros(len(obs))
  
    mss_pred = np.sum((pred-obs)**2)
    mss_clim = np.sum((clim-obs)**2)

    score = 1. - mss_pred/mss_clim
     
    return round(score,2)

#-----------------------------------------------------------------------
# function to cal. bss - probability
#-----------------------------------------------------------------------

# e.g., pred = 82%, clim = 50%, obs = 100% or 0%

def cal_bss(pred, clim, obs):

    pred = np.array(pred)
    clim = np.array(clim)
    obs = np.array(obs)

    for i in np.arange(len(obs)):
       if obs[i]>=1.:
          obs[i] = 1.
       else:
          obs[i] = 0.

    score = np.zeros(len(obs))

    bs_pred = np.sum((pred-obs)**2)
    bs_clim = np.sum((clim-obs)**2)

    score = 1. - bs_pred/bs_clim

    return round(score,2)

###############################################################
# functions - plots
###############################################################

#-----------------------------------------------------------------------
# A colormap for cloud image
#-----------------------------------------------------------------------

def cloud_color_map():

    # This function was developped by Kai-Yuan Cheng
    # source code from Kai-Yuan Cheng

    # get colormap
    ncolors = 256
    colors = plt.cm.binary_r(np.linspace(0.0,1.0,256))

    # change alpha values
    colors[:,-1] = np.linspace(0.0,1.0,ncolors)

    # create a colormap object
    mymap = mcolors.LinearSegmentedColormap.from_list(name='cloud_color_map',colors=colors)

    # register this new colormap with matplotlib
    if 'cloud_color_map' not in mpl.colormaps:
        mpl.colormaps.register(cmap=mymap)

#-----------------------------------------------------------------------
# Radar colormap 
#-----------------------------------------------------------------------

def radar_colormap():
    nws_reflectivity_colors = [
    "#646464", # ND
    "#ccffff", # -30
    "#cc99cc", # -25
    "#996699", # -20
    "#663366", # -15
    "#cccc99", # -10
    "#999966", # -5
    "#646464", # 0
    "#04e9e7", # 5
    "#019ff4", # 10
    "#0300f4", # 15
    "#02fd02", # 20
    "#01c501", # 25
    "#008e00", # 30
    "#fdf802", # 35
    "#e5bc00", # 40
    "#fd9500", # 45
    "#fd0000", # 50
    "#d40000", # 55
    "#bc0000", # 60
    "#f800fd", # 65
    "#9854c6", # 70
    "#fdfdfd" # 75
    ]
    return mpl.colors.ListedColormap(nws_reflectivity_colors)

#-----------------------------------------------------------------------
# function to draw a rectangle 
#-----------------------------------------------------------------------
# https://stackoverflow.com/questions/12251189/how-to-draw-rectangles-on-a-basemap
def draw_screen_poly( lats, lons, m):
    x, y = m( lons, lats )
    xy = zip(x,y)
    poly = Polygon( xy, facecolor='red', alpha=0.5 )
    plt.gca().add_patch(poly)

#-----------------------------------------------------------------------
# function to remove mid value
#-----------------------------------------------------------------------

def remove_mid(a):
    mid=int(len(a)/2)
    b = np.delete(a,mid)
    return b

def delete_mid(lev):
   n = (len(lev)-1)/2
   lev_new = np.delete(lev,n)
   return lev_new

#-----------------------------------------------------------------------
# function to plot histogram
#-----------------------------------------------------------------------

def plot_hist(ax,var,bins):

    ft=12
    ft1=12

    xtick = bins
    xtick_label = bins

    dist,edges = np.histogram(var,bins)

    dist_plot = dist/float(sum(dist))
     
    ax.bar(bins[:-1], dist_plot, width = 0.5)
    #ax.legend(loc='upper right',frameon=False)

    # set x axis
    ax.set_xlim(min(bins),max(bins))
    ax.set_xticks(xtick)
    ax.set_xticklabels(xtick_label,rotation=45)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ft1)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ft1)

    return dist

#-----------------------------------------------------------------------
# function to plot bar
#-----------------------------------------------------------------------

def plt_bar(ax,x,y,color,label,xtick,xtick_label,ytick):

   ft=16
   ft1=16

   ax.bar(x, y, color = color, label = label)
   ax.legend(loc='upper right',frameon=False)

   ax.set_xticks(xtick+0.4)
   ax.set_xticklabels(xtick_label)
   #ax.set_xlim(np.min(xtick)-0.5,np.max(xtick)+0.5)

   ax.set_ylim(np.min(ytick),np.max(ytick))
   ax.set_yticks(ytick)

   for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(ft1)
   for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(ft1)

#-----------------------------------------------------------------------
# function to set up colorbar property
#-----------------------------------------------------------------------

def get_cax(ax):

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)

    return cax

def get_cax2(ax):

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)

    return cax

#-----------------------------------------------------------------------
# function to set up hurricane symbol 
#-----------------------------------------------------------------------

def get_hurricane():
    u = np.array([  [2.444,7.553],
                    [0.513,7.046],
                    [-1.243,5.433],
                    [-2.353,2.975],
                    [-2.578,0.092],
                    [-2.075,-1.795],
                    [-0.336,-2.870],
                    [2.609,-2.016]  ])
    u[:,0] -= 0.098
    codes = [1] + [2]*(len(u)-2) + [2]
    u = np.append(u, -u[::-1], axis=0)
    codes += codes

    return mpath.Path(3*u, codes, closed=False)

#-----------------------------------------------------------------------
# function to create phase plot
#-----------------------------------------------------------------------

def add_phase_labels(ax):

    ft = 22
    col = '0.5'

# add boundaries
    circle_th = np.linspace(0, 2.*np.pi, 100)
    ax.plot(np.cos(circle_th), np.sin(circle_th), '--', color='0.5', linewidth=2)

    circle_th = np.linspace(0, 2.*np.pi, 9)
    for th in circle_th:
        xx = np.array([1,10])*np.cos(th)
        yy = np.array([1,10])*np.sin(th)
        ax.plot(xx, yy, '--', color='0.5', linewidth=2)

# add phase labels

    ax.annotate('Phase 8',
                xy=(0.2, 0.7), xycoords='figure fraction',
                horizontalalignment='right', verticalalignment='bottom',
                fontsize=ft,rotation=90, rotation_mode='anchor', color = col)
    ax.annotate('Phase 1',
                xy=(0.2, 0.3), xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='bottom',
                fontsize=ft,rotation=90, rotation_mode='anchor', color = col)

    ax.annotate('Phase 2',
                xy=(.3, 0.2), xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='center',
                fontsize=ft, color = col)
    ax.annotate('Phase 3',
                xy=(.7, 0.2), xycoords='figure fraction',
                horizontalalignment='right', verticalalignment='center',
                fontsize=ft, color = col)

    ax.annotate('Phase 4',
                xy=(0.8, 0.3), xycoords='figure fraction',
                horizontalalignment='right', verticalalignment='bottom',
                fontsize=ft,rotation=-90, rotation_mode='anchor', color = col)
    ax.annotate('Phase 5',
                xy=(0.8, 0.7), xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='bottom',
                fontsize=ft,rotation=-90, rotation_mode='anchor', color = col)

    ax.annotate('Phase 6',
                xy=(.7, 0.8), xycoords='figure fraction',
                horizontalalignment='right', verticalalignment='center',
                fontsize=ft, color = col)
    ax.annotate('Phase 7',
                xy=(.3, 0.8), xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='center',
                fontsize=ft, color = col)

    return ax

#-----------------------------------------------------------------------
# function to set up basemap
#-----------------------------------------------------------------------

def setup_m(basin, fill=True, fill_col='0.8', ft=16, drawcoast=True):

    if basin == 'Global':
        sx = 0
        ex = 360
        sy = -90
        ey = 90

    if basin == 'Global_60deg':
        sx = 0
        ex = 360
        sy = -60
        ey = 60

    if basin == 'Global_40deg':
        sx = 0
        ex = 360
        sy = -40
        ey = 40

    if basin == 'NH':
        sx = 0
        ex = 360
        sy = 0
        ey = 50

    if basin == 'NH_tc':
        sx = 40
        ex = 340
        sy = 0
        ey = 35

    if basin == 'NH_PA':
        sx = 100
        ex = 360
        sy = 0
        ey = 40

    if basin == 'Global_MJO' or basin == 'MJO_animation':
        sx = 0
        ex = 360
        sy = -30
        ey = 30

    if basin == 'MC':
        sx = 95
        ex = 155
        sy = -15
        ey = 15

    if basin == 'MC2':
        sx = 75 
        ex = 165
        sy = -18
        ey = 18

    if basin == 'MC3':
        sx = 40 
        ex = 180
        sy = -25
        ey = 25

    if basin == 'Global_MJO_15deg':
        sx = 0
        ex = 360
        sy = -15
        ey = 15

    if basin == 'MJO_MC':
        sx = 60 
        ex = 160
        sy = -30
        ey = 30

    if basin == 'GPI_NH':
        sx = 40
        ex = 345
        sy = 0
        ey = 30

    if basin == 'NAtl' or basin == 'NAtl_wnest' or basin == 'NAtl_fill' or basin == 'NAtl_wnest_fill':
        sx = 255
        ex = 345
        sy = 0
        ey = 50

    if basin == 'NA_US':
        sx = 360-120
        ex = 360-40
        sy = 10 
        ey = 55 

    if basin == 'NAtl_track':
        sx = 360-100
        ex = 360-40 
        sy = 8 
        ey = 43

    if basin == 'NAtl_conv_hord':
        sx = 360-62
        ex = 360-42
        sy = 19
        ey = 33

    if basin == 'NAtl_small':
        sx = 260
        ex = 340
        sy = 8 
        ey = 45 

    if basin == 'NAtl_dropsonde':
        sx = 260
        ex = 320
        sy = 5
        ey = 45

    if basin == 'WNAtl':
        sx = 260
        ex = 360-65
        sy = 10
        ey = 50

    if basin == 'WPac':
        sx = 105
        ex = 180
        sy = 0
        ey = 45

    if basin == 'WPac_3km':
        sx = 120
        ex = 140
        sy = 15
        ey = 25 

    if basin == 'EPac':
        sx = 210
        ex = 280
        sy = 0
        ey = 40

    if basin == 'EPac_EOF':
        sx = 230
        ex = 280
        sy = 0
        ey = 25

    if basin == 'NInd':
        sx = 40
        ex = 110
        sy = -5
        ey = 35

    if basin == 'MJO_NAtl':
        sx = 360-140
        ex = 360-10
        sy = 0
        ey = 50

    if basin == 'MJO_EPac':
        sx = 360-130
        ex = 360-70
        sy = 0
        ey = 30

    if basin == 'MJO_EPac2':
        sx = 360-140 
        ex = 360-60
        sy = 0
        ey = 30

    if basin == 'AEW':
        sx = -40 
        ex = 60
        sy = -20
        ey = 40

    if basin == 'NA' or basin == 'NAtl_3nests':
        sx = 360-140 
        ex = 360-0
        sy = -30
        ey = 70

    if basin == 'GoM' or basin == 'GoM_nofill':
        sx = 260
        ex = 290
        sy = 10
        ey = 35 

    if basin == 'GoM_large':
        sx = 245
        ex = 285 
        sy = 10
        ey = 35

    if basin == 'GoM_GPI':
        sx = 360-100
        ex = 360-70
        sy = 10
        ey = 30

    if basin == 'China':
        sx = 90
        ex = 130
        sy = 10
        ey = 50

    if basin == 'US':
        sx = 360-130
        ex = 295
        sy = 25
        ey = 50 

    if basin == 'NJ':
        sx = 282.5 
        ex = 287.5 
        sy = 37.5 
        ey = 41

    if basin == 'US2':
        sx = 360-125
        ex = 290
        sy = 25
        ey = 50

    if basin == 'US3':
        sx = 360-135
        ex = 290
        sy = 5
        ey = 50

    m = Basemap(projection='cyl',llcrnrlon=sx,llcrnrlat=sy,urcrnrlon=ex,urcrnrlat=ey,resolution='l')

    if basin == 'Global' or basin == 'NH' or basin == 'Global_MJO' or basin == 'Global_MJO_15deg' :
       parallels = np.arange(-90.,90+15.,15.)
       meridians = np.arange(20.,360.+40,40.)

    elif basin == 'Global_40deg' :
       parallels = np.arange(-80.,80+20.,20.)
       meridians = np.arange(20.,360.+40,40.)

    elif basin == 'MJO_animation' :
       parallels = np.arange(-45.,45+15.,15.)
       meridians = np.arange(20.,360.+40,40.)

    elif basin == 'NH_tc' :
       parallels = np.arange(-45.,45+15.,15.)
       meridians = np.arange(20.,360.+40,40.)

    elif basin == 'GPI_NH' :
       parallels = np.arange(0.,30+15.,15.)
       meridians = np.arange(20.,360.+40,40.)

    elif basin == 'MJO_EPac':
       parallels = [10,20,30] 
       meridians = [230,  250,  270,  290] 

    elif basin == 'MJO_EPac2':
       parallels = [10,20,30] 
       meridians = np.arange(10.,360.+20.,20.)

    elif basin == 'GoM' :
       parallels = [20, 30, 40] 
       meridians = [260, 270, 280, 290] 

    elif basin == 'GoM_nofill':
       parallels = [20, 30, 40] 
       meridians = [260, 270, 280, 290] 

    elif basin == 'GoM_GPI' :
       parallels = [10, 20,  30]
       meridians = [270,280,290]

    elif basin == 'WPac_3km' or basin == 'US' :
       parallels = np.arange(-90.,90+5.,5.)
       meridians = np.arange(20.,360.+5,5.)

    elif basin == 'NAtl_fill' or basin == 'NAtl_wnest_fill' :
       parallels = np.arange(-90.,90+10.,10.)
       meridians = np.arange(10.,360.+20,20.)

       m.drawparallels(parallels,labels=[1,0,0,0],color='grey',linewidth=0.,fontsize=16)
       m.drawmeridians(meridians,labels=[0,0,0,1],color='grey',linewidth=0.,fontsize=16)
       m.drawmapboundary(fill_color='navy')

    elif basin == 'NAtl_small':
       parallels = np.arange(-90.,90+10.,10.)
       meridians = np.arange(10.,360.+20,20.)

    elif basin == 'AEW':
       parallels = np.arange(-90.,90+10.,10.)
       meridians = np.arange(-180.,180.+10,10.)

    elif basin == 'China':
       parallels = np.arange(-90.,90+5.,5.)
       meridians = np.arange(10.,180.+5,5.)

    elif basin == 'NAtl_conv_hord':
       parallels = [720, 721]
       meridians = [720, 721]

    else:
       parallels = np.arange(-80.,80+10,10.)
       meridians = np.arange(10.,360.+20.,20.)

    m.drawparallels(parallels,labels=[1,0,0,0],color='grey',linewidth=0.,fontsize=ft)
    m.drawmeridians(meridians,labels=[0,0,0,1],color='grey',linewidth=0.,fontsize=ft)

    if basin == 'US':
       m.drawstates()
       m.drawparallels(parallels,labels=[1,0,0,0],color='grey',linewidth=1.,fontsize=ft)
       m.drawmeridians(meridians,labels=[0,0,0,1],color='grey',linewidth=1.,fontsize=ft)

    if fill:
       m.fillcontinents(color=fill_col)

    if drawcoast:
       m.drawcoastlines(color='grey')

    '''
    if basin == 'NAtl_wnest' or basin == 'NAtl_wnest_fill':

	   #file = '/home/kng/plot_grid/grid_spec_for_py/grid_spec.nest02.nc'
        file = '/work/kng/FV3_INPUT_DATA/GRID/C768r10n4_atl_new/grid_spec.nest02.tile7.nc'
        f1 = Dataset(file, 'r')
        lat = f1.variables['grid_lat'][:]
        lon = f1.variables['grid_lon'][:]
        lat1 = lat[1,:] 
        lat2 = lat[:,1]
        lat3 = lat[:,-1]
	    lat4 = lat[-1,:]

	    lon1 = lon[1,:]
	    lon2 = lon[:,1]
	    lon3 = lon[:,-1]
	    lon4 = lon[-1,:]

        col = 'grey'
        linewi = 1.5

	    m.plot(lon1,lat1,color = col, linestyle = '--',linewidth = linewi)
	    m.plot(lon2,lat2,color = col, linestyle = '--',linewidth = linewi)
	    m.plot(lon3,lat3,color = col, linestyle = '--',linewidth = linewi)
	    m.plot(lon4,lat4,color = col, linestyle = '--',linewidth = linewi)

    if basin == 'NAtl_3nests':

      grids = ['C768r10n4_atl_new','C768r10n4_atl_vi', 'C768r10n4_atl_vi_large']
      colors = ['gray', 'k', 'r']
      for grid, col in zip(grids, colors):

        #file = '/work/kng/FV3_INPUT_DATA/GRID/' + grid + '/grid_spec.nest02.tile7.nc'
        file1 = '/work/kng/FV3_INPUT_DATA/GRID/' + grid + '/C768_grid.tile7.nc'
        file2 = '/lustre/f2/dev/gfdl/Kun.Gao/SHiELD_IC_v16/' + grid + '/INPUT/C768_grid.tile7.nc'

        try:
          f1 = Dataset(file1, 'r')
        except:
          f1 = Dataset(file2, 'r')

        lat = f1.variables['y'][:]
        lon = f1.variables['x'][:]

        lat1 = lat[1,:]
        lat2 = lat[:,1]
        lat3 = lat[:,-1]
        lat4 = lat[-1,:]

        lon1 = lon[1,:]
        lon2 = lon[:,1]
        lon3 = lon[:,-1]
        lon4 = lon[-1,:]

        #col = 'grey'
        linewi = 1.5

        m.plot(lon1,lat1,color = col, linestyle = '--',linewidth = linewi)
        m.plot(lon2,lat2,color = col, linestyle = '--',linewidth = linewi)
        m.plot(lon3,lat3,color = col, linestyle = '--',linewidth = linewi)
        m.plot(lon4,lat4,color = col, linestyle = '--',linewidth = linewi)
    '''

    return m

#-----------------------------------------------------------------------
# function to make contour plots - pln
#-----------------------------------------------------------------------

def make_contourfs_3p(basin,lon,lat,var_plot1,var_plot2,var_plot3,  \
                      level1,level2,level3,levelc1,levelc2,levelc3, \
                      cmap1,cmap2,cmap3, \
                      title1,title2,title3):

   lonm,latm = np.meshgrid(lon,lat)

   cmin1 = np.min(level1)
   cmin2 = np.min(level2)
   cmin3 = np.min(level3)
   cmax1 = np.max(level1)
   cmax2 = np.max(level2)
   cmax3 = np.max(level3)

#--------------------
   ax1 = plt.subplot(311)
   m1 = setup_m(basin)
   m1.contourf(lonm, latm, var_plot1, cmap=cmap1,levels=level1,extend="both")
   cax = get_cax(ax1)
   plt.colorbar(cax = cax, ticks=levelc1)
   plt.clim([cmin1,cmax1])

#--------------------
   ax2 = plt.subplot(312)
   m2 = setup_m(basin)
   m2.contourf(lonm, latm, var_plot2, cmap=cmap2,levels=level2,extend="both")
   cax = get_cax(ax2)
   plt.colorbar(cax = cax, ticks=levelc2)
   plt.clim([cmin2,cmax2])

#--------------------
   ax3 = plt.subplot(313)
   m3 = setup_m(basin)
   m3.contourf(lonm, latm, var_plot3, cmap=cmap3,levels=level3,extend="both")
   cax = get_cax(ax3)
   plt.colorbar(cax = cax, ticks=levelc3)
   plt.clim([cmin3,cmax3])

#--------------------
   ft=16
   ax1.set_title(title1,fontsize=ft)
   ax2.set_title(title2,fontsize=ft)
   ax3.set_title(title3,fontsize=ft)


#-----------------------------------------------------------------------
# function to make contour plots - sec
#-----------------------------------------------------------------------

def make_contourfs_sec_3p(x1d,z1d,var_plot1,var_plot2,var_plot3,  \
                      level1,level2,level3,levelc1,levelc2,levelc3, \
                      title1,title2,title3,xtick,ytick,xlabel,ylabel,cmap=cm.bwr):

   xm,zm = np.meshgrid(x1d,z1d)

   cmin1 = np.min(level1)
   cmin2 = np.min(level2)
   cmin3 = np.min(level3)
   cmax1 = np.max(level1)
   cmax2 = np.max(level2)
   cmax3 = np.max(level3)

   xmin = np.min(xtick)
   xmax = np.max(xtick)

   ymin = np.min(ytick)
   ymax = np.max(ytick)

#--------------------
   ax1 = plt.subplot(311)
   c1 = plt.contourf(xm, zm, var_plot1, cmap=cmap,levels=level1,extend="both")

   ax1.set_xlim(xmin,xmax)
   ax1.set_xticks(xtick)
   ax1.set_ylim(ymin,ymax)
   ax1.set_yticks(ytick)   

   plt.gca().invert_yaxis()

   cax = get_cax(ax1)
   plt.colorbar(c1, cax = cax, ticks=levelc1)
   plt.clim([cmin1,cmax1])

#--------------------
   ax2 = plt.subplot(312)
   c2 = plt.contourf(xm, zm, var_plot2, cmap=cmap,levels=level2,extend="both")

   ax2.set_xlim(xmin,xmax)
   ax2.set_xticks(xtick)
   ax2.set_ylim(ymin,ymax)
   ax2.set_yticks(ytick)  

   plt.gca().invert_yaxis()

   cax = get_cax(ax2)
   plt.colorbar(c2, cax = cax, ticks=levelc2)
   plt.clim([cmin2,cmax2])

#--------------------
   ax3 = plt.subplot(313)
   c3 = plt.contourf(xm, zm, var_plot3, cmap=cmap,levels=level3,extend="both")

   ax3.set_xlim(xmin,xmax)
   ax3.set_xticks(xtick)
   ax3.set_ylim(ymin,ymax)
   ax3.set_yticks(ytick) 
   plt.gca().invert_yaxis()

   cax = get_cax(ax3)
   plt.colorbar(c3, cax = cax, ticks=levelc3)
   plt.clim([cmin3,cmax3])

#--------------------
   ft=16
   ax1.set_title(title1,fontsize=ft)
   ax2.set_title(title2,fontsize=ft)
   ax3.set_title(title3,fontsize=ft)

   #ax1.set_xlabel(xlabel,fontsize=ft)
   #ax2.set_xlabel(xlabel,fontsize=ft)
   ax3.set_xlabel(xlabel,fontsize=ft)

   ax1.set_ylabel(ylabel,fontsize=ft)
   ax2.set_ylabel(ylabel,fontsize=ft)
   ax3.set_ylabel(ylabel,fontsize=ft)

########################################################################
# functions - TC analysis 
########################################################################

#-----------------------------------------------------------------------
# function to get Cd from z0
#-----------------------------------------------------------------------

def get_cd(znot, zm):

   cd=0.4**2/(np.log(zm/znot))**2

   return cd

#-----------------------------------------------------------------------
# function to cal RMW in km 
#-----------------------------------------------------------------------
def cal_rmw1(lat1, rmw1):

   dist = cal_dist_2p(lat1, 100, lat1, 101)

   return rmw1*dist

def cal_rmw(tc_lat_all, tc_rmw_all):

   ntc = len(tc_lat_all)

   tc_rmw_all1 = []

   for tc in np.arange(ntc):

       lat1 = tc_lat_all[tc]
       rmw1 = tc_rmw_all[tc]

       dist = cal_dist_2p(lat1, 100, lat1, 101)
       tc_rmw_all1.append(rmw1*dist)

   return tc_rmw_all1

#-----------------------------------------------------------------------
# function to trim TC records in nested grid 
#-----------------------------------------------------------------------

def trim_for_nest(tc_date, tc_lon, tc_lat, tc_pres, tc_wind, distance, nest_edges):

    tc_rec_flag = np.arange(len(tc_lon))

    tc_date_new1 = []
    tc_lon_new1 = []
    tc_lat_new1 = []
    tc_pres_new1 = []
    tc_wind_new1 = []
    tc_rec_flag_new1 = []

    #nest_lat1 = 10. 
    #nest_lat2 = 37.
    #nest_lon1 = 265.
    #nest_lon2 = 335.

    #lat1 = nest_lat1 + distance
    #lat2 = nest_lat2 - distance
    #lon1 = nest_lon1 + distance
    #lon2 = nest_lon2 - distance

    for i in np.arange(len(tc_lon)):

       all_dist2 = (nest_edges[0,:] - tc_lon[i])**2 + (nest_edges[1,:] - tc_lat[i])**2 
       all_dist = np.sqrt(all_dist2)
       min_dist = np.min(all_dist)
 
       #if tc_lon[i] >= lon1 and tc_lon[i] <= lon2 and \
       #   tc_lat[i] >= lat1 and tc_lat[i] <= lat2:
       
       if min_dist >= distance:
   
          tc_date_new1 += [tc_date[i]]
          tc_lon_new1  += [tc_lon[i]]
          tc_lat_new1  += [tc_lat[i]]
          tc_pres_new1 += [tc_pres[i]]
          tc_wind_new1 += [tc_wind[i]]
          tc_rec_flag_new1 += [tc_rec_flag[i]]
  
    newnn = 0 
    for i in np.arange(len(tc_rec_flag_new1)-1):
       if (tc_rec_flag_new1[i+1] - tc_rec_flag_new1[i]) > 1:
          newnn = i
          break
       else:
          newnn = 0

    if newnn != 0: 

       tc_date_new = tc_date_new1[0:newnn+1]
       tc_lon_new  = tc_lon_new1[0:newnn+1]
       tc_lat_new  = tc_lat_new1[0:newnn+1]
       tc_pres_new = tc_pres_new1[0:newnn+1]
       tc_wind_new = tc_wind_new1[0:newnn+1]

    else:

       tc_date_new = tc_date_new1
       tc_lon_new  = tc_lon_new1
       tc_lat_new  = tc_lat_new1
       tc_pres_new = tc_pres_new1
       tc_wind_new = tc_wind_new1

    return tc_date_new, tc_lon_new, tc_lat_new, tc_pres_new, tc_wind_new 

#-----------------------------------------------------------------------
# function to trim TC records in periodic domain 
#-----------------------------------------------------------------------

def trim_for_aqua(tc_date, tc_lon, tc_lat, tc_pres, tc_wind, distance):

    tc_rec_flag = np.arange(len(tc_lon))

    tc_date_new1 = []
    tc_lon_new1 = []
    tc_lat_new1 = []
    tc_pres_new1 = []
    tc_wind_new1 = []
    tc_rec_flag_new1 = []

    nest_lat1 = -80. 
    nest_lat2 = 80.
    nest_lon1 = 20.
    nest_lon2 = 340.

    lat1 = nest_lat1 + distance
    lat2 = nest_lat2 - distance
    lon1 = nest_lon1 + distance
    lon2 = nest_lon2 - distance

    for i in np.arange(len(tc_lon)):

       if tc_lon[i] >= lon1 and tc_lon[i] <= lon2 and \
          tc_lat[i] >= lat1 and tc_lat[i] <= lat2:
          
          tc_date_new1 += [tc_date[i]]
          tc_lon_new1  += [tc_lon[i]]
          tc_lat_new1  += [tc_lat[i]]
          tc_pres_new1 += [tc_pres[i]]
          tc_wind_new1 += [tc_wind[i]]
          tc_rec_flag_new1 += [tc_rec_flag[i]]
  
    newnn = 0 
    for i in np.arange(len(tc_rec_flag_new1)-1):
       if (tc_rec_flag_new1[i+1] - tc_rec_flag_new1[i]) > 1:
          newnn = i
          break
       else:
          newnn = 0

    if newnn != 0: 

       tc_date_new = tc_date_new1[0:newnn+1]
       tc_lon_new  = tc_lon_new1[0:newnn+1]
       tc_lat_new  = tc_lat_new1[0:newnn+1]
       tc_pres_new = tc_pres_new1[0:newnn+1]
       tc_wind_new = tc_wind_new1[0:newnn+1]

    else:

       tc_date_new = tc_date_new1
       tc_lon_new  = tc_lon_new1
       tc_lat_new  = tc_lat_new1
       tc_pres_new = tc_pres_new1
       tc_wind_new = tc_wind_new1

    return tc_date_new, tc_lon_new, tc_lat_new, tc_pres_new, tc_wind_new 


#-----------------------------------------------------------------------
# function to trim wind records
#-----------------------------------------------------------------------

def trim_wind(all_wind, wind_min):

  tc_str = [ n for n,i in enumerate(all_wind) if i>wind_min ][0]
  tc_end = np.where(all_wind == np.max(all_wind))
  tc_end = np.squeeze(tc_end)
  if np.size(tc_end) > 1:
     tc_end = tc_end[0]

  tc_wind = all_wind[tc_str:tc_end]

  return tc_wind

#-----------------------------------------------------------------------
# simple tracker
#-----------------------------------------------------------------------

def simple_tracker(lon, lat, ws10m, slp, dr=1./32):

        try:
          NT, nx, ny = np.shape(ws10m)
        except:
          nx, ny = np.shape(ws10m)
          ws10m_new = np.zeros((1,nx,ny))
          slp_new = np.zeros((1,nx,ny))
          ws10m_new[0,:,:] = ws10m
          slp_new[0,:,:] = slp
          ws10m = ws10m_new
          slp = slp_new
          NT = 1

        vmax_all = []
        pmin_all = []
        lon_pmin_all = []
        lat_pmin_all = []

        mid = int((nx-1)/2)
        scope = int(5/dr)

        for i in np.arange(NT):

            slp1 = slp[i, :, :] #mid-scope:mid+scope,mid-scope:mid+scope]
            lon1 = lon #[mid-scope:mid+scope,mid-scope:mid+scope]
            lat1 = lat #[mid-scope:mid+scope,mid-scope:mid+scope]

            #if dr <= 0.01: # 1km
            #   slp1 = smooth_2d(slp1, 1)

            ws10m1 = ws10m[i,:,:]

            vmax1 = np.max(ws10m1)
            pmin1 = np.min(slp1)

            lon_pmin = lon1[np.where(slp1==pmin1)]
            lat_pmin = lat1[np.where(slp1==pmin1)]
           
            #print lon_pmin

            vmax_all.append(vmax1)
            pmin_all.append(pmin1)
            lon_pmin_all.append(lon_pmin[0])
            lat_pmin_all.append(lat_pmin[0])

        return np.array(vmax_all), np.array(pmin_all), np.array(lon_pmin_all), np.array(lat_pmin_all)

#-----------------------------------------------------------------------
# simple tracker - opt 2 (find max wind location and then pmin location)
#-----------------------------------------------------------------------

def simple_tracker_opt2(lon, lat, ws10m, slp, dr=1./32):

        try:
          NT, nx, ny = np.shape(ws10m)
        except:
          nx, ny = np.shape(ws10m)
          ws10m_new = np.zeros((1,nx,ny))
          slp_new = np.zeros((1,nx,ny))
          ws10m_new[0,:,:] = ws10m
          slp_new[0,:,:] = slp
          ws10m = ws10m_new
          slp = slp_new
          NT = 1

        vmax_all = []
        pmin_all = []
        lon_pmin_all = []
        lat_pmin_all = []

        for t in np.arange(NT):

            ws10m1 = ws10m[t,:,:]

            vmax1 = np.max(ws10m1)

            _i, _j = np.where(ws10m1==vmax1)
            scope = int(1./dr)
            i, j = _i[0],  _j[0]

            slp1 = slp[t, i-scope:i+scope, j-scope:j+scope]
            lon1 = lon[i-scope:i+scope, j-scope:j+scope]
            lat1 = lat[i-scope:i+scope, j-scope:j+scope]

            pmin1 = np.min(slp1)

            lon_pmin = lon1[np.where(slp1==pmin1)]
            lat_pmin = lat1[np.where(slp1==pmin1)]

            vmax_all.append(vmax1)
            pmin_all.append(pmin1)
            lon_pmin_all.append(lon_pmin[0])
            lat_pmin_all.append(lat_pmin[0])

        return np.array(vmax_all), np.array(pmin_all), np.array(lon_pmin_all), np.array(lat_pmin_all)

def simplest_tracker(ws10m, slp):

        NT, nx, ny = np.shape(ws10m)
        vmax_all = []
        pmin_all = []

        for i in np.arange(NT):

            ws10m1 = ws10m[i,:,:]
            slp1 = slp[i,:,:]

            vmax1 = np.max(ws10m1)
            pmin1 = np.min(slp1)

            vmax_all.append(vmax1)
            pmin_all.append(pmin1)

        return np.array(vmax_all), np.array(pmin_all)

#-----------------------------------------------------------------------
# function to perform azimuthal average 
#-----------------------------------------------------------------------

# --- 1var

def azi_ave_1var(xm,ym,var1,radius,bin_width):

    dist = (xm**2+ym**2)**0.5

    var_bin = np.zeros(len(radius))

    for ri in np.arange(len(radius)):

       mask = abs(dist-radius[ri]) <= bin_width
       var_masked = np.ma.MaskedArray(var1,mask=~mask)

       var_bin[ri] = np.nanmean(var_masked)

    return var_bin

#--- wind

def azi_ave_wind(xm,ym,u1,v1,radius,bin_width):

    dist = (xm**2+ym**2)**0.5
    vel1  = (u1**2+v1**2)**0.5

    xc, yc = np.where(dist == dist.min())
    dist[xc,yc] = 0.00001 # set center distance to a nonzero value

## (x,y) to (r,t)

    sina = ym/dist
    cosa = xm/dist

    vr1 =  u1*cosa+v1*sina
    vt1 = -u1*sina+v1*cosa

## bin

    vr_bin = np.zeros(len(radius))
    vt_bin = np.zeros(len(radius))
    vel_bin = np.zeros(len(radius))

    for ri in np.arange(len(radius)):

       mask = abs(dist-radius[ri]) <= bin_width
       vr_masked = np.ma.MaskedArray(vr1,mask=~mask)
       vt_masked = np.ma.MaskedArray(vt1,mask=~mask)
       vel_masked = np.ma.MaskedArray(vel1,mask=~mask)

       #print vr_masked
       vr_bin[ri] = np.nanmean(vr_masked)
       vt_bin[ri] = np.nanmean(vt_masked)
       vel_bin[ri] = np.nanmean(vel_masked)

    return vr_bin, vt_bin, vel_bin

#-----------------------------------------------------------------------
# function to read TC dropsonde data 
#-----------------------------------------------------------------------

def read_tc_dropsonde(tc_files, var_name):

  lon_ebt_all  = []
  lat_ebt_all  = []
  wind_ebt_all = []
  pres_ebt_all = []
  rmw_ebt_all  = []

  lon_drop_all  = []
  lat_drop_all  = []
  rad_drop_all = []
  azi_drop_all = []

  var_all = []

  for tc_file in tc_files:

    f1 = Dataset(tc_file, 'r')

    lon_ebt  = f1.variables['lon_ebt'][:]
    lat_ebt  = f1.variables['lat_ebt'][:]
    pres_ebt = f1.variables['pres_ebt'][:]
    wind_ebt = f1.variables['wind_ebt'][:]
    rmw_ebt  = f1.variables['rmw_ebt'][:]

    lon_drop  = f1.variables['lon_drop'][:]
    lat_drop  = f1.variables['lat_drop'][:]
    rad_drop = f1.variables['rad_drop'][:]
    azi_drop = f1.variables['azi_drop'][:]

    var  = f1.variables[var_name][:]

    lon_ebt_all.append(lon_ebt)
    lat_ebt_all.append(lat_ebt)
    wind_ebt_all.append(wind_ebt)
    pres_ebt_all.append(pres_ebt)
    rmw_ebt_all.append(rmw_ebt)
    lon_drop_all.append(lon_drop)
    lat_drop_all.append(lat_drop)
    rad_drop_all.append(rad_drop)
    azi_drop_all.append(azi_drop)
    var_all.append(var)

  return lon_ebt_all, lat_ebt_all, wind_ebt_all, pres_ebt_all, rmw_ebt_all, \
         lon_drop_all, lat_drop_all, rad_drop_all, azi_drop_all, var_all


#-----------------------------------------------------------------------
# function to read 2d TC field 
#-----------------------------------------------------------------------

def read_tc_pln(tc_files, var_name, pre_peak=False):

  tc_id_all   = []
  tc_lon_all  = []
  tc_lat_all  = []
  tc_pres_all = []
  tc_wind_all = []
  tc_rmw_all  = []

  latm_all  = []
  lonm_all  = []

  var_all = []

  tc_info_all = []

  tc_id = 0

  for tc_file in tc_files:

    tc_id = tc_id +1

    f1 = Dataset(tc_file, 'r')

    tc_lon  = f1.variables['tc_lon'][:]
    tc_lat  = f1.variables['tc_lat'][:]
    tc_pres = f1.variables['tc_pres'][:]
    tc_wind = f1.variables['tc_wind'][:]
    tc_rmw  = f1.variables['tc_rmw'][:]

    lonm  = f1.variables['lonm'][:]
    latm  = f1.variables['latm'][:]
    var  = f1.variables[var_name][:]

    nrec = len(tc_lon)

    if pre_peak:
       nrec = find_nearest(tc_wind,np.max(tc_wind))
       nrec = nrec +1

    for i in np.arange(nrec):

      #if tc_rmw[i] > 0.001:

        tc_id_all.append(tc_id)
        tc_lon_all.append(tc_lon[i])
        tc_lat_all.append(tc_lat[i])
        tc_pres_all.append(tc_pres[i])
        tc_wind_all.append(tc_wind[i])
        tc_rmw_all.append(tc_rmw[i])
      
        lonm_all.append(lonm[i,:,:])
        latm_all.append(latm[i,:,:])
        var_all.append(var[i,:,:])

  tc_info_all.append(tc_id_all)
  tc_info_all.append(tc_lon_all)
  tc_info_all.append(tc_lat_all)
  tc_info_all.append(tc_pres_all)
  tc_info_all.append(tc_wind_all)
  tc_info_all.append(tc_rmw_all)
   
  return tc_info_all, lonm_all, latm_all, var_all

# no tc_rmw
def read_tc_pln_2(tc_files, var_name, pre_peak=False):

  tc_id_all   = []
  tc_lon_all  = []
  tc_lat_all  = []
  tc_pres_all = []
  tc_wind_all = []
  #tc_rmw_all  = []

  latm_all  = []
  lonm_all  = []

  var_all = []

  tc_info_all = []

  tc_id = 0

  for tc_file in tc_files:

    tc_id = tc_id +1

    f1 = Dataset(tc_file, 'r')

    tc_lon  = f1.variables['tc_lon'][:]
    tc_lat  = f1.variables['tc_lat'][:]
    tc_pres = f1.variables['tc_pres'][:]
    tc_wind = f1.variables['tc_wind'][:]
    #tc_rmw  = f1.variables['tc_rmw'][:]

    lonm  = f1.variables['lonm'][:]
    latm  = f1.variables['latm'][:]
    var  = f1.variables[var_name][:]

    nrec = len(tc_lon)

    if pre_peak:
       nrec = find_nearest(tc_wind,np.max(tc_wind))
       nrec = nrec +1

    for i in np.arange(nrec):

      #if tc_rmw[i] > 0.001:

        tc_id_all.append(tc_id)
        tc_lon_all.append(tc_lon[i])
        tc_lat_all.append(tc_lat[i])
        tc_pres_all.append(tc_pres[i])
        tc_wind_all.append(tc_wind[i])
        #tc_rmw_all.append(tc_rmw[i])
      
        lonm_all.append(lonm[i,:,:])
        latm_all.append(latm[i,:,:])
        var_all.append(var[i,:,:])

  tc_info_all.append(tc_id_all)
  tc_info_all.append(tc_lon_all)
  tc_info_all.append(tc_lat_all)
  tc_info_all.append(tc_pres_all)
  tc_info_all.append(tc_wind_all)
  #tc_info_all.append(tc_rmw_all)
   
  return tc_info_all, lonm_all, latm_all, var_all

#-----------------------------------------------------------------------
# function to read azimuthal averaged TC 
#-----------------------------------------------------------------------

def read_tc_azi(tc_files, var_name, pre_peak=False, skip=0):
  tc_id_all   = []
  tc_lon_all  = []
  tc_lat_all  = []
  tc_pres_all = []
  tc_wind_all = []
  tc_rmw_all  = []
  rstar_all  = []
  var_all = []
  tc_info_all = []
  tc_id = 0
  for tc_file in tc_files:
    #print tc_file
    tc_id = tc_id +1
    f1 = Dataset(tc_file, 'r')
    tc_lon  = f1.variables['tc_lon'][:]
    tc_lat  = f1.variables['tc_lat'][:]
    tc_pres = f1.variables['tc_pres'][:]
    tc_wind = f1.variables['tc_wind'][:]
    tc_rmw  = f1.variables['tc_rmw'][:]
    rstar  = f1.variables['rstar'][:]
    var  = f1.variables[var_name][:]
    nrec = len(tc_rmw)
    if pre_peak:
       nrec = find_nearest(tc_wind,np.max(tc_wind))
       nrec = nrec +1
    for i in np.arange(skip,nrec,1):
      #if tc_rmw[i] > 0.001:
        #print i
        tc_id_all.append(tc_id)
        tc_lon_all.append(tc_lon[i])
        tc_lat_all.append(tc_lat[i])
        tc_pres_all.append(tc_pres[i])
        tc_wind_all.append(tc_wind[i])
        tc_rmw_all.append(tc_rmw[i])
        rstar_all.append(rstar[i,:])
        var_all.append(var[i,:])
  tc_info_all.append(tc_id_all)
  tc_info_all.append(tc_lon_all)
  tc_info_all.append(tc_lat_all)
  tc_info_all.append(tc_pres_all)
  tc_info_all.append(tc_wind_all)
  tc_info_all.append(tc_rmw_all)
   
  return tc_info_all, rstar_all, var_all


#-----------------------------------------------------------------------
# function to read azimuthal averaged TC
#-----------------------------------------------------------------------

def read_tc_azi2(tc_files, var_name, pre_peak=False, skip=0):
  tc_id_all   = []
  tc_lon_all  = []
  tc_lat_all  = []
  tc_pres_all = []
  tc_wind_all = []
  tc_rmw_all  = []
  tc_time_all = []
  rstar_all  = []
  var_all = []
  tc_info_all = []
  tc_id = 0
  for tc_file in tc_files:
    tc_id = tc_id +1
    f1 = Dataset(tc_file, 'r')
    tc_lon  = f1.variables['tc_lon'][:]
    tc_lat  = f1.variables['tc_lat'][:]
    tc_pres = f1.variables['tc_pres'][:]
    tc_wind = f1.variables['tc_wind'][:]
    tc_rmw  = f1.variables['tc_rmw'][:]
    tc_time = f1.variables['tc_time'][:]
    rstar  = f1.variables['rstar'][:]
    var  = f1.variables[var_name][:]
    nrec = len(tc_rmw)
    if pre_peak:
       nrec = find_nearest(tc_wind,np.max(tc_wind))
       nrec = nrec +1
    for i in np.arange(skip,nrec,1):
      #if tc_rmw[i] > 0.001:
        #print i
        tc_id_all.append(tc_id)
        tc_lon_all.append(tc_lon[i])
        tc_lat_all.append(tc_lat[i])
        tc_pres_all.append(tc_pres[i])
        tc_wind_all.append(tc_wind[i])
        tc_rmw_all.append(tc_rmw[i])
        tc_time_all.append(tc_time[i])
        rstar_all.append(rstar[i,:])
        var_all.append(var[i,:])
  tc_info_all.append(tc_id_all)
  tc_info_all.append(tc_lon_all)
  tc_info_all.append(tc_lat_all)
  tc_info_all.append(tc_pres_all)
  tc_info_all.append(tc_wind_all)
  tc_info_all.append(tc_rmw_all)
  tc_info_all.append(tc_time_all)

  return tc_info_all, rstar_all, var_all

#-----------------------------------------------------------------------
# function to remap z-r TC section to uniform radius 
#-----------------------------------------------------------------------

def interp_sec(rstar_ave,rstar,var_2d):

    nz,nr1 = np.shape(var_2d)
    nr = len(rstar_ave)

    var_2d_new = np.zeros((nz,nr))

    for k in np.arange(nz):

         var = np.squeeze(var_2d[k,:])
         var_2d_new[k,:] = np.interp(rstar_ave,rstar,var)

    return var_2d_new

#-----------------------------------------------------------------------
# function to determine if a storm is over land 
#-----------------------------------------------------------------------

def over_land(lons, lats):

    maskres = 0.25 # degrees
    landmaskfile = '/work/lmh/research/seasonal/quick_tracks/land_mask_c384_1440x720.dat'

    lx  = np.arange(  0.+maskres*0.5, 360.+maskres, maskres)
    ly  = np.arange(-90.+maskres*0.5, 90.+maskres , maskres)
    nx_grid = lx.size
    ny_grid = ly.size
    lon_grid, lat_grid = np.meshgrid(lx,ly)

    land_mask = np.zeros(np.array(lon_grid.shape))
    land_mask[:-1,:-1] = np.loadtxt(landmaskfile).reshape(np.array(lon_grid.shape) - 1) > 0.5
    land_mask[-1,:] = land_mask[0,:]
    land_mask[:,-1] = land_mask[:,0]

    iis = map(int, np.floor(lons/maskres).tolist())
    jjs = map(int, np.floor((lats + 90.)/maskres).tolist())

    return land_mask[jjs,iis]

def if_landfall_1tc(basin, lon1, lat1, wind1, land_mask, maskres):

    tsn = 0
    hn = 0
    tse = 0.
    he = 0.

# define parameters
    wind_th1 = 17.5 # min wind speed when making landfall 
    wind_th2 = 32.5

# define landfall area 
    lat_cri1 = 10
    lat_cri2 = 50
    if basin == 'NAtl':
       lon_cri1 = 360-100
       lon_cri2 = 360-65 # 60W

    elif basin == 'GoM':
       lon_cri1 = 360-100
       lon_cri2 = 360-70 

    elif basin == 'EPac':
       lon_cri1 = 360-140
       lon_cri2 = 360-70 

    elif basin == 'WPac':
       lon_cri1 = 100
       lon_cri2 = 150 

# start looking for TCs making landfall
    for a in np.arange(1):

       iis = map(int, np.floor(np.array(lon1)/maskres).tolist())
       jjs = map(int, np.floor((np.array(lat1) + 90.)/maskres).tolist())
       is_land = land_mask[jjs,iis]

       # get TS/Hur landfall number
       ts_landfall = 0
       h_landfall = 0

       for j in np.arange(len(is_land)-1):
          if is_land[j+1] > 0.5 and is_land[j] < 0.5 and \
             lon1[j]>lon_cri1 and lon1[j]<lon_cri2 and lat1[j]>lat_cri1 and lat1[j]<lat_cri2:

             if wind1[j] > wind_th1: 
                ts_landfall = ts_landfall+1

             if wind1[j] > wind_th2: 
                h_landfall = h_landfall+1

       if ts_landfall > 0:
          tsn = tsn + 1

       if h_landfall > 0:
          hn = hn + 1
             
       # get TS/Hur landfall tce

       wind1 = np.array(wind1)
       wind1[ wind1 < wind_th1 ] = 0
       wind1_land = np.array(wind1)*is_land
       wind1_land_kt = wind1_land * 1.94384
       tse = tse + np.sum(wind1_land_kt**2) 

       wind1[ wind1 < wind_th2 ] = 0
       wind1_land = np.array(wind1)*is_land
       wind1_land_kt = wind1_land * 1.94384
       he = he + np.sum(wind1_land_kt**2) 

    return tsn, hn, tse, he

# ts, h, mh, c45h

def if_landfall_1tc_more(basin, lon1, lat1, wind1, land_mask, maskres, hur_min, mhur_min, c45hur_min):

    tsn = 0
    hn = 0
    mhn = 0
    c45hn = 0

    tse = 0.
    he = 0.
    mhe = 0.
    c45he = 0.

# define parameters
    wind_th1 = 17.5 # min wind speed when making landfall 
    wind_th2 = hur_min
    wind_th3 = mhur_min
    wind_th4 = c45hur_min

# define landfall area 
    lat_cri1 = 10
    lat_cri2 = 50
    if basin == 'NAtl':
       lon_cri1 = 360-100
       lon_cri2 = 360-65 # 60W

    elif basin == 'GoM':
       lon_cri1 = 360-100
       lon_cri2 = 360-70 

    elif basin == 'EPac':
       lon_cri1 = 360-140
       lon_cri2 = 360-70 

    elif basin == 'WPac':
       lon_cri1 = 100
       lon_cri2 = 150 

# start looking for TCs making landfall
    for a in np.arange(1):

       iis = map(int, np.floor(np.array(lon1)/maskres).tolist())
       jjs = map(int, np.floor((np.array(lat1) + 90.)/maskres).tolist())
       is_land = land_mask[jjs,iis]

       # get landfall number
       ts_landfall = 0
       h_landfall = 0
       mh_landfall = 0
       c45h_landfall = 0

       for j in np.arange(len(is_land)-1):
          if is_land[j+1] > 0.5 and is_land[j] < 0.5 and \
             lon1[j]>lon_cri1 and lon1[j]<lon_cri2 and lat1[j]>lat_cri1 and lat1[j]<lat_cri2:

             if wind1[j] > wind_th1: 
                ts_landfall = ts_landfall+1

             if wind1[j] > wind_th2: 
                h_landfall = h_landfall+1

             if wind1[j] > wind_th3: 
                mh_landfall = mh_landfall+1

             if wind1[j] > wind_th4: 
                c45h_landfall = c45h_landfall+1

       if ts_landfall > 0:
          tsn = tsn + 1

       if h_landfall > 0:
          hn = hn + 1

       if mh_landfall > 0:
          mhn = mhn + 1

       if c45h_landfall > 0:
          c45hn = c45hn + 1
             
       # get landfall tce

       wind1 = np.array(wind1)

       wind1[ wind1 < wind_th1 ] = 0
       wind1_land = np.array(wind1)*is_land
       wind1_land_kt = wind1_land * 1.94384
       tse = tse + np.sum(wind1_land_kt**2) 

       wind1[ wind1 < wind_th2 ] = 0
       wind1_land = np.array(wind1)*is_land
       wind1_land_kt = wind1_land * 1.94384
       he = he + np.sum(wind1_land_kt**2) 

       wind1[ wind1 < wind_th3 ] = 0
       wind1_land = np.array(wind1)*is_land
       wind1_land_kt = wind1_land * 1.94384
       mhe = mhe + np.sum(wind1_land_kt**2) 

       wind1[ wind1 < wind_th4 ] = 0
       wind1_land = np.array(wind1)*is_land
       wind1_land_kt = wind1_land * 1.94384
       c45he = c45he + np.sum(wind1_land_kt**2) 

    return tsn, hn, mhn, c45hn, tse, he, mhe, c45he

#-----------------------------------------------------------------------
# function to find genesis record
#-----------------------------------------------------------------------

def find_first_record(lon_all,lat_all,wind_all,TCmin):

   lon_1st=[]
   lat_1st=[]

   for i in np.arange(len(lon_all)):
      all_lon  = lon_all[i]
      all_lat  = lat_all[i]
      all_wind = wind_all[i]

      tind = [ n for n,i in enumerate(all_wind) if i>TCmin ][0] 
      lon_1st += [all_lon[tind]]
      lat_1st += [all_lat[tind]]

   return lon_1st, lat_1st

def find_first_record_2(all_wind,TCmin):

   tind = [ n for n,i in enumerate(all_wind) if i>TCmin ][0] 

   return tind

#-----------------------------------------------------------------------
# function to calculate tc days/duration
#-----------------------------------------------------------------------

def cal_tc_day1(all_tc_wind, all_tc_date, hur_min, mhur_min, c45hur_min):

    # number of days that has hur or mhur (overlapping days removed)
    ts_day  = 0
    h_day   = 0
    mh_day  = 0
    c45h_day= 0

    ntc = len(all_tc_wind)

    ts_date_rec = []
    h_date_rec = []
    mh_date_rec = []
    c45h_date_rec = []

    if ntc > 0:
    
      for tc in np.arange(ntc):
          wind = np.array(all_tc_wind[tc])
          date = all_tc_date[tc]

          for i in np.arange(len(wind)):
              if wind[i] >= 17.5:
                 if date[i] not in ts_date_rec:
                    ts_date_rec.append(date[i])

          for i in np.arange(len(wind)):
              if wind[i] >= hur_min:
                 if date[i] not in h_date_rec:
                    h_date_rec.append(date[i])

              if wind[i] >= mhur_min:
                 if date[i] not in mh_date_rec:
                    mh_date_rec.append(date[i])
  
              if wind[i] >= c45hur_min:
                 if date[i] not in c45h_date_rec:
                    c45h_date_rec.append(date[i])

    ts_day = float(len(ts_date_rec))/4.
    h_day = float(len(h_date_rec))/4.
    mh_day = float(len(mh_date_rec))/4.
    c45h_day = float(len(c45h_date_rec))/4.

    return ts_day, h_day, mh_day, c45h_day


def cal_tc_day2(all_tc_wind, all_tc_date, hur_min, mhur_min, c45hur_min):

    # number of records that maintains ts, hur, mhur, c45hur intensity
    ts_hour  = 0
    h_hour   = 0
    mh_hour  = 0
    c45h_hour= 0

    ntc = len(all_tc_wind)

    if ntc > 0:
    
      for tc in np.arange(ntc):
          wind = np.array(all_tc_wind[tc])

          ts_hour = (wind>=17.5).sum() + ts_hour
          h_hour = (wind>=hur_min).sum() + h_hour
          mh_hour = (wind>=mhur_min).sum() + mh_hour
          c45h_hour = (wind>=c45hur_min).sum() + c45h_hour

    ts_day = float(ts_hour)/4.
    h_day = float(h_hour)/4.
    mh_day = float(mh_hour)/4.
    c45h_day = float(c45h_hour)/4.

    return ts_day, h_day, mh_day, c45h_day

#-----------------------------------------------------------------------
# function to calculate acumulative storm energy
#-----------------------------------------------------------------------

def cal_ace(wind, hur_min):

    wind = np.array(wind)

    ts_ace = 0
    hur_ace = 0

    wind[ wind < 17.5 ] = 0
    wind_kt = np.array(wind) * 1.94384
  
    ts_ace = np.sum(wind_kt**2)

    if max(wind) > hur_min:
       wind[ wind < hur_min ] = 0
       wind_kt = np.array(wind) * 1.94384 
       hur_ace = np.sum(wind_kt**2)

    return ts_ace, hur_ace

def cal_ace_more(wind, hur_min, mhur_min, c45hur_min):

    wind = np.array(wind) # important !!!

    ts_ace   = 0
    hur_ace  = 0
    mhur_ace = 0
    c45hur_ace = 0

    wind[ wind < 17.5 ] = 0
    wind_kt = np.array(wind) * 1.94384
  
    ts_ace = np.sum(wind_kt**2)

    if max(wind) > hur_min:
       wind[ wind < hur_min ] = 0
       wind_kt = np.array(wind) * 1.94384 
       hur_ace = np.sum(wind_kt**2)

    if max(wind) > mhur_min:
       wind[ wind < mhur_min ] = 0
       wind_kt = np.array(wind) * 1.94384 
       mhur_ace = np.sum(wind_kt**2)

    if max(wind) > c45hur_min:
       wind[ wind < c45hur_min ] = 0
       wind_kt = np.array(wind) * 1.94384 
       c45hur_ace = np.sum(wind_kt**2)

    return ts_ace, hur_ace, mhur_ace, c45hur_ace

# add geographic constraint

def cal_ace_more_geo(wind, hur_min, mhur_min, c45hur_min, lon_tc, lat_tc, lon1, lon2, lat1, lat2):

    wind = np.array(wind)
    lon_tc = np.array(lon_tc)
    lat_tc = np.array(lat_tc)

    ts_ace   = 0
    hur_ace  = 0
    mhur_ace = 0
    c45hur_ace = 0

    wind[ wind < 17.5 ] = 0

    wind[ lon_tc <= lon1 ] = 0
    wind[ lon_tc >= lon2 ] = 0
    wind[ lat_tc <= lat1 ] = 0
    wind[ lat_tc >= lat2 ] = 0

    wind_kt = np.array(wind) * 1.94384
  
    ts_ace = np.sum(wind_kt**2)

    if max(wind) > hur_min:
       wind[ wind < hur_min ] = 0
       wind_kt = np.array(wind) * 1.94384 
       hur_ace = np.sum(wind_kt**2)

    if max(wind) > mhur_min:
       wind[ wind < mhur_min ] = 0
       wind_kt = np.array(wind) * 1.94384 
       mhur_ace = np.sum(wind_kt**2)

    if max(wind) > c45hur_min:
       wind[ wind < c45hur_min ] = 0
       wind_kt = np.array(wind) * 1.94384 
       c45hur_ace = np.sum(wind_kt**2)

    return ts_ace, hur_ace, mhur_ace, c45hur_ace

#-----------------------------------------------------------------------
# function to cal. GPI
#-----------------------------------------------------------------------

def cal_GPI3(eta,rh,shr):

    GPI = (np.abs(1e5*eta))**(1.5)    \
         *(rh/50.)**(3)               \
         *(1+0.1*shr)**(-2)
    return GPI

#-----------------------------------------------------------------------
# function to determine if a storm is a TC
#-----------------------------------------------------------------------

def if_tc(lon,lat,wind,lat_max):

    storm_life = 12 # minmun record length to be considered as a TC
    TCmin = 17.5
    lat_max = 30  # max lat for the 1st record

    is_tc = False
    is_tc1 = False
    is_tc2 = False

# step 1: basics

    if len(lat) > storm_life and max(wind) > TCmin \
       and lat[0] < lat_max :

       is_tc1 = True

# step 2: wind speed criterion (maintain TCmin for at least 36h)

    if is_tc1 : 
      for i in np.arange(len(lon)-5):
        if wind[i] >= TCmin and wind[i+1] >= TCmin and wind[i+2] >= TCmin \
           and wind[i+3] >= TCmin and wind[i+4] >= TCmin and wind[i+5] >= TCmin:
           
           is_tc2 = True

    if is_tc1 and is_tc2:
       is_tc = True 

    return is_tc

#-----------------------------------------------------------------------
# function to determine if a storm is a hurricane 
#-----------------------------------------------------------------------

def is_hur(wind_list):

    hur1 = 0 # hurricane
    hur2 = 0 # major hurricane (cat >=3)

    wind_th1 = 32.5
    wind_th2 = 49.5

    nn = len(wind_list)
    for i in np.arange(nn):

       wind1=wind_list[i]

       if max(wind1)>=wind_th1:
          hur1 = hur1+1

       if max(wind1)>=wind_th2:
          hur2 = hur2+1

    return hur1, hur2

#-----------------------------------------------------------------------
# function to determine if a storm goes through RI
#-----------------------------------------------------------------------

def if_ri(wind):    
    
    is_ri = False 

    nn = len(wind)
    ri_c = 30*0.51 

    for i in np.arange(nn-4):
       if wind[i] > 10. and (wind[i+4] - wind [i]) >= ri_c :
          is_ri = True
       
    return is_ri

#-----------------------------------------------------------------------
# function to read time series of TC counts/ACE
#-----------------------------------------------------------------------

def read_tc_num(file):

    f = open(file, "r")
    dates=[]
    ts_obs=[]
    ts_mod=[]
    h_obs=[]
    h_mod=[]

    for line in f:
        L = line.split()
        dates += [str(L[0])]
        ts_obs += [float(L[1])]
        ts_mod += [float(L[2])]
        h_obs  += [float(L[3])]
        h_mod  += [float(L[4])]
    return dates, ts_obs, ts_mod, h_obs, h_mod

def read_tc_num_more(file):

    f = open(file, "r")
    dates=[]
    ts_obs=[]
    ts_mod=[]
    h_obs=[]
    h_mod=[]
    mh_obs=[]
    mh_mod=[]
    c45h_obs=[]
    c45h_mod=[]

    for line in f:
        L = line.split()
        dates += [str(L[0])]
        ts_obs += [float(L[1])]
        ts_mod += [float(L[2])]
        h_obs  += [float(L[3])]
        h_mod  += [float(L[4])]
        mh_obs  += [float(L[5])]
        mh_mod  += [float(L[6])]
        c45h_obs  += [float(L[7])]
        c45h_mod  += [float(L[8])]
    return dates, ts_obs, ts_mod, h_obs, h_mod, mh_obs, mh_mod, c45h_obs, c45h_mod

#-----------------------------------------------------------------------
# function to plot TC tracks
#-----------------------------------------------------------------------

def plot_track_withdate(m, lon1, lat1, wind1, date1, color):

# kgao - fix cross 0E issue
    nn=np.size(lon1)
    for i in np.arange(nn):
        if (lon1[i]-lon1[i-1]) < -250.:
            newnn = i
            break
        else:
            newnn = 0
    
    if newnn != 0: 
       lon=lon1[0:newnn]
       lat=lat1[0:newnn]
       wind=wind1[0:newnn]
       date=date1[0:newnn]
    else:
       lon=lon1
       lat=lat1
       wind=wind1
       date=date1

    TSmask = np.array(wind) >= 17.5
    hurmask = np.array(wind) >= 32.5

    x = lon
    y = lat

    xs = np.ma.MaskedArray(x,mask=~TSmask)
    ys = np.ma.MaskedArray(y,mask=~TSmask)
    xh = np.ma.MaskedArray(x,mask=~hurmask)
    yh = np.ma.MaskedArray(y,mask=~hurmask)

    m.plot(x,y,color=color,linewidth=1.)
    m.plot(xs,ys,color=color,linewidth=2.)
    m.plot(xh,yh,color=color,linewidth=4.)

    yoffset = 0.022*(m.ymax-m.ymin)/3

    xi=x[0]
    yi=y[0]
    datei=date[0]
    #if xi >= m.xmin and xi <= m.xmax and yi >= m.ymin and yi <= m.ymax:
    #   plt.text(xi,yi+yoffset,datei[4:8],color=color,ha='right',fontsize=16)
    #   plt.text(xi,yi,'o',fontsize=12,color=color,ha='center',va='center')

#-----------------------------------------------------------------------
# function to plot TC tracks - cat 1-5
#-----------------------------------------------------------------------

def plot_track_cat(m, lon1, lat1, wind1):

    col_td = 'deepskyblue'
    col_ts = 'deepskyblue'
    col_c1 = 'orange'
    col_c2 = 'coral'
    col_c3 = 'tomato'
    col_c4 = 'red'
    col_c5 = 'maroon'

    col_td = 'dodgerblue'
    col_ts = 'aqua'
    col_c1 = 'lightyellow'
    col_c2 = 'gold'
    col_c3 = 'orange'
    col_c4 = 'darkorange'
    col_c5 = 'red'

    col_td = 'dodgerblue'
    col_ts = 'aqua'
    col_c1 = 'orange'
    col_c2 = 'coral'
    col_c3 = 'tomato'
    col_c4 = 'red'
    col_c5 = 'maroon'

    linewi = 1.5

# kgao - fix cross 0E issue
    nn=np.size(lon1)
    for i in np.arange(nn):
        if (lon1[i]-lon1[i-1]) < -250.:
            newnn = i
            break
        else:
            newnn = 0
    
    if newnn != 0: 
       lon=lon1[0:newnn]
       lat=lat1[0:newnn]
       wind=wind1[0:newnn]
    else:
       lon=lon1
       lat=lat1
       wind=wind1

    ts_mask = np.array(wind) >= 17.5
    c1_mask = np.array(wind) >= 33.
    c2_mask = np.array(wind) >= 43.
    c3_mask = np.array(wind) >= 49.5
    c4_mask = np.array(wind) >= 58.
    c5_mask = np.array(wind) >= 70.

    x = lon
    y = lat

    xs = np.ma.MaskedArray(x,mask=~ts_mask)
    ys = np.ma.MaskedArray(y,mask=~ts_mask)

    xc1 = np.ma.MaskedArray(x,mask=~c1_mask)
    yc1 = np.ma.MaskedArray(y,mask=~c1_mask)

    xc2 = np.ma.MaskedArray(x,mask=~c2_mask)
    yc2 = np.ma.MaskedArray(y,mask=~c2_mask)

    xc3 = np.ma.MaskedArray(x,mask=~c3_mask)
    yc3 = np.ma.MaskedArray(y,mask=~c3_mask)

    xc4 = np.ma.MaskedArray(x,mask=~c4_mask)
    yc4 = np.ma.MaskedArray(y,mask=~c4_mask)

    xc5 = np.ma.MaskedArray(x,mask=~c5_mask)
    yc5 = np.ma.MaskedArray(y,mask=~c5_mask)

    m.plot(x,y,color=col_td,linewidth=linewi)
    m.plot(xs,ys,color=col_ts,linewidth=linewi)
    m.plot(xc1,yc1,color=col_c1,linewidth=linewi)
    m.plot(xc2,yc2,color=col_c2,linewidth=linewi)
    m.plot(xc3,yc3,color=col_c3,linewidth=linewi)
    m.plot(xc4,yc4,color=col_c4,linewidth=linewi)
    m.plot(xc5,yc5,color=col_c5,linewidth=linewi)

#-----------------------------------------------------------------------
# function to plot TC tracks - simple 
#-----------------------------------------------------------------------

def plot_track_simple(m, lon1, lat1, wind1):

# kgao - fix cross 0E issue
    nn=np.size(lon1)
    for i in np.arange(nn):
        if (lon1[i]-lon1[i-1]) < -250. :
            newnn = i
            break
        else:
            newnn = 0
    
    if newnn != 0: 
       lon=lon1[0:newnn]
       lat=lat1[0:newnn]
       wind=wind1[0:newnn]

    else:
       lon=lon1
       lat=lat1
       wind=wind1

    TSmask = np.array(wind) >= 17.5
    hurmask = np.array(wind) >= 32.5

    x = lon
    y = lat

    xs = np.ma.MaskedArray(x,mask=~TSmask)
    ys = np.ma.MaskedArray(y,mask=~TSmask)
    xh = np.ma.MaskedArray(x,mask=~hurmask)
    yh = np.ma.MaskedArray(y,mask=~hurmask)

    m.plot(x,y,color='b',linewidth=1.5)
    #m.plot(xs,ys,color='b',linewidth=1.5)
    m.plot(xh,yh,color='r',linewidth=1.5)
#    m.plot(x,y,'ko')


#-----------------------------------------------------------------------
# function to plot TC tracks - simple 2  
#-----------------------------------------------------------------------

def plot_track_simple2(m, lon, lat, wind):


    TSmask = np.array(wind) >= 17.5
    hurmask = np.array(wind) >= 32.5
    mhurmask = np.array(wind) >= 49.5

    x = lon
    y = lat

    xs = np.ma.MaskedArray(x,mask=~TSmask)
    ys = np.ma.MaskedArray(y,mask=~TSmask)
    xh = np.ma.MaskedArray(x,mask=~hurmask)
    yh = np.ma.MaskedArray(y,mask=~hurmask)
    xmh = np.ma.MaskedArray(x,mask=~mhurmask)
    ymh = np.ma.MaskedArray(y,mask=~mhurmask)

    m.plot(x,y,'b.')
    m.plot(xh,yh,'r.')
    m.plot(xmh,ymh,'m.')

#-----------------------------------------------------------------------
# function to plot TC tracks - simple 3  
#-----------------------------------------------------------------------

def plot_track_simple3(m, lon, lat, wind):


      TSmask = np.array(wind) >= 17.5
      hurmask = np.array(wind) >= 32.5

      x = lon
      y = lat

      xs = np.ma.MaskedArray(x,mask=~TSmask)
      ys = np.ma.MaskedArray(y,mask=~TSmask)
      xh = np.ma.MaskedArray(x,mask=~hurmask)
      yh = np.ma.MaskedArray(y,mask=~hurmask)

      m.plot(x,y,'k-')
      m.plot(x,y,'ko')
      #m.plot(xs,ys,'bo')
      #m.plot(xh,yh,'ro')

#-----------------------------------------------------------------------
# function to plot TC tracks - only over land
#-----------------------------------------------------------------------

def plot_track_land(m, lon1, lat1, wind1):

    is_land = over_land(np.array(lon1), np.array(lat1))
    lat1 = np.array(lat1)*is_land
    lon1 = np.array(lon1)*is_land

    m.plot(lon1,lat1,'ko')

#-----------------------------------------------------------------------
# function to plot time series of TC number 
#-----------------------------------------------------------------------

def plot_tc_num(ax, dates, record, col, leg, linewi, marksi):

    nn=np.size(dates)
    time = np.arange(nn)
    date_ticks=[]
    for i in np.arange(nn):

       if isinstance(dates[i], basestring):
          date_ticks.append(dates[i][2:])
       else:
          date_ticks.append(dates[i].strftime('%m%d'))

    ft=16
    ft1=12

    ax.plot(time,record,color=col, label=leg, linewidth=linewi, marker='o', markersize=marksi)
    ax.grid(True)

    ax.legend(loc='upper right',frameon=False)

    # set y axis
    ax.set_ylim(-1,16)
    ax.set_yticks(np.arange(0,16+2,2))
    ax.set_ylabel('TC number',fontsize=ft)

    # set x axis
    ax.set_xlim(-1,nn)
    ax.set_xticks(time)
    ax.set_xticklabels(date_ticks,rotation=90)
    #ax.set_xlabel('Exp start date',fontsize=ft)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ft1)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ft1)

#-----------------------------------------------------------------------
# function to plot TC prediction score
#-----------------------------------------------------------------------

def plt_bar(ax,x,y,color,label,xtick,xtick_label,ytick):

   ft=12
   ft1=12

   ax.bar(x, y, color = color, label = label)
   ax.legend(loc='upper right',frameon=False)

   ax.set_xlim(np.min(xtick)-1,np.max(xtick)+2)
   ax.set_xticks(xtick+0.4)
   ax.set_xticklabels(xtick_label)

   ax.set_ylim(np.min(ytick),np.max(ytick))
   ax.set_yticks(ytick)

   for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(ft1)
   for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(ft1)

def plot_tc_score_bar(ax, dates, score, width, col):

    nn=np.size(dates)
    time = np.arange(nn)

    date_ticks=[]
    for i in np.arange(nn):

       if isinstance(dates[i], basestring):
          date_ticks.append(dates[i][0:])
       else:
          date_ticks.append(dates[i].strftime('%m%d'))

    ft=20 # need to be consistent with following 
    ft1=20

    ax.bar(time, score, width, color = col)

    ax.legend(loc='upper right',frameon=False,fontsize=ft1)

    # set y axis
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1,0,1])

    # set x axis
    ax.set_xlim(-1,nn)
    ax.set_xticks(time[0::5])
    ax.set_xticklabels(date_ticks[0::5],rotation=45)
    #ax.set_xlabel('Exp start date',fontsize=ft)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ft1)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ft1)

#-----------------------------------------------------------------------
# function to plot time series of TC statistics (counts, ace, etc.)
#-----------------------------------------------------------------------

def plot_tc_stat(ax, dates, record, col, leg, linewi, marksi, yticks):

    nn=np.size(dates)
    time = np.arange(nn)

    date_ticks=[]
    for i in np.arange(nn):

       if isinstance(dates[i], basestring):
          date_ticks.append(dates[i][2:])
       else:
          date_ticks.append(dates[i].strftime('%m%d'))

    ft=16 # need to be consistent with following 
    ft1=16

    ax.plot(time,record,color=col, label=leg, linewidth=linewi, marker='o', markersize=marksi)
    ax.grid(True)

    ax.legend(loc='upper right',frameon=False,fontsize=ft1)

    # set y axis
    ax.set_ylim(np.min(yticks),np.max(yticks))
    ax.set_yticks(yticks)

    # set x axis
    ax.set_xlim(-1,nn)
    ax.set_xticks(time[0::])
    ax.set_xticklabels(date_ticks[0::],rotation=90)
    #ax.set_xlabel('Exp start date',fontsize=ft)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ft1)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ft1)

#-----------------------------------------------------------------------
# function to plot time series of TC statistics (counts, ace, etc.)
#-----------------------------------------------------------------------

def plot_tc_stat_wrange(ax, dates, record, record_lb, record_ub, col, leg, linewi, marksi, yticks):

    nn=np.size(dates)
    time = np.arange(nn)

    date_ticks=[]
    for i in np.arange(nn):

       if isinstance(dates[i], basestring):
          date_ticks.append(dates[i][2:])
       else:
          date_ticks.append(dates[i].strftime('%m%d')) 
 
    ft=16 # need to be consistent with above
    ft1=16

    ax.plot(time,record,color=col, label=leg, linewidth=linewi, marker='o', markersize=marksi)
    ax.fill_between(time, record_lb, record_ub, facecolor = col, alpha = 0.25)
    ax.grid(True)

    ax.legend(loc='upper right',frameon=False,fontsize=ft1)

    # set y axis
    ax.set_ylim(np.min(yticks),np.max(yticks))
    ax.set_yticks(yticks)

    # set x axis
    ax.set_xlim(-1,nn)
    ax.set_xticks(time[0::5])
    ax.set_xticklabels(date_ticks[0::5],rotation=45)
    #ax.set_xlabel('Exp start date',fontsize=ft)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ft1)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ft1)

#-----------------------------------------------------------------------
# function to plot time series of TC statistics (counts, ace, etc.)
#-----------------------------------------------------------------------

def plot_tc_stat_wrange_wobs(ax, dates, record_obs, col_obs, leg_obs,\
                                        record, record_lb, record_ub, col, leg, \
                                        linewi, marksi, yticks, ylabel):

    nn=np.size(dates)
    time = np.arange(nn)

    date_ticks=[]
    for i in np.arange(nn):

       if isinstance(dates[i], basestring):
          #date_ticks.append(dates[i][2:])
          #date_ticks.append(dates[i][:-2])
           date_ticks.append(dates[i][:4]+'-'+dates[i][4:6])
       else:
          date_ticks.append(dates[i].strftime('%m%d'))


    #date_ticks_new = len(date_ticks)
    #for dd in np.arange(len(date_ticks)):
    #    date_ticks_new[dd]=date_tick[dd][:4]+'-'+date_tick[dd][4:6]
    #date_ticks = date_ticks_new  

    ft=24 
    ft1=24

    ax.plot(time,record_obs,color=col_obs, label=leg_obs, linewidth=linewi, marker='o', markersize=marksi)

    ax.plot(time,record,color=col, label=leg, linewidth=linewi, marker='o', markersize=marksi)
    ax.fill_between(time, record_lb, record_ub,facecolor=col, alpha=0.25)

    ax.grid(True)

    ax.legend(loc='upper right',frameon=False,fontsize=ft1)

    # set y axis
    ax.set_ylim(np.min(yticks),np.max(yticks))
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel,fontsize=ft)

    # set x axis
    ax.set_xlim(-1,nn)
    ax.set_xticks(time[0::5])
    ax.set_xticklabels(date_ticks[0::5],rotation=90)
    #ax.set_xlabel('Exp start date',fontsize=ft)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ft1)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ft1)

#-----------------------------------------------------------------------
# function to calculate TC track density - 
# defined as number of TCs passing certain lat range 
# (Based on Vitart 2009, GRL)
#-----------------------------------------------------------------------

def find_tc_track_density(lonm, latm, lon_ntc, lat_ntc, wind_ntc, TSmin, Hmin, deg):

  nx,ny = np.shape(lonm)

  tc_num=np.zeros((nx,ny))
  ts_num=np.zeros((nx,ny))
  hur_num=np.zeros((nx,ny))
    
  ntc = len(lon_ntc)
  
  for tc1 in np.arange(ntc):

#    if ntc > 100 :
#       print 'TC record', tc1
    all_lon  = lon_ntc[tc1]
    all_lat  = lat_ntc[tc1]
    all_wind = wind_ntc[tc1]

# remove TC track record cross 0E
    nn=np.size(all_lon)
    for i in np.arange(nn):
        if (all_lon[i]-all_lon[i-1]) < -250.:
            newnn = i
            break
        else:
            newnn = 0
    if newnn != 0: 
       lon=all_lon[0:newnn]
       lat=all_lat[0:newnn]
    else:
       lon=all_lon
       lat=all_lat

    nn = len(lon)

    for k in np.arange(nn-1):

        dist2=(lonm-lon[k])**2 + (latm-lat[k])**2
        indi,indj=np.where(dist2==np.min(dist2))

        dlon_mesh=lonm[1][0]-lonm[0][0] 
        scope = int(2*deg/dlon_mesh)

        sel_i=np.arange(indi[0]-scope,indi[0]+scope)
        sel_j=np.arange(indj[0]-scope,indj[0]+scope)

        for i in sel_i:
          for j in sel_j:
            if i >-1 and i < nx-1 and j > -1 and j < ny-1:
                case1=False
                case2=False
                case3=False
                case4=False
                case5=False
                case6=False
                if lon[k]>=lonm[i][j] and lon[k+1]<=lonm[i][j] \
                   and lat[k]>=latm[i][j]-deg and lat[k]<=latm[i][j]+deg:
                   case1=True
                elif lon[k]>=lonm[i][j] and lon[k+1]<=lonm[i][j] \
                   and lat[k+1]>=latm[i][j]-deg and lat[k+1]<=latm[i][j]+deg:
                   case2=True 
                elif lon[k]<=lonm[i][j] and lon[k+1]>=lonm[i][j] \
                   and lat[k]>=latm[i][j]-deg and lat[k]<=latm[i][j]+deg:
                   case3=True
                elif lon[k]<=lonm[i][j] and lon[k+1]>=lonm[i][j] \
                   and lat[k+1]>=latm[i][j]-deg and lat[k+1]<=latm[i][j]+deg:
                   case4=True 
                elif lon[k]>=lonm[i][j] and lon[k+1]<=lonm[i][j] \
                   and lat[k]<=latm[i][j]-deg and lat[k+1]>=latm[i][j]+deg:
                   case5=True 
                elif lon[k]<=lonm[i][j] and lon[k+1]>=lonm[i][j] \
                   and lat[k]<=latm[i][j]-deg and lat[k+1]>=latm[i][j]+deg:
                   case6=True 
                if case1 or case2 or case3 or case4 or case5 or case6:
                   tc_num[i][j]=tc_num[i][j]+1

                   if all_wind[k] >= TSmin or all_wind[k+1] >= TSmin:
                      ts_num[i][j]=ts_num[i][j]+1

                   if all_wind[k] >= Hmin or all_wind[k+1] >= Hmin:
                      hur_num[i][j]=hur_num[i][j]+1

  return tc_num, ts_num, hur_num

#-----------------------------------------------------------------------
# function to calculate TC genesis density  
#-----------------------------------------------------------------------

def find_tc_gen_density(lonm, latm, lon, lat, deg):

    nx,ny = np.shape(lonm)

    tc_num=np.zeros((nx,ny))

    nn = len(lon)

    for k in np.arange(nn):

        dist2=(lonm-lon[k])**2 + (latm-lat[k])**2
        indi,indj=np.where(dist2==np.min(dist2))

        dlon_mesh=lonm[1][0]-lonm[0][0] 
        scope = int(1.5*deg/dlon_mesh)
        sel_i=np.arange(indi[0]-scope,indi[0]+scope+1)
        sel_j=np.arange(indj[0]-scope,indj[0]+scope+1)

        for i in sel_i:
          for j in sel_j:
            if i >-1 and i < nx-1 and j > -1 and j < ny-1:
                if lon[k]>=lonm[i][j]-deg and lon[k]<=lonm[i][j]+deg \
                   and lat[k]>=latm[i][j]-deg and lat[k]<=latm[i][j]+deg:

                   tc_num[i][j]=tc_num[i][j]+1

    return tc_num

#-----------------------------------------------------------------------
# function to calculate TC record density  
#-----------------------------------------------------------------------

# need wind input - different from find genesis density

def find_tc_record_density(lonm, latm, lon, lat, wind, wind_th, deg):

    nx,ny = np.shape(lonm)

    tc_num=np.zeros((nx,ny))

    nn = len(lon)

    for k in np.arange(nn):

        dist2=(lonm-lon[k])**2 + (latm-lat[k])**2
        indi,indj=np.where(dist2==np.min(dist2))

        dlon_mesh=lonm[1][0]-lonm[0][0] 
        scope = int(1.5*deg/dlon_mesh)
        sel_i=np.arange(indi[0]-scope,indi[0]+scope+1)
        sel_j=np.arange(indj[0]-scope,indj[0]+scope+1)

        for i in sel_i:
          for j in sel_j:
            if i >-1 and i < nx-1 and j > -1 and j < ny-1:
                if lon[k]>=lonm[i][j]-deg and lon[k]<=lonm[i][j]+deg \
                   and lat[k]>=latm[i][j]-deg and lat[k]<=latm[i][j]+deg \
                   and wind[k]>=wind_th : 

                   tc_num[i][j]=tc_num[i][j]+1

    return tc_num


def find_tc_record_density_3(lonm, latm, lon, lat, wind, deg):

    nx,ny = np.shape(lonm)

    ts_num=np.zeros((nx,ny))
    hur_num=np.zeros((nx,ny))
    mhur_num=np.zeros((nx,ny))

    nn = len(lon)

    for k in np.arange(nn):

        dist2=(lonm-lon[k])**2 + (latm-lat[k])**2
        indi,indj=np.where(dist2==np.min(dist2))

        dlon_mesh=lonm[1][0]-lonm[0][0] 
        scope = int(1.5*deg/dlon_mesh)
        sel_i=np.arange(indi[0]-scope,indi[0]+scope+1)
        sel_j=np.arange(indj[0]-scope,indj[0]+scope+1)

        for i in sel_i:
          for j in sel_j:
            if i >-1 and i < nx-1 and j > -1 and j < ny-1:
                if lon[k]>=lonm[i][j]-deg and lon[k]<=lonm[i][j]+deg \
                   and lat[k]>=latm[i][j]-deg and lat[k]<=latm[i][j]+deg:


                   if wind[k]>=17.5 : 
                      ts_num[i][j]=ts_num[i][j]+1

                   if wind[k]>=32.5 : 
                      hur_num[i][j]=hur_num[i][j]+1

                   if wind[k]>=49.5 : 
                      mhur_num[i][j]=mhur_num[i][j]+1

    return ts_num, hur_num, mhur_num

#-----------------------------------------------------------------------
# function to sort out TC records 
# return number and ace of ts/h/mh/c45
#        landfall to be added ?
#-----------------------------------------------------------------------

def sort_tc_records_season(lon_all, lat_all, wind_all):

    ts_num = 0.
    h_num = 0.
    mh_num = 0.
    c45h_num = 0.

    ts_ace = 0.
    h_ace = 0.
    mh_ace = 0.
    c45h_ace = 0.

    ntc = len(lon_all) 

    if ntc > 0:

     for tc in np.arange(ntc):

         lon = np.array(lon_all[tc]) 
         lat = np.array(lat_all[tc])
         wind = np.array(wind_all[tc])

         wind_max = np.max(wind)
         if wind_max >= 17.5:
            ts_num = ts_num+1 
         if wind_max >= 32.5:
            h_num = h_num+1 
         if wind_max >= 49.5:
            mh_num = mh_num+1 
         if wind_max >= 58.:
            c45h_num = c45h_num+1 

         ts_ace1, h_ace1, mh_ace1, c45h_ace1 = cal_ace_more(wind, 32.5, 49.5, 58.)
         ts_ace = ts_ace + ts_ace1
         h_ace = h_ace + h_ace1
         mh_ace = mh_ace + mh_ace1
         c45h_ace = c45h_ace + c45h_ace1

    return ts_num, h_num, mh_num, c45h_num, ts_ace, h_ace, mh_ace, c45h_ace


def sort_tc_records_month(lon_all, lat_all, wind_all, date_all):

    time_list = ['070100','080100','090100','100100','110100','120100']

    nm = len(time_list)-1

    ts_num_all = np.zeros(nm)
    h_num_all = np.zeros(nm)
    mh_num_all = np.zeros(nm)
    c45h_num_all = np.zeros(nm)

    ts_ace_all = np.zeros(nm)
    h_ace_all = np.zeros(nm)
    mh_ace_all = np.zeros(nm)
    c45h_ace_all = np.zeros(nm)

    ntc = len(lon_all) 

    if ntc > 0:

     for tt in np.arange(nm):

      ts_num = 0.
      h_num = 0.
      mh_num = 0.
      c45h_num = 0.

      ts_ace = 0.
      h_ace = 0.
      mh_ace = 0.
      c45h_ace = 0.

      tstart = int(time_list[tt])
      tend = int(time_list[tt+1])

      for tc in np.arange(ntc):
 
       lon = np.array(lon_all[tc]) 
       lat = np.array(lat_all[tc])
       wind = np.array(wind_all[tc])

       tind = [ n for n,i in enumerate(wind) if i>17.5 ][0]
       time_1st = int(date_all[tc][tind][4:])

       #time_1st = int(date_all[tc][0][4:]) 

       if time_1st >=tstart and time_1st < tend:
          
         wind_max = np.max(wind)
         if wind_max >= 17.5:
            ts_num = ts_num+1 
         if wind_max >= 32.5:
            h_num = h_num+1 
         if wind_max >= 49.5:
            mh_num = mh_num+1 
         if wind_max >= 58.:
            c45h_num = c45h_num+1 

         ts_ace1, h_ace1, mh_ace1, c45h_ace1 = cal_ace_more(wind, 32.5, 49.5, 58.)
         ts_ace = ts_ace + ts_ace1
         h_ace = h_ace + h_ace1
         mh_ace = mh_ace + mh_ace1
         c45h_ace = c45h_ace + c45h_ace1

       ts_num_all[tt]=ts_num
       h_num_all[tt]=h_num
       mh_num_all[tt]=mh_num
       c45h_num_all[tt]=c45h_num

       ts_ace_all[tt]=ts_ace
       h_ace_all[tt]=h_ace
       mh_ace_all[tt]=mh_ace
       c45h_ace_all[tt]=c45h_ace

    return ts_num_all, h_num_all, mh_num_all, c45h_num_all, ts_ace_all, h_ace_all, mh_ace_all, c45h_ace_all

# simple version

def sort_tc_records(lon, lat, wind):

    nrec = len(lon) 

    ts_ace = 0.
    h_ace = 0.
    mh_ace = 0.

    if nrec > 0:

     wind_kt = np.array(wind) * 1.94384

     for k in np.arange(nrec):

      if wind[k]>=17.5:
         ts_ace=ts_ace+wind_kt[k]**2

      if wind[k]>=32.5:
         h_ace=h_ace+wind_kt[k]**2

      if wind[k]>=49.5:
         mh_ace=mh_ace+wind_kt[k]**2

    return ts_ace, h_ace, mh_ace

#-----------------------------------------------------------------------
# function to calculate regional TC activity  
#-----------------------------------------------------------------------

def find_tc_gen_regional(lonm, latm, lon, lat, deg):

     nx,ny = np.shape(lonm)

     ts_num=np.zeros((nx,ny))

     ntc = len(lon)

     for k in np.arange(ntc): # loop over all TCs

        dist2=(lonm-lon[k])**2 + (latm-lat[k])**2
        indi,indj=np.where(dist2==np.min(dist2))

        dlon_mesh=lonm[1][0]-lonm[0][0] 
        scope = int(2.5*deg/dlon_mesh)
        sel_i=np.arange(indi[0]-scope,indi[0]+scope+1)
        sel_j=np.arange(indj[0]-scope,indj[0]+scope+1)

        for i in sel_i:
          for j in sel_j:
            if i >-1 and i < nx-1 and j >-1 and j < ny-1:
                if lon[k]>=lonm[i][j]-deg and lon[k]<=lonm[i][j]+deg \
                   and lat[k]>=latm[i][j]-deg and lat[k]<=latm[i][j]+deg: 
                 
                      ts_num[i][j]=ts_num[i][j]+1

     return ts_num

# --- 

def find_tc_regional(lonm, latm, lon_list, lat_list, wind_list, deg):

    nx,ny = np.shape(lonm)

    ts_num=np.zeros((nx,ny))
    h_num=np.zeros((nx,ny))
    mh_num=np.zeros((nx,ny))
    c45h_num=np.zeros((nx,ny))

    ts_ace=np.zeros((nx,ny))
    h_ace=np.zeros((nx,ny))
    mh_ace=np.zeros((nx,ny))
    c45h_ace=np.zeros((nx,ny))

    ntc = len(lon_list) # number of TCs

    if ntc > 0:

     for tc in np.arange(ntc): # loop over all TCs

      ts_num1=np.zeros((nx,ny))
      h_num1=np.zeros((nx,ny))
      mh_num1=np.zeros((nx,ny))
      c45h_num1=np.zeros((nx,ny))

      lon = lon_list[tc]
      lat = lat_list[tc]
      wind = wind_list[tc]

      wind_kt = np.array(wind) * 1.94384

      nrec = len(lon)

      for k in np.arange(nrec): 

        dist2=(lonm-lon[k])**2 + (latm-lat[k])**2
        indi,indj=np.where(dist2==np.min(dist2))

        dlon_mesh=lonm[1][0]-lonm[0][0] 
        scope = int(1.5*deg/dlon_mesh)
        sel_i=np.arange(indi[0]-scope,indi[0]+scope+1)
        sel_j=np.arange(indj[0]-scope,indj[0]+scope+1)

        for i in sel_i:
          for j in sel_j:
            if i >-1 and i < nx-1 and j >-1 and j < ny-1:
                if lon[k]>=lonm[i][j]-deg and lon[k]<=lonm[i][j]+deg \
                   and lat[k]>=latm[i][j]-deg and lat[k]<=latm[i][j]+deg: 

                   if wind[k]>=17.5:
                      ts_num1[i][j]=1
                      ts_ace[i][j]=ts_ace[i][j]+wind_kt[k]**2

                   if wind[k]>=32.5:
                      h_num1[i][j]=1
                      h_ace[i][j]=h_ace[i][j]+wind_kt[k]**2

                   if wind[k]>=49.5:
                      mh_num1[i][j]=1
                      mh_ace[i][j]=mh_ace[i][j]+wind_kt[k]**2

                   if wind[k]>=58.0:
                      c45h_num1[i][j]=1
                      c45h_ace[i][j]=c45h_ace[i][j]+wind_kt[k]**2

      ts_num = ts_num + ts_num1
      h_num = h_num + h_num1
      mh_num = mh_num + mh_num1
      c45_num = c45h_num + c45h_num1

    return ts_num, h_num, mh_num, c45h_num, ts_ace, h_ace, mh_ace, c45h_ace

########################################################################
# functions - EOF and MJO 
########################################################################

#----------------------------------
# function to sel obs rmm

def sel_WH04(date_sel, nt, rmm_file):

  year_sel = int(date_sel[:4])
  mo_sel   = int(date_sel[4:6])
  dd_sel   = int(date_sel[6:])

  date_str = dt.date(year_sel, mo_sel, dd_sel)
  date_end = date_str + dt.timedelta(days=nt)

  #rmm_file =  './0txt_obs_rmm/rmm.74toRealtime.txt'
  frmm = open(rmm_file, "r")

  pc1 = np.zeros(nt)
  pc2 = np.zeros(nt)
  amp = np.zeros(nt)
  phase = np.zeros(nt)

  ind = 0

  for line in frmm:
      L = line.split()
      yr_r = int(L[0])
      mo_r = int(L[1])
      dd_r = int(L[2])
      date_r = dt.date(yr_r, mo_r, dd_r)

      if date_r >= date_str and date_r < date_end:

           #print date_r, ind
           pc1[ind]  = float(L[3])
           pc2[ind]  = float(L[4])
           amp[ind]  = float(L[6])
           phase[ind]= int(L[5])
           ind=ind+1

  return pc1, pc2

#-----------------------------------------------------------------------
# function to get phase based on time index of EOF mode 1
#-----------------------------------------------------------------------

# phase 1 : < -3 std ...
        
def get_eof_phase(pc1):

    phase = np.zeros(len(pc1))

    for ind in np.arange(len(pc1)):

        if pc1[ind] <= -2 :
           phase[ind] = 1

        elif pc1[ind] > -2 and pc1[ind] <= -1 :
           phase[ind] = 2

        elif pc1[ind] > -1 and pc1[ind] < 0 :
           phase[ind] = 3

        elif pc1[ind] > 0 and pc1[ind] < 1 :
           phase[ind] = 4

        elif pc1[ind] >= 1 and pc1[ind] < 2 :
           phase[ind] = 5

        elif pc1[ind] >=2 :
           phase[ind] = 6

    return phase

#-----------------------------------------------------------------------
# function to sort out Phase n (1-8) days 
#-----------------------------------------------------------------------

def find_days(index1, index2_phase, index2_amp, target_phase):

   days1 = 0. 
   days2 = 0.
   days3 = 0.
   days4 = 0.
   days5 = 0.
   days6 = 0.

   for i in np.arange(len(index1)):

    if index2_phase[i] == target_phase and abs(index2_amp[i]) >= 1.:

      if index1[i] <= -2 :
         days1 = days1+1

      if index1[i] > -2 and index1[i] <= -1 :
         days2 = days2+1

      if index1[i] > -1 and index1[i] < 0 :
         days3 = days3+1

      if index1[i] > 0 and index1[i] < 1 :
         days4 = days4+1

      if index1[i] >= 1 and index1[i] < 2 :
         days5 = days5+1

      if index1[i] >=2 :
         days6 = days6+1

   days = [days1, days2, days3, days4, days5, days6]
   return days

#-----------------------------------------------------------------------
# function to write out MJO composites to nc files
#-----------------------------------------------------------------------

def write_eof_nc( var_name, output_file, lon, lat, var_clim_ave, \
                  var_p1, var_p2, var_p3, var_p4, var_p5, var_p6,
                  vara_p1,vara_p2,vara_p3,vara_p4,vara_p5,vara_p6,):

 # define var names
 var_namec = var_name+'_clim'
 var_name1 = var_name+'_p1'
 var_name2 = var_name+'_p2'
 var_name3 = var_name+'_p3'
 var_name4 = var_name+'_p4'
 var_name5 = var_name+'_p5'
 var_name6 = var_name+'_p6'

 vara_name1 = var_name+'a_p1'
 vara_name2 = var_name+'a_p2'
 vara_name3 = var_name+'a_p3'
 vara_name4 = var_name+'a_p4'
 vara_name5 = var_name+'a_p5'
 vara_name6 = var_name+'a_p6'

 fnc = Dataset(output_file, 'w',format='NETCDF4_CLASSIC')

 # create dim
 nx,ny = np.shape(var_p1)
 X = fnc.createDimension('X', nx)
 Y = fnc.createDimension('Y', ny)

 # create var 
 lon_mjo = fnc.createVariable('lon', np.float32, ('Y',))
 lat_mjo = fnc.createVariable('lat', np.float32, ('X',))
 
 var_mjo_clim  = fnc.createVariable(var_namec, np.float32, ('X', 'Y'))
 var_mjo_p1  = fnc.createVariable(var_name1, np.float32, ('X', 'Y'))
 var_mjo_p2  = fnc.createVariable(var_name2, np.float32, ('X', 'Y'))
 var_mjo_p3  = fnc.createVariable(var_name3, np.float32, ('X', 'Y'))
 var_mjo_p4  = fnc.createVariable(var_name4, np.float32, ('X', 'Y'))
 var_mjo_p5  = fnc.createVariable(var_name5, np.float32, ('X', 'Y'))
 var_mjo_p6  = fnc.createVariable(var_name6, np.float32, ('X', 'Y'))

 vara_mjo_p1  = fnc.createVariable(vara_name1, np.float32, ('X', 'Y'))
 vara_mjo_p2  = fnc.createVariable(vara_name2, np.float32, ('X', 'Y'))
 vara_mjo_p3  = fnc.createVariable(vara_name3, np.float32, ('X', 'Y'))
 vara_mjo_p4  = fnc.createVariable(vara_name4, np.float32, ('X', 'Y'))
 vara_mjo_p5  = fnc.createVariable(vara_name5, np.float32, ('X', 'Y'))
 vara_mjo_p6  = fnc.createVariable(vara_name6, np.float32, ('X', 'Y'))

 # write var
 lon_mjo[:] = lon
 lat_mjo[:] = lat

 var_mjo_clim[:,:] = var_clim_ave
 var_mjo_p1[:,:] = var_p1
 var_mjo_p2[:,:] = var_p2
 var_mjo_p3[:,:] = var_p3
 var_mjo_p4[:,:] = var_p4
 var_mjo_p5[:,:] = var_p5
 var_mjo_p6[:,:] = var_p6

 vara_mjo_p1[:,:] = vara_p1
 vara_mjo_p2[:,:] = vara_p2
 vara_mjo_p3[:,:] = vara_p3
 vara_mjo_p4[:,:] = vara_p4
 vara_mjo_p5[:,:] = vara_p5
 vara_mjo_p6[:,:] = vara_p6
 
 fnc.close()

#-----------------------------------------------------------------------
# function to read mjo all
#-----------------------------------------------------------------------

def read_eof_all_nc(file, var_name, lon1, lon2, lat1, lat2):

 #var_namec = var_name+'_clim'
 var_name1 = var_name+'_p1_all'
 var_name2 = var_name+'_p2_all'
 var_name3 = var_name+'_p3_all'
 var_name4 = var_name+'_p4_all'
 var_name5 = var_name+'_p5_all'
 var_name6 = var_name+'_p6_all'

 f1 = Dataset(file, 'r')

 lon_all = f1.variables['lon'][:]
 lat_all = f1.variables['lat'][:]

 if lat_all[1] > lat_all[0]:
    indx1 = find_nearest(lat_all,lat1)
    indx2 = find_nearest(lat_all,lat2)+1
 else:
    indx1 = find_nearest(lat_all,lat2)
    indx2 = find_nearest(lat_all,lat1)+1

 indy1 = find_nearest(lon_all,lon1)
 indy2 = find_nearest(lon_all,lon2)+1

 lat = lat_all[indx1:indx2]
 lon = lon_all[indy1:indy2]

 #var_clim = f1.variables[var_namec][:,indx1:indx2,indy1:indy2]
 var_p1 = f1.variables[var_name1][:,indx1:indx2,indy1:indy2]
 var_p2 = f1.variables[var_name2][:,indx1:indx2,indy1:indy2]
 var_p3 = f1.variables[var_name3][:,indx1:indx2,indy1:indy2]
 var_p4 = f1.variables[var_name4][:,indx1:indx2,indy1:indy2]
 var_p5 = f1.variables[var_name5][:,indx1:indx2,indy1:indy2]
 var_p6 = f1.variables[var_name6][:,indx1:indx2,indy1:indy2]

 return lon,lat, var_p1,var_p2,var_p3,var_p4,var_p5,var_p6

# with clim
def read_eof_all_nc_2(file, var_name, lon1, lon2, lat1, lat2):

 var_namec = var_name+'_clim'
 var_name1 = var_name+'_p1_all'
 var_name2 = var_name+'_p2_all'
 var_name3 = var_name+'_p3_all'
 var_name4 = var_name+'_p4_all'
 var_name5 = var_name+'_p5_all'
 var_name6 = var_name+'_p6_all'

 f1 = Dataset(file, 'r')

 lon_all = f1.variables['lon'][:]
 lat_all = f1.variables['lat'][:]

 if lat_all[1] > lat_all[0]:
    indx1 = find_nearest(lat_all,lat1)
    indx2 = find_nearest(lat_all,lat2)+1
 else:
    indx1 = find_nearest(lat_all,lat2)
    indx2 = find_nearest(lat_all,lat1)+1

 indy1 = find_nearest(lon_all,lon1)
 indy2 = find_nearest(lon_all,lon2)+1

 lat = lat_all[indx1:indx2]
 lon = lon_all[indy1:indy2]

 var_clim = f1.variables[var_namec][:,indx1:indx2,indy1:indy2]
 var_p1 = f1.variables[var_name1][:,indx1:indx2,indy1:indy2]
 var_p2 = f1.variables[var_name2][:,indx1:indx2,indy1:indy2]
 var_p3 = f1.variables[var_name3][:,indx1:indx2,indy1:indy2]
 var_p4 = f1.variables[var_name4][:,indx1:indx2,indy1:indy2]
 var_p5 = f1.variables[var_name5][:,indx1:indx2,indy1:indy2]
 var_p6 = f1.variables[var_name6][:,indx1:indx2,indy1:indy2]

 return lon,lat, var_clim, var_p1,var_p2,var_p3,var_p4,var_p5,var_p6

#-----------------------------------------------------------------------
# function to write out MJO composites to nc files
#-----------------------------------------------------------------------

def write_mjo_nc( var_name, output_file, lon, lat, var_clim_ave, \
                  var_p1, var_p2, var_p3, var_p4, var_p5, var_p6, var_p7, var_p8,
                  vara_p1,vara_p2,vara_p3,vara_p4,vara_p5,vara_p6,vara_p7,vara_p8 ):

 # define var names
 var_namec = var_name+'_clim'
 var_name1 = var_name+'_p1'
 var_name2 = var_name+'_p2'
 var_name3 = var_name+'_p3'
 var_name4 = var_name+'_p4'
 var_name5 = var_name+'_p5'
 var_name6 = var_name+'_p6'
 var_name7 = var_name+'_p7'
 var_name8 = var_name+'_p8'

 vara_name1 = var_name+'a_p1'
 vara_name2 = var_name+'a_p2'
 vara_name3 = var_name+'a_p3'
 vara_name4 = var_name+'a_p4'
 vara_name5 = var_name+'a_p5'
 vara_name6 = var_name+'a_p6'
 vara_name7 = var_name+'a_p7'
 vara_name8 = var_name+'a_p8'

 fnc = Dataset(output_file, 'w',format='NETCDF4_CLASSIC')

 # create dim
 nx,ny = np.shape(var_p1)
 X = fnc.createDimension('X', nx)
 Y = fnc.createDimension('Y', ny)

 # create var 
 lon_mjo = fnc.createVariable('lon', np.float32, ('Y',))
 lat_mjo = fnc.createVariable('lat', np.float32, ('X',))
 
 var_mjo_clim  = fnc.createVariable(var_namec, np.float32, ('X', 'Y'))
 var_mjo_p1  = fnc.createVariable(var_name1, np.float32, ('X', 'Y'))
 var_mjo_p2  = fnc.createVariable(var_name2, np.float32, ('X', 'Y'))
 var_mjo_p3  = fnc.createVariable(var_name3, np.float32, ('X', 'Y'))
 var_mjo_p4  = fnc.createVariable(var_name4, np.float32, ('X', 'Y'))
 var_mjo_p5  = fnc.createVariable(var_name5, np.float32, ('X', 'Y'))
 var_mjo_p6  = fnc.createVariable(var_name6, np.float32, ('X', 'Y'))
 var_mjo_p7  = fnc.createVariable(var_name7, np.float32, ('X', 'Y'))
 var_mjo_p8  = fnc.createVariable(var_name8, np.float32, ('X', 'Y'))

 vara_mjo_p1  = fnc.createVariable(vara_name1, np.float32, ('X', 'Y'))
 vara_mjo_p2  = fnc.createVariable(vara_name2, np.float32, ('X', 'Y'))
 vara_mjo_p3  = fnc.createVariable(vara_name3, np.float32, ('X', 'Y'))
 vara_mjo_p4  = fnc.createVariable(vara_name4, np.float32, ('X', 'Y'))
 vara_mjo_p5  = fnc.createVariable(vara_name5, np.float32, ('X', 'Y'))
 vara_mjo_p6  = fnc.createVariable(vara_name6, np.float32, ('X', 'Y'))
 vara_mjo_p7  = fnc.createVariable(vara_name7, np.float32, ('X', 'Y'))
 vara_mjo_p8  = fnc.createVariable(vara_name8, np.float32, ('X', 'Y'))

 # write var
 lon_mjo[:] = lon
 lat_mjo[:] = lat

 var_mjo_clim[:,:] = var_clim_ave
 var_mjo_p1[:,:] = var_p1
 var_mjo_p2[:,:] = var_p2
 var_mjo_p3[:,:] = var_p3
 var_mjo_p4[:,:] = var_p4
 var_mjo_p5[:,:] = var_p5
 var_mjo_p6[:,:] = var_p6
 var_mjo_p7[:,:] = var_p7
 var_mjo_p8[:,:] = var_p8

 vara_mjo_p1[:,:] = vara_p1
 vara_mjo_p2[:,:] = vara_p2
 vara_mjo_p3[:,:] = vara_p3
 vara_mjo_p4[:,:] = vara_p4
 vara_mjo_p5[:,:] = vara_p5
 vara_mjo_p6[:,:] = vara_p6
 vara_mjo_p7[:,:] = vara_p7
 vara_mjo_p8[:,:] = vara_p8
 
 fnc.close()

#-----------------------------------------------------------------------
# function to read TC number at MJO phases
#-----------------------------------------------------------------------

def read_tc_mjo(file):

    f = open(file, "r")
    mjo_days=[]
    tc_num=[]

    for line in f:
        L = line.split()
        mjo_days += [float(L[0])]
        tc_num   += [float(L[1])]
    return mjo_days, tc_num


def read_tc_mjo_2(file):

    f = open(file, "r")
    mjo_days = []
    tc_num   = []
    h_num    = []
    ri_num   = []
    tse      = []
    he       = [] 

    for line in f:
        L = line.split()
        mjo_days += [int(L[0])]
        tc_num   += [int(L[1])]
        h_num    += [int(L[2])]
        ri_num   += [int(L[3])]
        tse      += [int(L[4])]
        he       += [int(L[5])]

    return mjo_days, tc_num, ri_num, h_num, tse, he

def read_tc_mjo_3(file):

    f = open(file, "r")
    mjo_days = []

    tc_num   = []
    h_num    = []
    mh_num   = []
    land_num   = []

    tse      = []
    he       = [] 
    lande       = []

    for line in f:
        L = line.split()

        mjo_days += [float(L[0])]

        tc_num   += [float(L[1])]
        h_num    += [float(L[2])]
        mh_num   += [float(L[3])]
        land_num += [float(L[4])]

        tse      += [float(L[5])]
        he       += [float(L[6])]
        lande    += [float(L[6])]

    return mjo_days, tc_num, h_num, mh_num, land_num, tse, he, lande


def read_tc_mjo_4(file):

    f = open(file, "r")
    mjo_days = []
    tse      = []
    he       = [] 
    mhe      = []

    for line in f:
        L = line.split()
        mjo_days += [float(L[0])]
        tse      += [float(L[1])]
        he       += [float(L[2])]
        mhe      += [float(L[3])]

    return np.array(mjo_days), np.array(tse), np.array(he), np.array(mhe)

#-----------------------------------------------------------------------
# function to get RMM1/2 amplitude and phase
#-----------------------------------------------------------------------

def get_amp_phase(pc1,pc2):
        
    nx=np.size(pc1)
    amp=np.zeros(nx)
    phase=np.zeros(nx)

    amp=np.sqrt(pc1**2+pc2**2)
       
    for i in np.arange(nx):

        if nx==1:
           x1=pc1
           x2=pc2
        else:
           x1=pc1[i]
           x2=pc2[i]

        ip = 0

        if x1>=0. and x2>=0.:
         if x1>=x2:
           ip=5
         else:
           ip=6

        elif x1>=0. and x2<=0.:
         if np.abs(x1)>=np.abs(x2):
           ip=4
         else:
           ip=3

        elif x1<=0. and x2<=0.:
          if np.abs(x1)>=np.abs(x2):
           ip=1
          else:
           ip=2

        else:
          if np.abs(x1)>=np.abs(x2):
           ip=8
          else:
           ip=7

        phase[i]=ip

    return amp, phase

#-----------------------------------------------------------------------
# function to get angle based on pc1, pc2
#-----------------------------------------------------------------------

def get_mjo_angle(pc1, pc2):
    pc1 = np.array(pc1)
    pc2 = np.array(pc2)
    mag = np.sqrt(pc1**2 + pc2**2)
    nrec = len(pc1)
    angle = np.zeros(nrec)
    for i in np.arange(nrec):
      if pc1[i] >=0:
         angle[i] = 180./np.pi * np.arcsin(pc2[i]/mag[i])
      else:
         angle[i] = 180. - 180./np.pi * np.arcsin(pc2[i]/mag[i])
    return angle

#-----------------------------------------------------------------------
# function to reconstruct anomaly fields based on 
# observed MJO EOF modes and extracted RMM1/2 indices 
#-----------------------------------------------------------------------

def mjo_reconstruct(mjo_file,eval1,eval2,olr_nf,u850_nf,u200_nf,pc1,pc2):

  nx = 144
  nt = len(pc1)

# step 1 - read mjo modes

  olr_eof1=np.zeros(nx)
  olr_eof2=np.zeros(nx)
  u850_eof1=np.zeros(nx)
  u850_eof2=np.zeros(nx)
  u200_eof1=np.zeros(nx)
  u200_eof2=np.zeros(nx)
  mjo_eof1=np.zeros(3*nx)
  mjo_eof2=np.zeros(3*nx)

  fmjo = open(mjo_file, "r")

  ind=0
  for line in fmjo:
    L = line.split()
    mjo_eof1[ind] = float(L[0])
    mjo_eof2[ind] = float(L[1])
    ind=ind+1

  olr_eof1=mjo_eof1[:nx]
  olr_eof2=mjo_eof2[:nx]
  u850_eof1=mjo_eof1[nx:2*nx]
  u850_eof2=mjo_eof2[nx:2*nx]
  u200_eof1=mjo_eof1[2*nx:]
  u200_eof2=mjo_eof2[2*nx:]

# step 2 - reconstruct the anomaly fields

  olr_rct=np.zeros((nt,nx))
  olr_rct_eof1=np.zeros((nt,nx))
  olr_rct_eof2=np.zeros((nt,nx))

  u850_rct=np.zeros((nt,nx))
  u850_rct_eof1=np.zeros((nt,nx))
  u850_rct_eof2=np.zeros((nt,nx))

  u200_rct=np.zeros((nt,nx))
  u200_rct_eof1=np.zeros((nt,nx))
  u200_rct_eof2=np.zeros((nt,nx))

  c1=pc1*np.sqrt(eval1)
  c2=pc2*np.sqrt(eval2)

  for t in np.arange(nt):
    olr_rct_eof1[t,:]=c1[t]*olr_eof1*olr_nf
    olr_rct_eof2[t,:]=c2[t]*olr_eof2*olr_nf
    u850_rct_eof1[t,:]=c1[t]*u850_eof1*u850_nf
    u850_rct_eof2[t,:]=c2[t]*u850_eof2*u850_nf
    u200_rct_eof1[t,:]=c1[t]*u200_eof1*u200_nf
    u200_rct_eof2[t,:]=c2[t]*u200_eof2*u200_nf

  olr_rct=olr_rct_eof1+olr_rct_eof2
  u850_rct=u850_rct_eof1+u850_rct_eof2
  u200_rct=u200_rct_eof1+u200_rct_eof2

  return olr_rct, u850_rct, u200_rct

#-----------------------------------------------------------------------
# function to read mjo all
#-----------------------------------------------------------------------

def read_mjo_all_nc(file, var_name, lon1, lon2, lat1, lat2):

 #var_namec = var_name+'_clim'
 var_name1 = var_name+'_p1_all'
 var_name2 = var_name+'_p2_all'
 var_name3 = var_name+'_p3_all'
 var_name4 = var_name+'_p4_all'
 var_name5 = var_name+'_p5_all'
 var_name6 = var_name+'_p6_all'
 var_name7 = var_name+'_p7_all'
 var_name8 = var_name+'_p8_all'

 f1 = Dataset(file, 'r')

 lon_all = f1.variables['lon'][:]
 lat_all = f1.variables['lat'][:]

 if lat_all[1] > lat_all[0]:
    indx1 = find_nearest(lat_all,lat1)
    indx2 = find_nearest(lat_all,lat2)+1
 else:
    indx1 = find_nearest(lat_all,lat2)
    indx2 = find_nearest(lat_all,lat1)+1

 indy1 = find_nearest(lon_all,lon1)
 indy2 = find_nearest(lon_all,lon2)+1

 lat = lat_all[indx1:indx2]
 lon = lon_all[indy1:indy2]

 #var_clim = f1.variables[var_namec][:,indx1:indx2,indy1:indy2]
 var_p1 = f1.variables[var_name1][:,indx1:indx2,indy1:indy2]
 var_p2 = f1.variables[var_name2][:,indx1:indx2,indy1:indy2]
 var_p3 = f1.variables[var_name3][:,indx1:indx2,indy1:indy2]
 var_p4 = f1.variables[var_name4][:,indx1:indx2,indy1:indy2]
 var_p5 = f1.variables[var_name5][:,indx1:indx2,indy1:indy2]
 var_p6 = f1.variables[var_name6][:,indx1:indx2,indy1:indy2]
 var_p7 = f1.variables[var_name7][:,indx1:indx2,indy1:indy2]
 var_p8 = f1.variables[var_name8][:,indx1:indx2,indy1:indy2]

 return lon,lat, var_p1,var_p2,var_p3,var_p4,var_p5,var_p6,var_p7,var_p8

# with clim
def read_mjo_all_nc_2(file, var_name, lon1, lon2, lat1, lat2):

 var_namec = var_name+'_clim'
 var_name1 = var_name+'_p1_all'
 var_name2 = var_name+'_p2_all'
 var_name3 = var_name+'_p3_all'
 var_name4 = var_name+'_p4_all'
 var_name5 = var_name+'_p5_all'
 var_name6 = var_name+'_p6_all'
 var_name7 = var_name+'_p7_all'
 var_name8 = var_name+'_p8_all'

 f1 = Dataset(file, 'r')

 lon_all = f1.variables['lon'][:]
 lat_all = f1.variables['lat'][:]

 if lat_all[1] > lat_all[0]:
    indx1 = find_nearest(lat_all,lat1)
    indx2 = find_nearest(lat_all,lat2)+1
 else:
    indx1 = find_nearest(lat_all,lat2)
    indx2 = find_nearest(lat_all,lat1)+1

 indy1 = find_nearest(lon_all,lon1)
 indy2 = find_nearest(lon_all,lon2)+1

 lat = lat_all[indx1:indx2]
 lon = lon_all[indy1:indy2]

 var_clim = f1.variables[var_namec][:,indx1:indx2,indy1:indy2]
 var_p1 = f1.variables[var_name1][:,indx1:indx2,indy1:indy2]
 var_p2 = f1.variables[var_name2][:,indx1:indx2,indy1:indy2]
 var_p3 = f1.variables[var_name3][:,indx1:indx2,indy1:indy2]
 var_p4 = f1.variables[var_name4][:,indx1:indx2,indy1:indy2]
 var_p5 = f1.variables[var_name5][:,indx1:indx2,indy1:indy2]
 var_p6 = f1.variables[var_name6][:,indx1:indx2,indy1:indy2]
 var_p7 = f1.variables[var_name7][:,indx1:indx2,indy1:indy2]
 var_p8 = f1.variables[var_name8][:,indx1:indx2,indy1:indy2]

 return lon,lat, var_clim, var_p1,var_p2,var_p3,var_p4,var_p5,var_p6,var_p7,var_p8

#-----------------------------------------------------------------------
# function to plot MJO indices
#-----------------------------------------------------------------------

def plot_mjo_index(pc1,pc2,amp,phase):

  ft=16
  ft1=16
  time = np.arange(len(pc1))+1

# RMM 1/2

  ax = plt.subplot(221)
  plt.plot(time,pc1, 'k-',  linewidth=3, label='RMM1')
  plt.plot(time,pc2, 'k--', linewidth=3, label='RMM2')
  plt.plot(np.zeros(len(pc1)), 'k', linewidth=1)
    
  ax.grid(True)
  ax.legend(loc='upper right',frameon=False)
  ax.set_ylabel('Normalized Amplitude',fontsize=ft)

  ax.set_ylim(-6,6)
  ax.set_yticks(np.arange(-6,7,1))
  ax.set_xlim(min(time),max(time))
  #ax.set_xticks([1,16,32,47,63,78,93,108,124,139])
  #ax.set_xticklabels(['07-01','07-15','08-01','08-15','09-01','09-15','10-01','10-15','11-01','11-15'],rotation=45)

  for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(ft1)
  for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(ft1)

# phase space

  ax = plt.subplot(222)
  plt.plot(pc1, pc2, 'k-',  linewidth=1)
  plt.plot(pc1[::5],pc2[::5], 'ko')
  #plt.plot(pc1[0], pc2[0],    'ro')
  #plt.plot(pc1[32],pc2[32],   'go')
  #plt.plot(pc1[63],pc2[63],   'bo')
  #plt.plot(pc1[93],pc2[93],   'co')
  #plt.plot(pc1[124],pc2[124], 'yo')
    
  ax.grid(True)
  ax.set_xlabel('RMM1',fontsize=ft)
  ax.set_ylabel('RMM2',fontsize=ft)
   
  plt.axis('equal')
  plt.axis([-4, 4, -4, 4])
  ax.set_xticks(np.arange(-4,5))
  ax.set_yticks(np.arange(-4,5))
  
  for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(ft1)
  for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(ft1)

## amp

  ax = plt.subplot(223)
  plt.plot(time,amp, 'k-',  linewidth=3)
  plt.plot(np.zeros(len(pc1))+1, 'k', linewidth=1)
    
  ax.grid(True)
  ax.set_ylabel('Normalized Amplitude',fontsize=ft)

  ax.set_ylim(0,6)
  ax.set_yticks(np.arange(0,7,1))
  ax.set_xlim(min(time),max(time))
  #ax.set_xticks([1,16,32,47,63,78,93,108,124,139])
  #ax.set_xticklabels(['07-01','07-15','08-01','08-15','09-01','09-15','10-01','10-15','11-01','11-15'],rotation=45)

  for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(ft1)
  for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(ft1)

## phase

  ax = plt.subplot(224)
  plt.plot(time,phase, 'k-',  linewidth=1)
  plt.plot(time,phase, 'ko',  linewidth=3)
    
  ax.grid(True)
  ax.set_ylabel('Phase',fontsize=ft)

  ax.set_ylim(0,9)
  ax.set_yticks(np.arange(1,9,1))
  ax.set_xlim(min(time),max(time))
  #ax.set_xticks([1,16,32,47,63,78,93,108,124,139])
  #ax.set_xticklabels(['07-01','07-15','08-01','08-15','09-01','09-15','10-01','10-15','11-01','11-15'],rotation=45)

  for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(ft1)
  for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(ft1)

#-----------------------------------------------------------------------
# function to plot tc tracks for 8 MJO phases
#-----------------------------------------------------------------------

def plot_tracks_mjo_phases(basin,sel,\
                       lon_p1_all, lon_p2_all, lon_p3_all, lon_p4_all,\
                       lon_p5_all, lon_p6_all, lon_p7_all, lon_p8_all,\
                       lat_p1_all, lat_p2_all, lat_p3_all, lat_p4_all,\
                       lat_p5_all, lat_p6_all, lat_p7_all, lat_p8_all,\
                       wind_p1_all,wind_p2_all,wind_p3_all,wind_p4_all,\
                       wind_p5_all,wind_p6_all,wind_p7_all,wind_p8_all,\
                       t1,t2,t3,t4,t5,t6,t7,t8):

#--- p1
 ax1 = plt.subplot(818) 
 m1 = setup_m(basin)
 ax1.set_title(t1)

 for i in np.arange(0,len(lon_p1_all),sel):

   all_lon = lon_p1_all[i] 
   all_lat = lat_p1_all[i]
   all_wind= wind_p1_all[i]
   plot_track_simple(m1, all_lon, all_lat, all_wind)

#--- p2
 ax2 = plt.subplot(811) 
 m2 = setup_m(basin)
 ax2.set_title(t2)

 for i in np.arange(0,len(lon_p2_all),sel):
   all_lon = lon_p2_all[i] 
   all_lat = lat_p2_all[i]
   all_wind= wind_p2_all[i]
   plot_track_simple(m2, all_lon, all_lat, all_wind)

#--- p3
 ax3 = plt.subplot(812) 
 m3 = setup_m(basin)
 ax3.set_title(t3)

 for i in np.arange(0,len(lon_p3_all),sel):
   all_lon = lon_p3_all[i] 
   all_lat = lat_p3_all[i]
   all_wind= wind_p3_all[i]
   plot_track_simple(m3, all_lon, all_lat, all_wind)

#--- p4
 ax4 = plt.subplot(813) 
 m4 = setup_m(basin)
 ax4.set_title(t4)

 for i in np.arange(0,len(lon_p4_all),sel):
   all_lon = lon_p4_all[i] 
   all_lat = lat_p4_all[i]
   all_wind= wind_p4_all[i]
   plot_track_simple(m4, all_lon, all_lat, all_wind)

#--- p5
 ax5 = plt.subplot(814) 
 m5 = setup_m(basin)
 ax5.set_title(t5)

 for i in np.arange(0,len(lon_p5_all),sel):
   all_lon = lon_p5_all[i] 
   all_lat = lat_p5_all[i]
   all_wind= wind_p5_all[i]
   plot_track_simple(m5, all_lon, all_lat, all_wind)

#--- p6
 ax6 = plt.subplot(815) 
 m6 = setup_m(basin)
 ax6.set_title(t6)

 for i in np.arange(0,len(lon_p6_all),sel):
   all_lon = lon_p6_all[i] 
   all_lat = lat_p6_all[i]
   all_wind= wind_p6_all[i]
   plot_track_simple(m6, all_lon, all_lat, all_wind)

#--- p7
 ax7 = plt.subplot(816) 
 m7 = setup_m(basin)
 ax7.set_title(t7)

 for i in np.arange(0,len(lon_p7_all),sel):
   all_lon = lon_p7_all[i] 
   all_lat = lat_p7_all[i]
   all_wind= wind_p7_all[i]
   plot_track_simple(m7, all_lon, all_lat, all_wind)

#--- p8
 ax8 = plt.subplot(817) 
 m8 = setup_m(basin)
 ax8.set_title(t8)

 for i in np.arange(0,len(lon_p8_all),sel):
   all_lon = lon_p8_all[i] 
   all_lat = lat_p8_all[i]
   all_wind= wind_p8_all[i]
   plot_track_simple(m8, all_lon, all_lat, all_wind)

## version 1.5

def plot_tracks_mjo_phases_2p(basin,sel,\
                       lon_pa_all, lon_pb_all, 
                       lat_pa_all, lat_pb_all, 
                       wind_pa_all,wind_pb_all,
                       ta,tb):

#--- p1
 ax1 = plt.subplot(211) 
 m1 = setup_m(basin)
 ax1.set_title(ta,fontsize=16)

 for i in np.arange(0,len(lon_pa_all),sel):

   all_lon = lon_pa_all[i] 
   all_lat = lat_pa_all[i]
   all_wind= wind_pa_all[i]
   plot_track_simple(m1, all_lon, all_lat, all_wind)

#--- p2
 ax2 = plt.subplot(212) 
 m2 = setup_m(basin)
 ax2.set_title(tb,fontsize=16)

 for i in np.arange(0,len(lon_pb_all),sel):
   all_lon = lon_pb_all[i] 
   all_lat = lat_pb_all[i]
   all_wind= wind_pb_all[i]
   plot_track_simple(m2, all_lon, all_lat, all_wind)


def plot_tracks_mjo_phases_3p(basin,sel,\
                       lon_pa_all, lon_pb_all, lon_pn_all,
                       lat_pa_all, lat_pb_all, lat_pn_all,
                       wind_pa_all,wind_pb_all, wind_pn_all,
                       ta,tb,tn):

#--- p1
 ax1 = plt.subplot(311) 
 m1 = setup_m(basin)
 ax1.set_title(ta,fontsize=16)

 for i in np.arange(0,len(lon_pa_all),sel):

   all_lon = lon_pa_all[i] 
   all_lat = lat_pa_all[i]
   all_wind= wind_pa_all[i]
   plot_track_simple(m1, all_lon, all_lat, all_wind)

#--- p2
 ax2 = plt.subplot(312) 
 m2 = setup_m(basin)
 ax2.set_title(tb,fontsize=16)

 for i in np.arange(0,len(lon_pb_all),sel):
   all_lon = lon_pb_all[i] 
   all_lat = lat_pb_all[i]
   all_wind= wind_pb_all[i]
   plot_track_simple(m2, all_lon, all_lat, all_wind)

#--- p3
 ax3 = plt.subplot(313) 
 m3 = setup_m(basin)
 ax3.set_title(tn,fontsize=16)

 for i in np.arange(0,len(lon_pn_all),sel):
   all_lon = lon_pn_all[i] 
   all_lat = lat_pn_all[i]
   all_wind= wind_pn_all[i]
   plot_track_simple(m2, all_lon, all_lat, all_wind)


def plot_tracks_mjo_phases_2p_land(basin,sel,\
                       lon_pa_all, lon_pb_all,
                       lat_pa_all, lat_pb_all,
                       wind_pa_all,wind_pb_all,
                       ta,tb):

#--- p1
 ax1 = plt.subplot(211)
 m1 = setup_m(basin)
 ax1.set_title(ta,fontsize=16)

 for i in np.arange(0,len(lon_pa_all),sel):

   all_lon = lon_pa_all[i]
   all_lat = lat_pa_all[i]
   all_wind= wind_pa_all[i]
   plot_track_land(m1, all_lon, all_lat, all_wind)

#--- p2
 ax2 = plt.subplot(212)
 m2 = setup_m(basin)
 ax2.set_title(tb,fontsize=16)

 for i in np.arange(0,len(lon_pb_all),sel):
   all_lon = lon_pb_all[i]
   all_lat = lat_pb_all[i]
   all_wind= wind_pb_all[i]
   plot_track_land(m2, all_lon, all_lat, all_wind)

## version 2

def plot_tracks_mjo_phases_2(basin,sel,\
                       lon_p1_all, lon_p2_all, lon_p3_all, lon_p4_all,\
                       lon_p5_all, lon_p6_all, lon_p7_all, lon_p8_all,\
                       lat_p1_all, lat_p2_all, lat_p3_all, lat_p4_all,\
                       lat_p5_all, lat_p6_all, lat_p7_all, lat_p8_all,\
                       wind_p1_all,wind_p2_all,wind_p3_all,wind_p4_all,\
                       wind_p5_all,wind_p6_all,wind_p7_all,wind_p8_all,\
                       date_p1_all,date_p2_all,date_p3_all,date_p4_all,\
                       date_p5_all,date_p6_all,date_p7_all,date_p8_all,\
                       t1,t2,t3,t4,t5,t6,t7,t8):

#--- p1
 ax1 = plt.subplot(241) 
 m1 = setup_m(basin)
 ax1.set_title(t1)

 for i in np.arange(0,len(lon_p1_all),sel):

   all_lon = lon_p1_all[i] 
   all_lat = lat_p1_all[i]
   all_wind= wind_p1_all[i]
   all_date= date_p1_all[i] 
   plot_track(m1, all_lon, all_lat, all_wind, all_date, 'b')

#--- p2
 ax2 = plt.subplot(242) 
 m2 = setup_m(basin)
 ax2.set_title(t2)

 for i in np.arange(0,len(lon_p2_all),sel):
   all_lon = lon_p2_all[i] 
   all_lat = lat_p2_all[i]
   all_wind= wind_p2_all[i]
   all_date= date_p2_all[i]
   plot_track(m2, all_lon, all_lat, all_wind, all_date, 'b')

#--- p3
 ax3 = plt.subplot(243) 
 m3 = setup_m(basin)
 ax3.set_title(t3)

 for i in np.arange(0,len(lon_p3_all),sel):
   all_lon = lon_p3_all[i] 
   all_lat = lat_p3_all[i]
   all_wind= wind_p3_all[i]
   all_date= date_p3_all[i] 
   plot_track(m3, all_lon, all_lat, all_wind, all_date, 'b')

#--- p4
 ax4 = plt.subplot(244) 
 m4 = setup_m(basin)
 ax4.set_title(t4)

 for i in np.arange(0,len(lon_p4_all),sel):
   all_lon = lon_p4_all[i] 
   all_lat = lat_p4_all[i]
   all_wind= wind_p4_all[i]
   all_date= date_p4_all[i] 
   plot_track(m4, all_lon, all_lat, all_wind, all_date, 'b')

#--- p5
 ax5 = plt.subplot(245) 
 m5 = setup_m(basin)
 ax5.set_title(t5)

 for i in np.arange(0,len(lon_p5_all),sel):
   all_lon = lon_p5_all[i] 
   all_lat = lat_p5_all[i]
   all_wind= wind_p5_all[i]
   all_date= date_p5_all[i] 
   plot_track(m5, all_lon, all_lat, all_wind, all_date, 'b')

#--- p6
 ax6 = plt.subplot(246) 
 m6 = setup_m(basin)
 ax6.set_title(t6)

 for i in np.arange(0,len(lon_p6_all),sel):
   all_lon = lon_p6_all[i] 
   all_lat = lat_p6_all[i]
   all_wind= wind_p6_all[i]
   all_date= date_p6_all[i] 
   plot_track(m6, all_lon, all_lat, all_wind, all_date, 'b')

#--- p7
 ax7 = plt.subplot(247) 
 m7 = setup_m(basin)
 ax7.set_title(t7)

 for i in np.arange(0,len(lon_p7_all),sel):
   all_lon = lon_p7_all[i] 
   all_lat = lat_p7_all[i]
   all_wind= wind_p7_all[i]
   all_date= date_p7_all[i] 
   plot_track(m7, all_lon, all_lat, all_wind, all_date, 'b')

#--- p8
 ax8 = plt.subplot(248) 
 m8 = setup_m(basin)
 ax8.set_title(t8)

 for i in np.arange(0,len(lon_p8_all),sel):
   all_lon = lon_p8_all[i] 
   all_lat = lat_p8_all[i]
   all_wind= wind_p8_all[i]
   all_date= date_p8_all[i] 
   plot_track(m8, all_lon, all_lat, all_wind, all_date, 'b')

#-----------------------------------------------------------------------
# function to make plots for 1D composite in 8 MJO phases
#-----------------------------------------------------------------------

#
# ----------- all in one
#

def plot_1d_mjo_phases(lon,var_clim,
       var_p1,var_p2,var_p3,var_p4,\
       var_p5,var_p6,var_p7,var_p8,\
       var_name, xticks, yticks):

 ft=16

 xmin = np.min(xticks)
 xmax = np.max(xticks)
 ymin = np.min(yticks)
 ymax = np.max(yticks)

 col = ['#0000ff','#1e90ff','#00bfff','#87ceeb',\
       	'#ff0000','#ff4500','#ff6347','#f08080']

 ax = plt.subplot(111)

 ax.plot(lon, var_clim, 'k--',        label= 'clim',    linewidth=2)
 ax.plot(lon, var_p1,   color=col[0], label= 'phase 1', linewidth=2)
 ax.plot(lon, var_p2,   color=col[1], label= 'phase 2', linewidth=2)
 ax.plot(lon, var_p3,   color=col[2], label= 'phase 3', linewidth=2)
 ax.plot(lon, var_p4,   color=col[3], label= 'phase 4', linewidth=2)
 ax.plot(lon, var_p5,   color=col[4], label= 'phase 5', linewidth=2)
 ax.plot(lon, var_p6,   color=col[5], label= 'phase 6', linewidth=2)
 ax.plot(lon, var_p7,   color=col[6], label= 'phase 7', linewidth=2)
 ax.plot(lon, var_p8,   color=col[7], label= 'phase 8', linewidth=2)

 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title(var_name,fontsize=ft)

#
# ----------- 8p with clim
#

def plot_1d_mjo_phases_8p(lon,var_clim,
       var_p1,var_p2,var_p3,var_p4,\
       var_p5,var_p6,var_p7,var_p8,\
       var_name, xticks, yticks):

 ft=16

 xmin = np.min(xticks)
 xmax = np.max(xticks)
 ymin = np.min(yticks)
 ymax = np.max(yticks)

# ------- phase 1
 ax = plt.subplot(241)

 ax.plot(lon, var_clim, 'k--',label= var_name + ' clim', linewidth=2)
 ax.plot(lon, var_p1,   'k',  label= var_name,           linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('a) Phase 1',fontsize=ft)


# ------- phase 2
 ax = plt.subplot(242)

 ax.plot(lon, var_clim, 'k--',label= var_name + ' clim', linewidth=2)
 ax.plot(lon, var_p2,   'k',  label= var_name,           linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('b) Phase 2',fontsize=ft)


# ------- phase 3
 ax = plt.subplot(243)

 ax.plot(lon, var_clim, 'k--',label= var_name + ' clim', linewidth=2)
 ax.plot(lon, var_p3,   'k',  label= var_name,           linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('c) Phase 3',fontsize=ft)


# ------- phase 4
 ax = plt.subplot(244)

 ax.plot(lon, var_clim, 'k--',label= var_name + ' clim', linewidth=2)
 ax.plot(lon, var_p4,   'k',  label= var_name,           linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('d) Phase 4',fontsize=ft)


# ------- phase 5
 ax = plt.subplot(245)

 ax.plot(lon, var_clim, 'k--',label= var_name + ' clim', linewidth=2)
 ax.plot(lon, var_p5,   'k',  label= var_name,           linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('e) Phase 5',fontsize=ft)


# ------- phase 6
 ax = plt.subplot(246)

 ax.plot(lon, var_clim, 'k--',label= var_name + ' clim', linewidth=2)
 ax.plot(lon, var_p6,   'k',  label= var_name,           linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('f) Phase 6',fontsize=ft)


# ------- phase 7
 ax = plt.subplot(247)

 ax.plot(lon, var_clim, 'k--',label= var_name + ' clim', linewidth=2)
 ax.plot(lon, var_p7,   'k',  label= var_name,           linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('g) Phase 7',fontsize=ft)


# ------- phase 8
 ax = plt.subplot(248)

 ax.plot(lon, var_clim, 'k--',label= var_name + ' clim', linewidth=2)
 ax.plot(lon, var_p8,   'k',  label= var_name,           linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('h) Phase 8',fontsize=ft)

#
# ----------- 8p - no clim
#

def plot_1d_mjo_phases_8p_2(lon,
       var_p1,var_p2,var_p3,var_p4,\
       var_p5,var_p6,var_p7,var_p8,\
       var_name, xticks, yticks):

 ft=16

 xmin = np.min(xticks)
 xmax = np.max(xticks)
 ymin = np.min(yticks)
 ymax = np.max(yticks)

# ------- phase 1
 ax = plt.subplot(241)

# ax.plot(lon, var_clim, 'k--',label= var_name + ' clim', linewidth=2)
 ax.plot(lon, var_p1,   'k',  label= var_name,           linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('a) Phase 1',fontsize=ft)


# ------- phase 2
 ax = plt.subplot(242)

# ax.plot(lon, var_clim, 'k--',label= var_name + ' clim', linewidth=2)
 ax.plot(lon, var_p2,   'k',  label= var_name,           linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('b) Phase 2',fontsize=ft)


# ------- phase 3
 ax = plt.subplot(243)

# ax.plot(lon, var_clim, 'k--',label= var_name + ' clim', linewidth=2)
 ax.plot(lon, var_p3,   'k',  label= var_name,           linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('c) Phase 3',fontsize=ft)


# ------- phase 4
 ax = plt.subplot(244)

# ax.plot(lon, var_clim, 'k--',label= var_name + ' clim', linewidth=2)
 ax.plot(lon, var_p4,   'k',  label= var_name,           linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('d) Phase 4',fontsize=ft)


# ------- phase 5
 ax = plt.subplot(245)

# ax.plot(lon, var_clim, 'k--',label= var_name + ' clim', linewidth=2)
 ax.plot(lon, var_p5,   'k',  label= var_name,           linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('e) Phase 5',fontsize=ft)


# ------- phase 6
 ax = plt.subplot(246)

# ax.plot(lon, var_clim, 'k--',label= var_name + ' clim', linewidth=2)
 ax.plot(lon, var_p6,   'k',  label= var_name,           linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('f) Phase 6',fontsize=ft)


# ------- phase 7
 ax = plt.subplot(247)

# ax.plot(lon, var_clim, 'k--',label= var_name + ' clim', linewidth=2)
 ax.plot(lon, var_p7,   'k',  label= var_name,           linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('g) Phase 7',fontsize=ft)


# ------- phase 8
 ax = plt.subplot(248)

# ax.plot(lon, var_clim, 'k--',label= var_name + ' clim', linewidth=2)
 ax.plot(lon, var_p8,   'k',  label= var_name,           linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('h) Phase 8',fontsize=ft)

#
# ----------- 8p obs_vs_mod
#

def plot_1d_mjo_phases_8p_obs_vs_mod(lon,
       var_p1,var_p2,var_p3,var_p4,\
       var_p5,var_p6,var_p7,var_p8,\
       var2_p1,var2_p2,var2_p3,var2_p4,\
       var2_p5,var2_p6,var2_p7,var2_p8,\
       var_name, xticks, yticks):

 ft=16

 xmin = np.min(xticks)
 xmax = np.max(xticks)
 ymin = np.min(yticks)
 ymax = np.max(yticks)

# ------- phase 1
 ax = plt.subplot(241)

 ax.plot(lon, var_p1,   'k',  label= var_name + ' - obs',   linewidth=2)
 ax.plot(lon, var2_p1,  'k--',label= var_name + ' - mod',   linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('a) Phase 1',fontsize=ft)


# ------- phase 2
 ax = plt.subplot(242)

 ax.plot(lon, var_p2,   'k',  label= var_name + ' - obs',   linewidth=2)
 ax.plot(lon, var2_p2,  'k--',label= var_name + ' - mod',   linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('b) Phase 2',fontsize=ft)


# ------- phase 3
 ax = plt.subplot(243)

 ax.plot(lon, var_p3,   'k',  label= var_name + ' - obs',   linewidth=2)
 ax.plot(lon, var2_p3,  'k--',label= var_name + ' - mod',   linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('c) Phase 3',fontsize=ft)


# ------- phase 4
 ax = plt.subplot(244)

 ax.plot(lon, var_p4,   'k',  label= var_name + ' - obs',   linewidth=2)
 ax.plot(lon, var2_p4,  'k--',label= var_name + ' - mod',   linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('d) Phase 4',fontsize=ft)


# ------- phase 5
 ax = plt.subplot(245)

 ax.plot(lon, var_p5,   'k',  label= var_name + ' - obs',   linewidth=2)
 ax.plot(lon, var2_p5,  'k--',label= var_name + ' - mod',   linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('e) Phase 5',fontsize=ft)


# ------- phase 6
 ax = plt.subplot(246)

 ax.plot(lon, var_p6,   'k',  label= var_name + ' - obs',   linewidth=2)
 ax.plot(lon, var2_p6,  'k--',label= var_name + ' - mod',   linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('f) Phase 6',fontsize=ft)


# ------- phase 7
 ax = plt.subplot(247)

 ax.plot(lon, var_p7,   'k',  label= var_name + ' - obs',   linewidth=2)
 ax.plot(lon, var2_p7,  'k--',label= var_name + ' - mod',   linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('g) Phase 7',fontsize=ft)


# ------- phase 8
 ax = plt.subplot(248)

 ax.plot(lon, var_p8,   'k',  label= var_name + ' - obs',   linewidth=2)
 ax.plot(lon, var2_p8,  'k--',label= var_name + ' - mod',   linewidth=2)
 ax.plot(lon, np.zeros(len(lon)),'k-')

 ax.grid(True)
 ax.legend(loc='lower left',frameon=False)

# set y axis
 ax.set_ylim(ymin,ymax)
 ax.set_yticks(yticks)

# set x axis
 ax.set_xlim(xmin,xmax)
 ax.set_xticks(xticks)
 ax.set_xlabel('Lon',fontsize=ft)
 ax.set_title('h) Phase 8',fontsize=ft)

#-----------------------------------------------------------------------
# function to make contour plots for 8 MJO phases
#-----------------------------------------------------------------------

def make_contourfs_mjo_8p(basin,lonm,latm,\
                              var_p1,var_p2,var_p3,var_p4,\
                              var_p5,var_p6,var_p7,var_p8,\
                              t1,t2,t3,t4,t5,t6,t7,t8,cmap,\
                              lev,levc):

 cmin = np.min(lev)
 cmax = np.max(lev)

 ft = 16

# ------- phase 1
 ax1 = plt.subplot(811)
 m = setup_m(basin)
 m.contourf(lonm, latm, var_p1, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax1)
 plt.colorbar(cax = cax, ticks=levc)
 plt.clim([cmin,cmax])

# ------- phase 2
 ax2 = plt.subplot(812)
 m = setup_m(basin)
 m.contourf(lonm, latm, var_p2, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax2)
 plt.colorbar(cax = cax, ticks=levc)
 plt.clim([cmin,cmax])

# ------- phase 3
 ax3 = plt.subplot(813)
 m = setup_m(basin)
 m.contourf(lonm, latm, var_p3, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax3)
 plt.colorbar(cax = cax, ticks=levc)
 plt.clim([cmin,cmax])

# ------- phase 4
 ax4 = plt.subplot(814)
 m = setup_m(basin)
 m.contourf(lonm, latm, var_p4, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax4)
 plt.colorbar(cax = cax, ticks=levc)
 plt.clim([cmin,cmax])

# ------- phase 5
 ax5 = plt.subplot(815)
 m = setup_m(basin)
 m.contourf(lonm, latm, var_p5, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax5)
 plt.colorbar(cax = cax, ticks=levc)
 plt.clim([cmin,cmax])

# ------- phase 6
 ax6 = plt.subplot(816)
 m = setup_m(basin)
 m.contourf(lonm, latm, var_p6, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax6)
 plt.colorbar(cax = cax, ticks=levc)
 plt.clim([cmin,cmax])

# ------- phase 7
 ax7 = plt.subplot(817)
 m = setup_m(basin)
 m.contourf(lonm, latm, var_p7, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax7)
 plt.colorbar(cax = cax, ticks=levc)
 plt.clim([cmin,cmax])

# ------- phase 8
 ax8 = plt.subplot(818)
 m = setup_m(basin)
 m.contourf(lonm, latm, var_p8, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax8)
 plt.colorbar(cax = cax, ticks=levc)
 plt.clim([cmin,cmax])

 ax1.set_title(t1,fontsize = ft)
 ax2.set_title(t2,fontsize = ft)
 ax3.set_title(t3,fontsize = ft)
 ax4.set_title(t4,fontsize = ft)
 ax5.set_title(t5,fontsize = ft)
 ax6.set_title(t6,fontsize = ft)
 ax7.set_title(t7,fontsize = ft)
 ax8.set_title(t8,fontsize = ft)


def make_contourfs_mjo_2p(basin,lonm,latm,\
                              var_p1,var_p2,\
                              t1,t2,cmap,\
                              lev,levc):

 cmin = np.min(lev)
 cmax = np.max(lev)

 ft = 16

# ------- phase 1
 ax1 = plt.subplot(211)
 m = setup_m(basin)
 m.contourf(lonm, latm, var_p1, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax1)
 plt.colorbar(cax = cax, ticks=levc)
 plt.clim([cmin,cmax])

# ------- phase 2
 ax2 = plt.subplot(212)
 m = setup_m(basin)
 m.contourf(lonm, latm, var_p2, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax2)
 plt.colorbar(cax = cax, ticks=levc)
 plt.clim([cmin,cmax])

 ax1.set_title(t1,fontsize = ft)
 ax2.set_title(t2,fontsize = ft)

#-----------------------------------------------------------------------
# function to make contour plots for comparing obs and mod
#-----------------------------------------------------------------------

## 4 phases

def make_contourfs_mjo_mod_vs_obs_4p(basin,lono,lato,lonm,latm,\
                              var_p1,var_p2,var_p3,var_p4,\
                              var_p5,var_p6,var_p7,var_p8,\
                              t1,t2,t3,t4,t5,t6,t7,t8,cmap,\
                              lev,levc,ft1,ft2,ft3):

 cmin = np.min(lev)
 cmax = np.max(lev)

# ------- obs 1p
 ax = plt.subplot(421)
 m = setup_m(basin, True, '0.8', ft1)
 m.contourf(lono, lato, var_p1, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax)
 cbar=plt.colorbar(cax = cax, ticks=levc)
 cbar.ax.tick_params(labelsize=ft3)
 plt.clim([cmin,cmax])
 ax.set_title(t1,fontsize=ft2)

# ------- obs 2p 
 ax = plt.subplot(423)
 m = setup_m(basin, True, '0.8', ft1)
 m.contourf(lono, lato, var_p2, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax)
 cbar=plt.colorbar(cax = cax, ticks=levc)
 cbar.ax.tick_params(labelsize=ft3)
 plt.clim([cmin,cmax])
 ax.set_title(t2,fontsize=ft2)

# ------- obs 3p
 ax = plt.subplot(425)
 m = setup_m(basin, True, '0.8', ft1)
 m.contourf(lono, lato, var_p3, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax)
 cbar=plt.colorbar(cax = cax, ticks=levc)
 cbar.ax.tick_params(labelsize=ft3)
 plt.clim([cmin,cmax])
 ax.set_title(t3,fontsize=ft2)

# ------- obs 4p
 ax = plt.subplot(427)
 m = setup_m(basin, True, '0.8', ft1)
 m.contourf(lono, lato, var_p4, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax)
 cbar=plt.colorbar(cax = cax, ticks=levc)
 cbar.ax.tick_params(labelsize=ft3)
 plt.clim([cmin,cmax])
 ax.set_title(t4,fontsize=ft2)

# ------- model 1p
 ax = plt.subplot(422)
 m = setup_m(basin, True, '0.8', ft1)
 m.contourf(lonm, latm, var_p5, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax)
 cbar=plt.colorbar(cax = cax, ticks=levc)
 cbar.ax.tick_params(labelsize=ft3)
 plt.clim([cmin,cmax])
 ax.set_title(t5,fontsize=ft2)

# ------- model 2p
 ax = plt.subplot(424)
 m = setup_m(basin, True, '0.8', ft1)
 m.contourf(lonm, latm, var_p6, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax)
 cbar=plt.colorbar(cax = cax, ticks=levc)
 cbar.ax.tick_params(labelsize=ft3)
 plt.clim([cmin,cmax])
 ax.set_title(t6,fontsize=ft2)

# ------- model 3p
 ax = plt.subplot(426)
 m = setup_m(basin, True, '0.8', ft1)
 m.contourf(lonm, latm, var_p7, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax)
 cbar=plt.colorbar(cax = cax, ticks=levc)
 cbar.ax.tick_params(labelsize=ft3)
 plt.clim([cmin,cmax])
 ax.set_title(t7,fontsize=ft2)

# ------- model 4p
 ax = plt.subplot(428)
 m = setup_m(basin, True, '0.8', ft1)
 m.contourf(lonm, latm, var_p8, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax)
 cbar=plt.colorbar(cax = cax, ticks=levc)
 cbar.ax.tick_params(labelsize=ft3)
 plt.clim([cmin,cmax])
 ax.set_title(t8,fontsize=ft2)


## 1 phase
def make_contourfs_mjo_1p(basin,lonm,latm,\
                          var_p1, t1, cmap, lev, levc):

 cmin = np.min(lev)
 cmax = np.max(lev)

 ax = plt.subplot(111)
 m = setup_m(basin)
 m.contourf(lonm, latm, var_p1, cmap=cmap,levels=lev,extend="both")

 cax = get_cax(ax)
 plt.colorbar(cax = cax, ticks=levc)
 plt.clim([cmin,cmax])
 ax.set_title(t1, fontsize = 18)

## 1 phase
def make_contourfs_mjo_mod_vs_obs_1p(a,b,i,basin,lono,lato,lonm,latm,\
                              var_p1,var_p2, \
                              t1,t2,cmap,\
                              lev,levc,ft,ft1):

 cmin = np.min(lev)
 cmax = np.max(lev)

 ax = plt.subplot(a,b,i) 
 m = setup_m(basin)
 m.contourf(lono, lato, var_p1, cmap=cmap,levels=lev,extend="both")

 plt.clim([cmin,cmax])
 ax.set_title(t1,fontsize=ft)

 ax = plt.subplot(a,b,i+1)  
 m = setup_m(basin)
 m.contourf(lonm, latm, var_p2, cmap=cmap,levels=lev,extend="both")

 cax = get_cax2(ax) 
 cbar=plt.colorbar(cax = cax, ticks=levc)
 cbar.ax.tick_params(labelsize=18)
 plt.clim([cmin,cmax])

 ax.set_title(t2,fontsize=ft)

## 1 phase - reversed colorbar
def make_contourfs_mjo_mod_vs_obs_1p_rcbar(a,b,i,basin,lono,lato,lonm,latm,\
                              var_p1,var_p2, \
                              t1,t2,cmap,\
                              lev,levc,ft,ft1):

 #print basin
 cmin = np.min(lev)
 cmax = np.max(lev)

 ax = plt.subplot(a,b,i) 
 m = setup_m(basin)
 m.contourf(lono, lato, -var_p1, cmap=cmap,levels=lev,extend="both")

 plt.clim([cmin,cmax])
 ax.set_title(t1,fontsize=ft)

 ax = plt.subplot(a,b,i+1)  
 m = setup_m(basin)
 m.contourf(lonm, latm, -var_p2, cmap=cmap,levels=lev,extend="both")

 cax = get_cax2(ax) 
 cbar=plt.colorbar(cax = cax, ticks=levc)
 cbar.ax.tick_params(labelsize=18)
 plt.clim([cmin,cmax])
 cbar.ax.invert_yaxis()
 cbar.set_ticklabels(levc[::-1])

 ax.set_title(t2,fontsize=ft)


## 1 phase - return ax
def make_contourfs_mjo_mod_vs_obs_1p_ax(a,b,i,basin,lono,lato,lonm,latm,\
                              var_p1,var_p2, \
                              t1,t2,cmap,\
                              lev,levc,ft,ft1):

 cmin = np.min(lev)
 cmax = np.max(lev)

 ax1 = plt.subplot(a,b,i) 
 m = setup_m(basin)
 m.contourf(lono, lato, var_p1, cmap=cmap,levels=lev,extend="both")

 plt.clim([cmin,cmax])
 ax1.set_title(t1,fontsize=ft)

 ax2 = plt.subplot(a,b,i+1)  
 m = setup_m(basin)
 m.contourf(lonm, latm, var_p2, cmap=cmap,levels=lev,extend="both")

 cax = get_cax2(ax2) 
 cbar=plt.colorbar(cax = cax, ticks=levc)
 cbar.ax.tick_params(labelsize=18)
 plt.clim([cmin,cmax])

 ax2.set_title(t2,fontsize=ft)

 return ax1, ax2

## 2 phase
def make_contourfs_mjo_mod_vs_obs_2p(a,b,i,basin,lono,lato,lonm,latm,\
                              var_p1,var_p2, var_p3, var_p4, \
                              t1, t2, t3, t4, cmap,\
                              lev,levc,ft,ft1):

 cmin = np.min(lev)
 cmax = np.max(lev)

 ax1 = plt.subplot(a,b,i) 
 m = setup_m(basin)
 m.contourf(lono, lato, var_p1, cmap=cmap,levels=lev,extend="both")
 plt.clim([cmin,cmax])


 ax2 = plt.subplot(a,b,i+1)  
 m = setup_m(basin)
 m.contourf(lonm, latm, var_p2, cmap=cmap,levels=lev,extend="both")
 plt.clim([cmin,cmax])


 ax3 = plt.subplot(a,b,i+2) 
 m = setup_m(basin)
 m.contourf(lono, lato, var_p3, cmap=cmap,levels=lev,extend="both")
 plt.clim([cmin,cmax])


 ax4 = plt.subplot(a,b,i+3)  
 m = setup_m(basin)
 m.contourf(lonm, latm, var_p4, cmap=cmap,levels=lev,extend="both")

 cax = get_cax2(ax4) 
 cbar=plt.colorbar(cax = cax, ticks=levc)
 cbar.ax.tick_params(labelsize=18)
 plt.clim([cmin,cmax])

 ax1.set_title(t1,fontsize=ft)
 ax2.set_title(t2,fontsize=ft)
 ax3.set_title(t3,fontsize=ft)
 ax4.set_title(t4,fontsize=ft)
