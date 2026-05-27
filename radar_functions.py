import numpy as np
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime
import sys
sys.path.append('/work/kng/py_functions/')
from functions import *
from tc_functions import *
from netCDF4 import num2date

def extract_sweep_time(radar, sweep):
  ray_times = num2date(
      radar.time['data'],
      radar.time['units'])
  start = radar.sweep_start_ray_index['data'][sweep]
  end   = radar.sweep_end_ray_index['data'][sweep]
  time_data = radar.time['data'][start:end+1]
  mean_time_num = time_data[0] #.mean()
  sweep_time = num2date(mean_time_num, radar.time['units'])
  return sweep_time.strftime("%Y%m%d_%H%M%S")

def extract_time_info(filename):
  match = re.search(r'_(\d{8})_(\d{6})', filename)
  date_part, time_part = match.groups()
  dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
  return dt

def read_laura_obs():
  fmt = "%Y%m%d%H"
  filename = '/work/Kun.Gao/trak_ver/all/bal132020.dat'
  tc_dict = read_atcf(filename, False)
  lon = tc_dict['lon']
  lat = tc_dict['lat']
  date_all = tc_dict['date']

  date_obs = []
  lon_obs = []
  lat_obs = []

  for i in range(len(date_all)):
   if (date_all[i] not in date_obs):
      lon_obs.append(lon[i])
      lat_obs.append(lat[i])
      date_obs.append(datetime.strptime(date_all[i], fmt))
  return date_obs, lon_obs, lat_obs

def get_tc_center(tc_date, tc_lon, tc_lat, target_date):

    # Convert datetime to numeric seconds since start
    time_sec = np.array([(t - tc_date[0]).total_seconds() for t in tc_date])
    target_sec = (target_date - tc_date[0]).total_seconds()

    # If outside the range, clip to bounds
    if target_sec <= time_sec[0]:
        return tc_lon[0], tc_lat[0]
    if target_sec >= time_sec[-1]:
        return tc_lon[-1], tc_lat[-1]

    # Linear interpolation
    lon_interp = np.interp(target_sec, time_sec, tc_lon)
    lat_interp = np.interp(target_sec, time_sec, tc_lat)

    return lon_interp, lat_interp
