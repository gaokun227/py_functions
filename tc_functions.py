import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import glob
import os
import re
from scipy.stats import norm
from matplotlib.ticker import PercentFormatter

# This files contains a collection of functions / class for
# 1) TC forecast verficaition based on Tim's package
# 2) ATCF file related operations

#===========================================================================
# Part 1: A very powerful CLASS: stratify_TC_error
#
# To gain insight on the error distribution, model differeces, etc based on 
# csv-type files that contain track, intensity or size error records
#===========================================================================

class stratify_TC_error():

  def __init__(self):
     self.col_list = ['b','g','r','m','orange','c','y']

  # ===== generate a pd dataframe containing columns defined in name_list

  def preprocess(self,filename,model_name_map):

     # ABOUT COL 7 to 14
     #7:  best track lat
     #8:  best track lon
     #9:  best track Vmax
     #10:  intensity forecast bias (kts) or error for track (nm) 
     #11:  mean forecast error for this lead time (kts)
     #12:  standard deviation of forecast error for this lead time (kts)
     #13:  forecast latitude
     #14:  forecast longitude

     # --- define a dict that map 4-digit modelID to a more readable name
     self.model_name_map = model_name_map

     # --- read in data as a pd dataframe
     #col_list = [0,1,3,4,9]
     #name_list = ['modelID', 'leadTime','stormID','stormName','Error']

     col_list = [0,1,3,4,6,7,8,9]
     name_list = ['modelID', 'leadTime','stormID','stormName', 'latObs', 'lonObs', 'vmaxObs', 'Error']

     self.df_xtr = pd.read_csv(filename, header=None, delimiter=r"\s+", usecols=col_list,names=name_list)
     self.df_xtr['Abs_Error'] =  self.df_xtr['Error'].abs()
     self.df_xtr['lonObs'] =  360.-self.df_xtr['lonObs']

     # --- generate a list contains all model 4-digit IDs
     self.modelID_list = self.df_xtr['modelID'].unique()

     # --- generate a storm name map 
     df_xtr = self.df_xtr[self.df_xtr['leadTime'] == 00]
     df_xtr['stormName'] = df_xtr['stormName']+'-'+df_xtr['stormID'].str[-4:]
     df_map = df_xtr[['stormID','stormName']].drop_duplicates().groupby('stormID').tail(1)
     self.storm_name_dict = dict(zip(df_map['stormID'], df_map['stormName']))

     # --- get all case mean error/bias at all leadtime (00h, 12h, 24h, ...)
     self.df_allcase_error = self.df_xtr.groupby(['modelID','leadTime'])['Abs_Error'].mean()
     self.df_allcase_bias  = self.df_xtr.groupby(['modelID','leadTime'])['Error'].mean()


  # ===== generate a pd dataframe containing columns defined in name_list for wind radii

  def preprocess_wind_radii(self,filename,model_name_map,target='R34'):

     # --- define a dict that map 4-digit modelID to a more readable name
     self.model_name_map = model_name_map

     # --- read in data as a pd dataframe

     if target == "R34":
       col_list =  [0,          1,         2,          3,          5,      6,      7,      8]
     if target == "R50":
       col_list =  [0,          1,         2,          3,          9,      10,     11,     12]
     if target == "R64":
       col_list =  [0,          1,         2,          3,         13,      14,     15,     16]

     name_list = ['modelID', 'leadTime','stormName', 'stormID',  'err1', 'err2', 'err3', 'err4']

     self.df_xtr = pd.read_csv(filename, header=0, usecols=col_list, names=name_list)

     cols = ['err1', 'err2', 'err3', 'err4']

     self.df_xtr[cols] = self.df_xtr[cols].apply(pd.to_numeric,errors='coerce')
     #print(self.df_xtr[['err1', 'err2', 'err3', 'err4']].dtypes)

     self.df_xtr["Error"] = self.df_xtr[['err1', 'err2', 'err3', 'err4']].mean(axis=1) # NaN excluded by default
     self.df_xtr = self.df_xtr.dropna(subset=['Error']) # drop rows where Error is NaN

     self.df_xtr['Abs_Error'] =  self.df_xtr['Error'] # just to make extra copy

     print(self.df_xtr)

     # --- generate a list contains all model 4-digit IDs
     self.modelID_list = self.df_xtr['modelID'].unique()

     # --- generate a storm name map 
     df_xtr = self.df_xtr[self.df_xtr['leadTime'] == 00]
     df_xtr['stormName'] = df_xtr['stormName']+'-'+df_xtr['stormID'].str[-4:]
     df_map = df_xtr[['stormID','stormName']].drop_duplicates().groupby('stormID').tail(1)
     self.storm_name_dict = dict(zip(df_map['stormID'], df_map['stormName']))

     # --- get all case mean error/bias at all leadtime (00h, 12h, 24h, ...)
     self.df_allcase_error = self.df_xtr.groupby(['modelID','leadTime'])['Abs_Error'].mean()
     self.df_allcase_bias  = self.df_xtr.groupby(['modelID','leadTime'])['Error'].mean() # there is useless 

  # ===== The following functions are to 
  #       obtain aggregated errors based on model and storm IDs at given lead time hh

  @staticmethod
  def get_percent_contribution(Mean_list, N_list):
     N_all = sum(N_list)
     Mean_all = np.sum(np.array(Mean_list)*np.array(N_list))/float(N_all)
     Percent = 100*(np.array(Mean_list)*np.array(N_list))/float(N_all*Mean_all)
     return Percent.tolist()

  @staticmethod
  def sort_lists(listA, listB, listC, list_ref):
     listA_sorted = [x for _,x in sorted(zip(list_ref, listA), reverse=True)]
     listB_sorted = [x for _,x in sorted(zip(list_ref, listB), reverse=True)]
     listC_sorted = [x for _,x in sorted(zip(list_ref, listC), reverse=True)]
     list_ref_sorted = sorted(list_ref, reverse=True)
     return listA_sorted, listB_sorted, listC_sorted, list_ref_sorted

  def perform_error_aggregation(self, hh):
     self.bias_dict = {}
     self.error_dict = {}
     self.count_dict = {}
     self.hh = hh

     df_xtr = self.df_xtr[self.df_xtr['leadTime'] == hh]

     df_bias = df_xtr.groupby(['modelID','stormID'])['Error'].mean()
     df_error = df_xtr.groupby(['modelID','stormID'])['Abs_Error'].mean()
     df_count = df_xtr.groupby(['modelID','stormID'])['Error'].count()

     #df_comb = pd.concat([df_mean, df_count], axis=1)
     #print df_bias.head()
     #print df_error.head()
     #print df_count.head()

     for modelID in self.modelID_list:
        self.bias_dict[modelID]  = df_bias[modelID].to_dict()
        self.error_dict[modelID] = df_error[modelID].to_dict()
        self.count_dict[modelID] = df_count[modelID].to_dict()
      
  # ===== plot storm-by-storm error in one model 

  def plot_by_storm_oneModel(self, type, modelID, do_sort=True):

     stormID_list = []
     value_list = []
     count_list = []

     if type == 'intensity_bias':
         oneModelDict = self.bias_dict[modelID]
     elif 'error' in type:
         oneModelDict = self.error_dict[modelID]
     else:
         print ('wrong type!!!')
     for stormID, value in oneModelDict.iteritems():
         value_list.append(value)
         stormID_list.append(stormID)
         count_list.append(self.count_dict[modelID][stormID])

     percent_list = self.get_percent_contribution(value_list, count_list)

     if do_sort:
       stormID_list, value_list, count_list, percent_list = self.sort_lists(stormID_list, value_list, count_list, percent_list)  

     # override xlabels - sort by storm ID below !!! change !!!
     stormID_list, value_list, count_list, percent_list = self.sort_lists(stormID_list, value_list, count_list, stormID_list)

     fig = plt.figure(figsize=(12,6))
     ax = plt.subplot(111)
     ft = 16
     ft1 = 16 
     ft2 = 41

     x = np.arange(len(value_list)) 
     width = 0.25 

     if 'bias' in type: 
       allcase_mean = self.df_allcase_bias[modelID,self.hh]
     elif 'error' in type:
       allcase_mean = self.df_allcase_error[modelID,self.hh]

     label = self.model_name_map[modelID] + ' (Mean = ' +  '%.1f' % allcase_mean + ')'

     ax.bar(x, value_list, width, color=self.col_list[0], label=label)

     low, high = plt.ylim()
     bound = max(abs(low), abs(high))
     if 'bias' in type:
        ax.axhline(y=0, c='k', lw=1)
        plt.ylim(-bound, bound)
     else:
        plt.ylim(0, bound*1.1)
        plt.ylim(0, 600)  #!!! change !!!

        if do_sort:
         for x1, p1 in zip(x, percent_list):
            plt.text(x1, bound, '%.1f' % p1 + '%', ha='center', va='bottom', fontsize=ft2)

     if 'bias' in type:
        plt.legend(loc='upper right', ncol=1,  fontsize=ft1)
     elif 'error' in type:
        plt.legend(loc='upper right', ncol=1,  fontsize=ft1)

     xticks = range(len(stormID_list))
     xticklabels=[]
     for stormID in stormID_list:
         count = self.count_dict[modelID][stormID] # using last modelID; homogenous sample
         xtick = self.storm_name_dict[stormID] + ' (' + str(count) + ')'
         xticklabels.append(xtick)

     ax.set_xticks(xticks)
     ax.set_xticklabels(xticklabels, rotation=270)

     for tick in ax.xaxis.get_major_ticks():
         tick.label1.set_fontsize(ft1)
     for tick in ax.yaxis.get_major_ticks():
         tick.label1.set_fontsize(ft1)
    
     if type == 'intensity_error':
         unit = 'kts'
         title_a = 'Intensity Error'
     elif type == 'intensity_bias':
         unit = 'kts'
         title_a = 'Intensity Bias' 
     elif type == 'track_error':
         unit = 'nm'
         title_a = 'Track Error'
     elif type == 'size_error':
         unit = 'nm'
         title_a = 'Size Error'
 
     ax.set_title(title_a + ' at '  + str(self.hh) + ' hr', fontsize=ft1) 
     ax.set_ylabel(unit, fontsize=ft1)

     filename = 'oneModel_'+modelID+'_'+type+'_'+str(self.hh)+'hr_by_storm'
     fig.savefig(filename+'.png',bbox_inches='tight')
     #plt.show()
                                         
  # ===== plot storm-by-storm errors in two models
                                  
  def plot_by_storm_twoModels(self, type, modelID1, modelID2, show_percent=True):

     stormID_list = []
     count_list = []

     value_list1 = []
     value_list2 = []

     if type == 'intensity_bias':
         oneModelDict1 = self.bias_dict[modelID1]
         oneModelDict2 = self.bias_dict[modelID2]
     elif 'error' in  type:
         oneModelDict1 = self.error_dict[modelID1]
         oneModelDict2 = self.error_dict[modelID2]
     else:
         print ('wrong type!!!')
     for stormID, value1 in oneModelDict1.items():
         stormID_list.append(stormID)
         count_list.append(self.count_dict[modelID1][stormID])
         value_list1.append(value1)
         value_list2.append(oneModelDict2[stormID])

     diff_list = (np.array(value_list2) - np.array(value_list1)).tolist()
     percent_list = self.get_percent_contribution(diff_list, count_list)
     stormID_list, value_list1, value_list2, percent_list = self.sort_lists(stormID_list, value_list1, value_list2, percent_list)

     fig = plt.figure(figsize=(24,6))
     ax=plt.subplot(111)
     ft = 16
     ft1 = 12
     ft2 = 10

     x = np.arange(len(value_list1))
     width = 0.35

     if 'bias' in type:
       allcase_mean1 = self.df_allcase_bias[modelID1,self.hh]
       allcase_mean2 = self.df_allcase_bias[modelID2,self.hh]
     elif 'error' in type:
       allcase_mean1 = self.df_allcase_error[modelID1,self.hh]
       allcase_mean2 = self.df_allcase_error[modelID2,self.hh]

     label1 = self.model_name_map[modelID1] + ' (Mean = ' +  '%.1f' % allcase_mean1 + ')'
     label2 = self.model_name_map[modelID2] + ' (Mean = ' +  '%.1f' % allcase_mean2 + ')'

     ax.bar(x-width/2, value_list1, width, color=self.col_list[0], label=label1)
     ax.bar(x+width/2, value_list2, width, color=self.col_list[1], label=label2)

     low, high = plt.ylim()
     bound = max(abs(low), abs(high))
     if 'bias' in type:
        ax.axhline(y=0, c='k', lw=1)
        plt.ylim(-bound, bound)
     else:
        plt.ylim(0, bound*1.1)
        if show_percent:
          for x1, p1 in zip(x, percent_list):
            plt.text(x1, bound, '%.1f' % p1 + '%', ha='center', va='bottom', fontsize=ft2)

     if 'bias' in type:
        plt.legend(loc='upper right', ncol=1,  fontsize=ft1)
     elif 'error' in type:
        plt.legend(loc='center right', ncol=1,  fontsize=ft1)

     xticks = range(len(stormID_list))
     xticklabels=[]
     for stormID in stormID_list:
         count = self.count_dict[modelID1][stormID]
         xtick = self.storm_name_dict[stormID] + ' (' + str(count) + ')'
         xticklabels.append(xtick)

     ax.set_xticks(xticks)
     ax.set_xticklabels(xticklabels, rotation=270)

     for tick in ax.xaxis.get_major_ticks():
         tick.label1.set_fontsize(ft1)
     for tick in ax.yaxis.get_major_ticks():
         tick.label1.set_fontsize(ft1)

     if type == 'intensity_error':
         unit = 'kts'
         title_a = 'Intensity Error'
     elif type == 'intensity_bias':
         unit = 'kts'
         title_a = 'Intensity Bias'    
     elif type == 'track_error':
         unit = 'nm'
         title_a = 'Track Error'
     elif type == 'size_error':
         unit = 'nm'
         title_a = 'Size Error'

     ax.set_title(title_a + ' at '  + str(self.hh) + ' hr', fontsize=ft1)                   
     ax.set_ylabel(unit, fontsize=ft1)

     filename = 'twoModel_'+modelID1+'_vs_'+modelID2+'_'+type+'_'+str(self.hh)+'hr_by_storm'
     fig.savefig(filename+'.png',bbox_inches='tight')
     #plt.show()


  # ===== show where the storms are located and their intensity distributions
  #       assuming homogeneous model comparison
  #       still work in progress

  def show_records_dist(self, option='track', vmax_cutoff=64, hh_sel=None):
      model_sel = self.modelID_list[0]

      if option == 'track': # show storm initial location 
        if hh_sel != None:
          df_xtr_sel = self.df_xtr[(self.df_xtr['modelID'] == model_sel) & (self.df_xtr['leadTime'] == hh_sel)]
        else:
          df_xtr_sel = self.df_xtr[self.df_xtr['modelID'] == model_sel]

        lon_obs = df_xtr_sel['lonObs'].tolist()
        lat_obs = df_xtr_sel['latObs'].tolist()
        vmax_obs = df_xtr_sel['vmaxObs'].tolist()

        lon_obs = np.array(lon_obs)
        lat_obs = np.array(lat_obs)
        vmax_obs = np.array(vmax_obs)

        fig = plt.figure(figsize = (12,8))

        m = setup_m('NAtl_wnest') 

        marker = 'o'
        ms = 6
        col = 'k'
        label = 'Obs'
 
        # seperate storms based on vmax_obs
        mask1 = vmax_obs < vmax_cutoff
        mask2 = vmax_obs >= vmax_cutoff
 
        lon_obs_g1 = np.ma.MaskedArray(lon_obs,mask=~mask1)
        lat_obs_g1 = np.ma.MaskedArray(lat_obs,mask=~mask1)
        lon_obs_g2 = np.ma.MaskedArray(lon_obs,mask=~mask2)
        lat_obs_g2 = np.ma.MaskedArray(lat_obs,mask=~mask2)

        x1, y1 = m(lon_obs_g1, lat_obs_g1)
        x2, y2 = m(lon_obs_g1, lat_obs_g2)

        m.plot(x1, y1, 'ko', ms=ms, label='Below cutoff intensity - ' +str(vmax_cutoff))
        m.plot(x2, y2, 'ro', ms=ms, label='Above cutoff intensity - ' +str(vmax_cutoff))

        plt.title('Storm location at initial time', fontsize=16)
        plt.legend()
        plt.show()

  # ===== compare the error distributions in two models at lead times given in hh_list

  def compare_error_twoModels(self, modelA, modelB, hh_list, opt=0):

    # ideally, modelA should be better than modelB
    for hh in hh_list:
       df_err = self.df_xtr[self.df_xtr['leadTime'] == hh]
       listA=df_err[df_err['modelID']==modelA]['Abs_Error'].tolist()
       listB=df_err[df_err['modelID']==modelB]['Abs_Error'].tolist()
       counter = 0
       for i in range(len(listA)):
          if listB[i] > listA[i]:
             counter += 1

       ratio = round(100*float(counter)/len(listA),1)

       # compare difference
       if opt == 0: 
         diff = np.array(listA) - np.array(listB) 
         mu = np.mean(diff)
         std = np.std(diff) 

         fig = plt.figure(figsize=(8,8))
         ax = plt.subplot(111)
         plt.hist(diff, bins=25, density=True, alpha=0.6, color='b')
  
         xmin, xmax = plt.xlim()
         x = np.linspace(xmin, xmax, 100)
         p = norm.pdf(x, mu, std)
         plt.plot(x, p, 'k', linewidth=2)

         ax.axvline(x=mu, c='k', lw=1, ls='--')
         ax.axvline(x= mu + 1*std, c='k', lw=1, ls='--')
         ax.axvline(x= mu - 1*std, c='k', lw=1, ls='--')
         ax.axvline(x= mu + 2*std, c='k', lw=1, ls='--')
         ax.axvline(x= mu - 2*std, c='k', lw=1, ls='--')

         title =  modelA + ' vs ' + modelB + ' at hour ' + str(hh) + '\n'\
               + 'sample size: ' + str(len(listA)) + '\n'\
               + 'model diff mean and std: {:.2f} and {:.2f}'.format(mu, std) + '\n'\
               + 'superior performance in ' + modelA + ': ' + str(ratio) + '%'
         plt.title(title)
         plt.show()

       elif opt == 1:
       # two panels - one for each model
         modelA_error = np.array(listA) 
         modelB_error = np.array(listB) 
         muA = np.mean(modelA_error)
         stdA = np.std(modelA_error) 
         muB = np.mean(modelB_error)
         stdB = np.std(modelB_error)

         bmin = 0
         bmax = max(np.max(modelA_error), np.max(modelB_error))
         bins = np.arange(bmin, bmax+25, 25)

         ft = 14
         fig = plt.figure(figsize=(12,6))

         # model A
         ax = plt.subplot(121)
         plt.hist(modelA_error, bins=bins, density=False, weights=np.ones(len(modelA_error))/len(modelA_error), alpha=0.6, color='b')

         plt.ylim([0, 0.2])
         plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

         title =  'PDF of error at Hour ' + str(hh) + ': ' +self.model_name_map[modelA] +  '\n' \
                 + 'mean and std: {:.2f}nm and {:.2f}nm'.format(muA, stdA)
         plt.title(title, fontsize = ft)
         plt.xlabel('nm', fontsize = ft)

         ax.axvline(x = muA, c='k', lw=2, ls='-')
         ax.axvline(x = muA + 1*stdA, c='k', lw=2, ls='--')
         ax.axvline(x = muA - 1*stdA, c='k', lw=2, ls='--')

         # model B
         ax = plt.subplot(122)
         plt.hist(modelB_error, bins=bins, density=False, weights=np.ones(len(modelB_error))/len(modelB_error), alpha=0.6, color='b')

         plt.ylim([0, 0.2])
         plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

         title =  'PDF of error at Hour ' + str(hh) + ': ' +self.model_name_map[modelB] +  '\n' \
                 + 'mean and std: {:.2f}nm and {:.2f}nm'.format(muB, stdB)
         plt.title(title, fontsize = ft)
         plt.xlabel('nm', fontsize = ft)

         ax.axvline(x = muB, c='k', lw=2, ls='-')
         ax.axvline(x = muB + 1*stdB, c='k', lw=2, ls='--')
         ax.axvline(x = muB - 1*stdB, c='k', lw=2, ls='--')

         plt.show()

# <=================== end of the class

#=====================================================================
#  Part 2: a collection of functions to run Tim's TC verification tool 
#=====================================================================

def run_verify(work_dir, card_tag, type, model_name_map, do_plot_only = False, model_sel=[], color_sel=[], atcf_dir = '../all/', do_hour0=False, sig_test_model_list=[]):

  if type == 'intensity':
    card_name = 'icard.'+card_tag
  elif type == 'track':
    card_name = 'tcard.'+card_tag
  elif type == 'radii':
    card_name = 'rcard.'+card_tag

  # --- run verification tool
  if not do_plot_only:
     cmd = '../exec/tcver.x {}{} {}'.format(work_dir, card_name, atcf_dir)
     os.system(cmd)

     # process radius verification output
     if type == 'radii':
        cmd = '/work/Kun.Gao/trak_ver/scripts/radiicut.sh {}{}.out > {}{}.out2'.format(work_dir, card_name, work_dir, card_name)
        os.system(cmd)

  # --- make plots
  if type == 'intensity':
    tc_dict = read_intensity_stat(work_dir+card_name+'.out')
    #for key, value in tc_dict.items():
    #    print(key, ' : ', value)
    plot_stat(tc_dict, 'intensity_error', card_name, work_dir, model_name_map, model_sel, color_sel, do_hour0, sig_test_model_list)
    plot_stat(tc_dict, 'intensity_bias', card_name, work_dir, model_name_map, model_sel, color_sel, do_hour0)
  elif type == 'track':
    tc_dict = read_track_stat(work_dir+card_name+'.out')
    plot_stat(tc_dict, 'track_error', card_name, work_dir, model_name_map, model_sel, color_sel, do_hour0, sig_test_model_list)
    plot_stat(tc_dict, 'track_xbias', card_name, work_dir, model_name_map, model_sel, color_sel, do_hour0)
    plot_stat(tc_dict, 'track_ybias', card_name, work_dir, model_name_map, model_sel, color_sel, do_hour0)
  elif type == 'radii':
    tc_dict = read_radii_stat(work_dir+card_name+'.out2')
    plot_stat(tc_dict, 'R34',       card_name, work_dir, model_name_map, model_sel, color_sel, do_hour0)
    plot_stat(tc_dict, 'R34_error', card_name, work_dir, model_name_map, model_sel, color_sel, do_hour0)
    plot_stat(tc_dict, 'R34_bias',  card_name, work_dir, model_name_map, model_sel, color_sel, do_hour0)
    plot_stat(tc_dict, 'R64',       card_name, work_dir, model_name_map, model_sel, color_sel, do_hour0)
    plot_stat(tc_dict, 'R64_error', card_name, work_dir, model_name_map, model_sel, color_sel, do_hour0)
    plot_stat(tc_dict, 'R64_bias',  card_name, work_dir, model_name_map, model_sel, color_sel, do_hour0)

def read_intensity_stat(filename):

  f1 = open(filename, "r")
  
  # step1: locate where the stats are
  counter = 0
  for line in f1:
     if 'AVERAGE INTENSITY ERRORS' in line:
        error_str = counter+2
     elif 'ERROR STANDARD DEVIATION' in line:
        error_end = counter-4
        std_str = counter+2
     elif 'AVERAGE INTENSITY BIAS' in line:
        std_end = counter-2
        bias_str = counter+2
     counter += 1

  # step2: read in values
  tc_dict = {}
  num_model = error_end - error_str + 1
  f1 = open(filename, "r")
  counter = 0
  for line in f1:
    if counter == error_str+num_model:
       L = line.split()
       tc_dict[L[0]] = list(map(int, L[1:]))
    elif counter >= error_str and counter <= error_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_intensity_error'] = list(map(float,L[1:]))
    elif counter >= std_str and counter <= std_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_error_std'] = map(float,L[1:])
    elif counter >= bias_str and counter <= bias_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_intensity_bias'] = list(map(float,L[1:]))
    counter += 1

  return tc_dict

def read_track_stat(filename):

  f1 = open(filename, "r")

  # step1: locate where the stats are  
  counter = 0
  for line in f1:
     if 'average track errors' in line:
        error_str = counter+2
     elif 'ERROR STANDARD DEVIATION' in line:
        error_end = counter-4
        std_str = counter+2
     elif 'AVERAGE XBIAS' in line:
        std_end = counter-2
        xbias_str = counter+2
     elif 'AVERAGE YBIAS' in line:
        ybias_str = counter+2
     counter += 1
 
  # step2: read in values
  tc_dict = {}
  num_model = error_end - error_str + 1
  f1 = open(filename, "r")

  counter = 0
  for line in f1:

    if counter == error_str+num_model:
       L = line.split()
       tc_dict[L[0]] = list(map(int, L[1:])) # Case number
    elif counter >= error_str and counter <= error_str+num_model-1: 
       L = line.split()
       tc_dict[L[0]+'_track_error'] = list(map(float,L[1:]))
    elif counter >= std_str and counter <= std_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_error_std'] = list(map(float,L[1:]))
    elif counter >= xbias_str and counter <= xbias_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_track_xbias'] = list(map(float,L[1:]))
    elif counter >= ybias_str and counter <= ybias_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_track_ybias'] = list(map(float,L[1:]))
    counter += 1
  return tc_dict

def read_radii_stat(filename):

  f1 = open(filename, "r")

  # step1: locate where the stats are  
  counter = 0
  for line in f1:
     if 'MEAN VALUES OF R34' in line:
        r34_str = counter+2
     elif 'MEAN ERROR OF R34' in line:
        r34_end = counter-3
        r34_err_str = counter+2
     elif 'MEAN BIAS OF R34' in line:
        r34_bias_str = counter+2

     if 'MEAN VALUES OF R50' in line:
        r50_str = counter+2
     elif 'MEAN ERROR OF R50' in line:
        r50_err_str = counter+2
     elif 'MEAN BIAS OF R50' in line:
        r50_bias_str = counter+2

     if 'MEAN VALUES OF R64' in line:
        r64_str = counter+2
     elif 'MEAN ERROR OF R64' in line:
        r64_err_str = counter+2
     elif 'MEAN BIAS OF R64' in line:
        r64_bias_str = counter+2

     counter += 1
 
  # step2: read in values

  
  tc_dict = {}

  num_model = r34_end - r34_str + 1

  f1 = open(filename, "r")

  counter = 0
  for line in f1:

    # get r34 stat
    if counter == r34_str+num_model:
       L = line.split()
       tc_dict[L[0]+'_R34'] = list(map(int, L[1:])) # Case number
    elif counter >= r34_str and counter <= r34_str+num_model-1: 
       L = line.split()
       tc_dict[L[0]+'_R34'] = list(map(float,L[1:]))
    elif counter >= r34_err_str and counter <= r34_err_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_R34_error'] = list(map(float,L[1:]))
    elif counter >= r34_bias_str and counter <= r34_bias_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_R34_bias'] = list(map(float,L[1:]))

    # get r50 stat
    if counter == r50_str+num_model:
       L = line.split()
       tc_dict[L[0]+'_R50'] = list(map(int, L[1:])) # Case number
    elif counter >= r50_str and counter <= r50_str+num_model-1: 
       L = line.split()
       tc_dict[L[0]+'_R50'] = list(map(float,L[1:]))
    elif counter >= r50_err_str and counter <= r50_err_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_R50_error'] = list(map(float,L[1:]))
    elif counter >= r50_bias_str and counter <= r50_bias_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_R50_bias'] = list(map(float,L[1:]))

    # get r64 stat
    if counter == r64_str+num_model:
       L = line.split()
       tc_dict[L[0]+'_R64'] = list(map(int, L[1:])) # Case number
    elif counter >= r64_str and counter <= r64_str+num_model-1: 
       L = line.split()
       tc_dict[L[0]+'_R64'] = list(map(float,L[1:]))
    elif counter >= r64_err_str and counter <= r64_err_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_R64_error'] = list(map(float,L[1:]))
    elif counter >= r64_bias_str and counter <= r64_bias_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_R64_bias'] = list(map(float,L[1:]))
    counter += 1

  #print r34_str, r34_end, r34_err_str, r34_bias_str
  #print num_model
  #print tc_dict['#CASES_R64']

  return tc_dict


def plot_stat(tc_dict, keyword, card_name, pic_dir, model_name_map, model_sel, color_sel, do_hour0=False, sig_test_model_list=[]):

    ms = 12
    mec = 'w'
    lw = 3

    ft = 22
    ft1 = 20

    if do_hour0:
       time = np.array([0, 12, 24, 36, 48, 72, 96, 120])
       xticks = [0, 2, 4, 5, 6, 7]
       xmin, xmax = -6, 132
    else:
       time = np.array([12, 24, 36, 48, 72, 96, 120])
       xticks = [1, 3, 4, 5, 6]
       xmin, xmax = 0, 132 

    if 'R34' in keyword or 'R64' in keyword or 'R50' in keyword:
        case_num = tc_dict['#CASES_'+keyword[:3]]
    else:
        case_num = tc_dict['#CASES']
    if not do_hour0:
       case_num = case_num[1:]

    if model_sel == []: # models to be plotted not specified
       keys = tc_dict.keys()
       if 'R34' in keyword or 'R64' in keyword or 'R50' in keyword:
          models = [s[:4] for s in keys if 'R34_error' in s]
       else:
          models = [s[:4] for s in keys if keyword in s]
       models = sorted(models)
       num_models = len(models)

       # extract model names from card_name
       # assuming card_name has this format:  xcard.mod1.mod2.mode3...
       model_name_string = card_name[6:6+5*num_models]
       models = []
       for i in range(num_models):
          models.append(model_name_string[5*i:5*(i+1)][:-1].upper())
    else:
       models = model_sel

    if color_sel == []: 
       colors = ['b', 'g', 'orange', 'r', 'm', 'c', 'y']
    else:
       colors = color_sel

    num_models = len(models)
 
    # significance test stuff (two-tail student t test)
    if  ('error' in keyword) and (len(sig_test_model_list) >=1):
       #print 'doing significance test'
       #print sig_test_model_list 
       for model in sig_test_model_list:
          alpha = 1.960
          #if comm_number >= 40 and comm_number < 80:
          #    alpha = 2.0
          #if comm_number >= 20 and comm_number < 40:
          #    alpha = 2.042
          #if comm_number < 20:
          #    alpha = 2.228
          tc_dict[model+'_dev'] = alpha * np.array( tc_dict[model+'_error_std'] / np.sqrt(tc_dict['#CASES']) )

    fig = plt.figure(figsize=(14,8))
    ax=plt.subplot(111)

    for i in range(num_models):
      #print 'plotting', models[i]+'_'+keyword
      label = model_name_map[models[i]]
      values = np.array(tc_dict[models[i]+'_'+keyword])
      values[np.abs(values) > 900] = np.nan
      if not do_hour0:
         values = values[1:]
 
      ax.plot(time, values, marker='o', lw=lw, ms=ms, mec=mec, label = label, c=colors[i])
      if ('error' in keyword) and (models[i] in sig_test_model_list):
         dev_u = tc_dict[models[i]+'_'+keyword]+tc_dict[models[i]+'_dev']
         dev_d = tc_dict[models[i]+'_'+keyword]-tc_dict[models[i]+'_dev']
         ax.fill_between(time, dev_d, dev_u, alpha=0.2, edgecolor=colors[i],facecolor=colors[i])

    if 'bias' in keyword:
        ax.axhline(y=0, c='k', lw=1)
        low, high = plt.ylim()
        bound = max(abs(low), abs(high))
        plt.ylim(-bound, bound)

    plt.legend(loc=0, fontsize=ft1)
    plt.grid()
   
    xticklabels=[]
    for i in range(len(xticks)):
        xtick1 =  str(time[xticks[i]])+'h\n(' + str(case_num[xticks[i]]) + ')'
        xticklabels.append(xtick1)
    
    ax.set_xlim([xmin, xmax])
    ax.set_xticks(time[xticks])
    ax.set_xticklabels(xticklabels)  
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(ft1)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(ft1)

    if keyword == 'intensity_error':
       title = 'Intensity Error'
    elif keyword == 'intensity_bias':
       title = 'Intensity Bias'
    elif keyword == 'track_error':
       title = 'Track Error'
    elif keyword == 'track_xbias':
       title = 'E-W Track Bias'
    elif keyword == 'track_ybias':
       title = 'S-N Track Bias'
    elif keyword == 'R34':
       title = 'Mean R34'
    elif keyword == 'R34_error':
       title = 'Mean R34 error'
    elif keyword == 'R34_bias':
       title = 'Mean R34 bias'
    elif keyword == 'R64':
       title = 'Mean R64'
    elif keyword == 'R64_error':
       title = 'Mean R64 error'
    elif keyword == 'R64_bias':
       title = 'Mean R64 bias'

    if 'intensity' in keyword:
       yname = 'knots'
    elif 'track' in keyword or 'R34' in keyword or 'R64' in keyword or 'R50' in keyword:
       yname = 'nm' 

    plt.title(title, fontsize=ft)
    #plt.xlabel('Forecast hour', fontsize=ft)
    plt.ylabel(yname, fontsize=ft)

    model_name_string='.'.join(models);
    fig.savefig(pic_dir+model_name_string+'_'+keyword+'.png', bbox_inches = 'tight')
    #plt.show()

def extract_radii_error(filename, target_models):

    FIELDS = [
      "ne34", "se34", "sw34", "nw34",
      "ne50", "se50", "sw50", "nw50",
      "ne64", "se64", "sw64", "nw64"
    ]

    MISSING_FLAGS = [6666, 7777, 9999]

    #target_models = [m.strip().upper() for m in models.split('.')]

    records = []

    # --------------------------------------------------------
    # Regex pattern
    # --------------------------------------------------------
    pattern = re.compile(
        r"^\s*(\w+)\s+(\d{3})\s+"
        r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+"
        r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+"
        r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)"
    )

    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------
    with open(filename, "r") as f:
        lines = f.readlines()

    storm_info = None

    # --------------------------------------------------------
    # Parse file
    # --------------------------------------------------------
    for line in lines:

        # ----------------------------------------------------
        # Storm header
        # ----------------------------------------------------
        header_match = re.match(
            r"^\s*(\w+)\s+([A-Z]+)\s+(\d{10})",
            line
        )

        if header_match:
            storm_info = {
                "storm_id": header_match.group(1).strip(),
                "storm_name": header_match.group(2).strip(),
                "cycle": header_match.group(3).strip(),
            }
            continue

        # ----------------------------------------------------
        # Data lines
        # ----------------------------------------------------
        m = pattern.match(line)

        if not m:
            continue

        model = m.group(1).strip()
        lead = int(m.group(2))

        # Requested models only
        if model not in target_models:
            continue

        # Skip invalid records
        if storm_info is None:
            continue

        if not storm_info["storm_name"]:
            continue

        if not model:
            continue

        # ----------------------------------------------------
        # Extract radii values
        # ----------------------------------------------------
        values = list(map(int, m.groups()[2:]))

        clean_values = [
            pd.NA if v in MISSING_FLAGS else v
            for v in values
        ]

        # ----------------------------------------------------
        # Build record
        # ----------------------------------------------------
        record = {
            "model": model,
            "lead": lead,
            "storm_name": storm_info["storm_name"],
            "storm_id": storm_info["storm_id"],
            "cycle": storm_info["cycle"],
        }

        for field, value in zip(FIELDS, clean_values):
            record[field] = value

        records.append(record)

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------
    df = pd.DataFrame(records)

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------
    ordered_cols = [
        "model",
        "lead",
        "storm_name",
        "storm_id",
        "cycle",
    ] + FIELDS

    df = df[ordered_cols]

    # --------------------------------------------------------
    # Sort data
    # --------------------------------------------------------
    df = df.sort_values(
        by=["lead", "model", "storm_name"]
    ).reset_index(drop=True)

    return df

# <=================== end of the collection of functions for running Tim's verification tool

# ===========================================
# Part 3: ATCF related functions
# ===========================================

def select_atcf_records(tc_dict, min_lti, max_ini_wind, max_lat) :

    wind_all = tc_dict['wind']
    lat_all = tc_dict['lat']
    rmw_all = tc_dict['rmw']

    # for testing
    try:
      rad1 = tc_dict['rad34_1']
      rad2 = tc_dict['rad34_2']
      rad3 = tc_dict['rad34_3']
      rad4 = tc_dict['rad34_4']
      r34_all = mean_wind_radii(rad1, rad2, rad3, rad4)
    except:
      r34_all = tc_dict['r34']

    tc_dict_g1 = {}
    tc_dict_g2 = {}

    # do not consider the TC if it is
    # 1) too weak
    # 2) too northwards
    # 3) initially too strong - only applies to model tracks 
    if np.max(wind_all) < min_lti or np.min(lat_all) > max_lat or wind_all[0] > max_ini_wind:
       return tc_dict_g1, tc_dict_g2

    # remove high-lat records
    counter = 0
    for lat in lat_all:
      if lat <= max_lat:
         counter += 1
      else:
         break
    wind_all = wind_all[:counter]
    lat_all = lat_all[:counter]
    rmw_all = rmw_all[:counter]
    r34_all = r34_all[:counter]

    # seperate records before and after peak intensity
    if len(wind_all) >= 2:
     idx = np.argmax(wind_all) 
   
     if idx >=1:
      wind_g1 = wind_all[:idx+1]
      lat_g1 = lat_all[:idx+1]
      rmw_g1 = rmw_all[:idx+1]
      r34_g1 = r34_all[:idx+1]
  
      tc_dict_g1['wind'] = wind_g1 
      tc_dict_g1['lat'] = lat_g1 
      tc_dict_g1['rmw'] = rmw_g1
      tc_dict_g1['r34'] = r34_g1

     if idx <= len(wind_all)-1:
      wind_g2 = wind_all[idx:]
      lat_g2 = lat_all[idx:]
      rmw_g2 = rmw_all[idx:]
      r34_g2 = r34_all[idx:]

      tc_dict_g2['wind'] = wind_g2
      tc_dict_g2['lat'] = lat_g2
      tc_dict_g2['rmw'] = rmw_g2
      tc_dict_g2['r34'] = r34_g2

    return tc_dict_g1, tc_dict_g2


def read_atcf_merged(filename, model_list, ini_date):

    tc_dict_all = {}

    col_list  = [2,         4,         5,          6,     7,      8,     9,      11,          13,     14,     15,     16,     19]
    name_list = ['iniDate', 'modelID', 'leadTime', 'lat', 'lon', 'vmax', 'pmin', 'radMarker', 'rad1', 'rad2', 'rad3', 'rad4', 'rmw']

    selected_hours = np.arange(0,120+6,6)
    df_all = pd.read_csv(filename, header=None, usecols=col_list,names=name_list)    
    df_all['iniDate'] =  df_all['iniDate'].astype(str)
    df_all['modelID'] =  df_all['modelID'].str.strip()

    df_all = df_all[ (df_all['iniDate'] == ini_date) & (df_all['leadTime'].isin(selected_hours)) ]
    #print df_all.head(n=10)

    if len(df_all) > 0:
     for model in model_list:
       tc_dict_ = {} 

       df = df_all[df_all['modelID']==model]

       if len(df) > 0:
         df1 = df[df['radMarker']==34]
         #print df.head(n=10)
         tc_dict_['leadTime'] = list(df1['leadTime'].astype(int))
         tc_dict_['lat'] = list(df1['lat'].str[:-1].astype(float)/10.)
         tc_dict_['lon'] =  list(360-df1['lon'].str[:-1].astype(float)/10.)
         tc_dict_['wind'] = list(df1['vmax'].astype(float)*0.51444)
         tc_dict_['pres'] = list(df1['pmin'].astype(float))
         tc_dict_['rad34_1'] = list(df1['rad1'].astype(float)*1.852)
         tc_dict_['rad34_2'] = list(df1['rad2'].astype(float)*1.852)
         tc_dict_['rad34_3'] = list(df1['rad3'].astype(float)*1.852)
         tc_dict_['rad34_4'] = list(df1['rad4'].astype(float)*1.852)
         tc_dict_['rmw'] = list(df1['rmw'].astype(float)*1.852)

         date_list=[]
         for hh in tc_dict_['leadTime']:  
           actual_time = datetime.strptime(ini_date,'%Y%m%d%H') + timedelta(hours=hh)
           dateSingle = actual_time.strftime('%Y%m%d%H')
           date_list.append(dateSingle)
         tc_dict_['date']=date_list

         #df2 = df[df['radMarker']==64]
         #tc_dict_['rad64_1'] = list(df2['rad1'].astype(float)*1.852)
         #tc_dict_['rad64_2'] = list(df2['rad2'].astype(float)*1.852)
         #tc_dict_['rad64_3'] = list(df2['rad3'].astype(float)*1.852)
         #tc_dict_['rad64_4'] = list(df2['rad4'].astype(float)*1.852)   
         tc_dict_all[model] = tc_dict_
 
       else:
         print (model, 'not found')
 
    return tc_dict_all

def read_atcf_obs(filename):

    print("Reading", filename)
    fo = open (filename, "r")

    lines = fo.readlines()

    date, lat, lon, wind, pres, rmw = [], [], [], [], [], []
    counter = 0
    for line in lines:

      line = str(line)
      fields = line.split(',')

      latSingle = int(fields[6][:-1])/10.0
      lonSingle = 360.-(int(fields[7][:-1])/10.0)
      dateSingle = fields[2][1:]

      hh = dateSingle[-2:]
      if hh in ['00', '06', '12', '18']:  # do not allow other hours
        date.append(dateSingle)
        lat.append(latSingle)
        lon.append(lonSingle)
        wind.append(0.5144*int(fields[8]))
        pres.append(int(fields[9]))

        # some rmw is missing from b-decks
        if len(fields)>=20:
           rmw.append(int(fields[19])*1.852)
        else:
           rmw.append(0.)

        if counter == 0:
           basin=fields[0]
           cycloneNum=fields[1].strip()
           warnDT=fields[2].strip()
           model=fields[4].strip()
        counter += 1

    tc_dict = {'basin'     : basin,
               'cycloneNum': cycloneNum,
               'warnDT'    : warnDT,
               'model'     : model,
               'date'      : date,
               'lat'       : lat,
               'lon'       : lon,
               'wind'      : wind,
               'pres'      : pres,
               'rmw'       : rmw
               }
    return tc_dict

def read_atcf(filename, isModel=True, read_wind_prof=False):

    print("Reading", filename)
    fo = open (filename, "r")

    lines = fo.readlines()

    date, lat, lon, wind, pres = [], [], [], [], []
    rmw = []
    rad34_1, rad34_2, rad34_3, rad34_4 = [], [], [], []
    #rad50_1, rad50_2, rad50_3, rad50_4 = [], [], [], []
    #rad64_1, rad64_2, rad64_3, rad64_4 = [], [], [], []

    counter = 0
    notNamed = True # to read obs storm name
    for line in lines:

        line = str(line)
        fields = line.split(',')

        if isModel:
          actual_time = datetime.strptime(fields[2][1:],'%Y%m%d%H') + timedelta(hours=int(fields[5]))
          dateSingle = actual_time.strftime('%Y%m%d%H')
        else:
          dateSingle = fields[2][1:]

        if int(fields[11]) == 34: 
           rad1, rad2, rad3, rad4 = float(fields[13]), float(fields[14]), float(fields[15]), float(fields[16]) 
        else: 
           rad1, rad2, rad3, rad4 = 0., 0., 0., 0.
 
        if dateSingle not in date:
           latSingle = int(fields[6][:-1])/10.0
           lonSingle = 360.-(int(fields[7][:-1])/10.0)

           date.append(dateSingle)
           lat.append(latSingle)
           lon.append(lonSingle)
           wind.append(0.5144*int(fields[8]))

           # for b-decks only
           if not isModel and notNamed:
              if len(fields)>=28 and int(fields[8]) >= 65:
                stormName=fields[27].strip()
                notNamed = False

           pres.append(int(fields[9]))
           rad34_1.append(rad1*1.852)                  
           rad34_2.append(rad2*1.852)   
           rad34_3.append(rad3*1.852)   
           rad34_4.append(rad4*1.852)  
          
           # some rmw is missing from b-decks 
           if len(fields)>=20:
             rmw.append(int(fields[19])*1.852)
           else:
             rmw.append(0.)

        # one time info - do it at first line
        if counter == 0:
           basin=fields[0]
           stormID=fields[1].strip()
           warnDT=fields[2].strip()
           model=fields[4].strip()
        counter += 1

    # one time info - do it at the last line
    if (not isModel) and (not notNamed):
       stormID=stormID+'-'+stormName

    if read_wind_prof:

       # get file name
       file_dir = os.path.dirname(filename)
       str_file = file_dir+'/*structure*'
       str_files = glob.glob(str_file)
       if len(str_files) != 1:
         print ('Warning: invalid structure filename')
       else:
         str_file = str_files[0]

       df_str = pd.read_csv(str_file, header=None)
       df_str.iloc[:,13:27] = df_str.iloc[:, 13:27].astype(float)/10.

       # ABOUT COL 11
       # 71: earth-relative winds
       # 72: storm-relative winds 
       # 81: Tangential winds, earth-relative
       # 82: Tangential winds, storm-relative
       # 91: Radial winds, earth-relative
       # 92: Radial winds, storm-relative

       df_str_new = df_str[(df_str.iloc[:,1] == int(stormID)) & (df_str.iloc[:,11] == 71)]
       df_str_NEE = df_str_new[(df_str_new.iloc[:,12].str.contains('NEE'))].iloc[:,13:27]
       df_str_SEE = df_str_new[(df_str_new.iloc[:,12].str.contains('SEE'))].iloc[:,13:27]
       df_str_SWE = df_str_new[(df_str_new.iloc[:,12].str.contains('SWE'))].iloc[:,13:27]
       df_str_NWE = df_str_new[(df_str_new.iloc[:,12].str.contains('NWE'))].iloc[:,13:27]

       wind_NEE = df_str_NEE.values.tolist()
       wind_SEE = df_str_SEE.values.tolist()
       wind_SWE = df_str_SWE.values.tolist()
       wind_NWE = df_str_NWE.values.tolist()

    tc_dict = {'basin'     : basin,
               'stormID'   : stormID,
               'warnDT'    : warnDT,
               'model'     : model,
               'date'      : date,
               'lat'       : lat,
               'lon'       : lon,
               'wind'      : wind,
               'pres'      : pres,
               'rmw'       : rmw,
               'rad34_1'   : rad34_1,
               'rad34_2'   : rad34_2,
               'rad34_3'   : rad34_3,
               'rad34_4'   : rad34_4
               }
    if read_wind_prof:

      tc_dict['wind_NWE'] = wind_NEE
      tc_dict['wind_SEE'] = wind_SEE
      tc_dict['wind_SWE'] = wind_SWE
      tc_dict['wind_NWE'] = wind_NWE

    return tc_dict

def read_atcf_2exp_withObs(atcf_dir, exp1, exp2, date):

  out_dict = {}

  dir_obs = '/net/mb/nmc13/verify/'

  storm_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32']

  for storm_sel in storm_list:

    year = date[:4]

    runname1 = atcf_dir+date[:8]+'.'+date[-2:]+'Z.C768r10n4_atl_new.nh.32bit.non-mono.'+exp1
    file1 = runname1+'/aal'+storm_sel+'*.dat'
    runname2 = atcf_dir+date[:8]+'.'+date[-2:]+'Z.C768r10n4_atl_new.nh.32bit.non-mono.'+exp2
    file2 = runname2+'/aal'+storm_sel+'*.dat'

    files1 = glob.glob(file1)
    files2 = glob.glob(file2)

    #print file1
    #print file2

    if len(files1) == 1 and len(files2) == 1:
      file1 = files1[0]
      file2 = files2[0]
 
      tc_dict1 = read_atcf(file1) 
      lon1 = tc_dict1['lon']
      lat1 = tc_dict1['lat']
      wind1 = tc_dict1['wind']
      date1 = tc_dict1['date']
 
      tc_dict2 = read_atcf(file2)   
      lon2 = tc_dict2['lon']
      lat2 = tc_dict2['lat']
      wind2 = tc_dict2['wind']
      date2 = tc_dict2['date']

      # ------ obs
      file1 = dir_obs + '/bal'+storm_sel+year+'.dat'
      #print file1

      files = glob.glob(file1)
      if len(files) == 1:
         filename = files[0]
         tc_dict = read_atcf(filename, False)
         lon = tc_dict['lon']
         lat = tc_dict['lat']
         wind_all = tc_dict['wind']
         date_all = tc_dict['date']

         date_obs = []
         lon_obs = []
         lat_obs = []
         wind_obs = []

         for i in range(len(date_all)):
             if (date_all[i] in date1) and (date_all[i] not in date_obs): # in forecast range
                lon_obs.append(lon[i]) 
                lat_obs.append(lat[i])
                wind_obs.append(wind_all[i])
                date_obs.append(date_all[i])

      # --- put model1, model2, obs together
      out_dict[storm_sel+'_lon_exp1'] = lon1
      out_dict[storm_sel+'_lat_exp1'] = lat1
      out_dict[storm_sel+'_lon_exp2'] = lon2
      out_dict[storm_sel+'_lat_exp2'] = lat2
      out_dict[storm_sel+'_lon_obs'] = lon_obs
      out_dict[storm_sel+'_lat_obs'] = lat_obs

      out_dict[storm_sel+'_date_exp1'] = date1
      out_dict[storm_sel+'_date_exp2'] = date2
      out_dict[storm_sel+'_date_obs'] = date_obs

      out_dict[storm_sel+'_wind_exp1'] = wind1
      out_dict[storm_sel+'_wind_exp2'] = wind2
      out_dict[storm_sel+'_wind_obs'] = wind_obs

  return out_dict

def read_atcf_merged_oneModel(filename, model_sel, ini_time_sel):

    print("Reading", filename)
    fo = open (filename, "r")

    lines = fo.readlines()

    date, lat, lon, wind, pres = [], [], [], [], []

    counter = 0

    for line in lines:

       line = str(line)
       fields = line.split(',')

       ini_time = fields[2][1:]
       model = fields[4].strip()
    
       if model == model_sel and ini_time == ini_time_sel: 
      
        actual_time = datetime.strptime(fields[2][1:],'%Y%m%d%H') + timedelta(hours=int(fields[5]))
        dateSingle = actual_time.strftime('%Y%m%d%H')

        if dateSingle not in date:
           latSingle = int(fields[6][:-1])/10.0
           lonSingle = 360.-(int(fields[7][:-1])/10.0)

           date.append(dateSingle)
           lat.append(latSingle)
           lon.append(lonSingle)
           wind.append(0.5144*int(fields[8]))
           pres.append(int(fields[9]))

       # one time info - do it at first line
       if counter == 0:
          basin=fields[0]
          stormID=fields[1].strip()
          counter = 1

    tc_dict = {'basin'     : basin,
               'warnDT'    : ini_time_sel,
               'model'     : model_sel,
               'date'      : date,
               'lat'       : lat,
               'lon'       : lon,
               'wind'      : wind,
               'pres'      : pres,
               }

    return tc_dict

def read_atcf_2exp_withObs_from_merged(atcf_dir, model1, model2, storm_id, date):

    out_dict = {}
    adeck = atcf_dir+'/a'+storm_id+'.dat'
    bdeck = atcf_dir+'/b'+storm_id+'.dat'
 
    tc_dict1 = read_atcf_merged_oneModel(adeck, model1, date) 
    lon1 = tc_dict1['lon']
    lat1 = tc_dict1['lat']
    wind1 = tc_dict1['wind']
    date1 = tc_dict1['date']
 
    tc_dict2 = read_atcf_merged_oneModel(adeck, model2, date)   
    lon2 = tc_dict2['lon']
    lat2 = tc_dict2['lat']
    wind2 = tc_dict2['wind']
    date2 = tc_dict2['date']

    # ------ obs
    tc_dict = read_atcf(bdeck, False)
    lon = tc_dict['lon']
    lat = tc_dict['lat']
    wind_all = tc_dict['wind']
    date_all = tc_dict['date']

    date_obs = []
    lon_obs = []
    lat_obs = []
    wind_obs = []

    for i in range(len(date_all)):
        if (date_all[i] in date1) and (date_all[i] not in date_obs): # in forecast range
           lon_obs.append(lon[i]) 
           lat_obs.append(lat[i])
           wind_obs.append(wind_all[i])
           date_obs.append(date_all[i])

    # --- put model1, model2, obs together
    out_dict['lon_'+model1] = lon1
    out_dict['lat_'+model1] = lat1
    out_dict['lon_'+model2] = lon2
    out_dict['lat_'+model2] = lat2
    out_dict['lon_obs'] = lon_obs
    out_dict['lat_obs'] = lat_obs

    out_dict['date_'+model1] = date1
    out_dict['date_'+model2] = date2
    out_dict['date_obs'] = date_obs

    out_dict['wind_'+model1] = wind1
    out_dict['wind_'+model2] = wind2
    out_dict['wind_obs'] = wind_obs

    return out_dict

def plot_track_atcf(ax, ccrs, file_name, color, add_date=False):

    tc_dict = read_atcf_obs(file_name)
    
    lon = tc_dict['lon']
    lat = tc_dict['lat']
    wind = tc_dict['wind']
    date = tc_dict['date']

    #print(lon)
    #print(wind)

    TSmask = np.array(wind) >= 17.5
    hurmask = np.array(wind) >= 32.5

    x = lon
    y = lat

    xs = np.ma.MaskedArray(x,mask=~TSmask)
    ys = np.ma.MaskedArray(y,mask=~TSmask)
    xh = np.ma.MaskedArray(x,mask=~hurmask)
    yh = np.ma.MaskedArray(y,mask=~hurmask)

    ax.plot(x,y,color=color,linewidth=1.,transform=ccrs)
    ax.plot(xs,ys,color=color,linewidth=2.,transform=ccrs)
    ax.plot(xh,yh,color=color,linewidth=4.,transform=ccrs)

    if add_date:
      yoffset = 0. #0.022*(m.ymax-m.ymin)/3

      for i in np.arange(0,len(x),2):
        xi=xs[i]
        yi=ys[i]
        datei=date[i]
        print(xi, yi, datei)
        ax.text(xi,yi+yoffset,datei[2:10],color=color,ha='right',fontsize=8,transform=ccrs)
        #ax.text(xi,yi,'o',fontsize=12,color=color,ha='center',va='center')

# <======================= end of ATCF related functions



