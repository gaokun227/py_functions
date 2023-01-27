import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import glob
import os
from scipy.stats import norm

#sys.path.append('/home/kng/Desktop/python_tool/py_functions/')
from functions import *

# --- class to process met-tc output

class process_mettc():

  def __init__(self):
     self.col_list = ['b','g','r','m','orange','c','y']
     self.hh_list = [12, 24, 36, 48, 72, 96, 120]

  def preprocess(self, filename):
     # --- read in data as a pd dataframe
     col_list =  [1,       2,        3,         5] # assuming caseNum = validNum
     self.df = pd.read_csv(filename, header=0, delimiter=r"\s+", usecols=col_list)
     #self.df['Abs_Error'] =  self.df['Error'].abs()

     # --- generate a list contains all model 4-digit IDs
     #self.modelID_list = self.df['modelID'].unique()

  def plot_stat(self, type, model_list, model_name_map, color_sel=[]):

    tc_dict = {} 
    for model in model_list:
      tc_dict[model] = self.df[ (self.df['COLUMN']==type) & (self.df['AMODEL']==model) ]['MEAN'].tolist() 
    tc_dict['case_num'] = self.df[ (self.df['COLUMN']==type) & (self.df['AMODEL']==model) ]['TOTAL'].tolist()

    ms = 12
    mec = 'w'
    lw = 3

    ft = 22
    ft1 = 20

    time = np.array(self.hh_list)

    case_num = tc_dict['case_num']

    if color_sel == []:
       colors = self.col_list
    else:
       colors = color_sel

    num_models = len(model_list)

    fig = plt.figure(figsize=(14,8))
    ax=plt.subplot(111)

    for model, col in zip(model_list, colors):
      print 'plotting', model, type
      label = model_name_map[model]
      ax.plot(time, tc_dict[model], marker='o', lw=lw, ms=ms, mec=mec, label = label, c=col)

    if type == 'ALTK_ERR' or type == 'CRTK_ERR':
        ax.axhline(y=0, c='k', lw=1)
        #low, high = plt.ylim()
        #bound = max(abs(low), abs(high))
        bound = 100.
        plt.ylim(-bound, bound)

    plt.legend(loc=0, fontsize=ft1)
    plt.grid()

    xticks = [1, 3, 4, 5, 6]
    xticklabels=[]
    for i in range(len(xticks)):
        xtick1 =  str(time[xticks[i]])+'h\n(' + str(case_num[xticks[i]]) + ')'
        xticklabels.append(xtick1)

    ax.set_xticks(time[xticks])
    ax.set_xticklabels(xticklabels)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(ft1)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(ft1)

    if type == 'TK_ERR':
       title = 'Track Error'
    elif type == 'ALTK_ERR':
       title = 'Along Track Bias'
    elif type == 'CRTK_ERR':
       title = 'Cross Track Bias'
    plt.title(title, fontsize=ft)
    #plt.xlabel('Forecast hour', fontsize=ft)
    plt.ylabel('nm', fontsize=ft)

    model_name_string='.'.join(model_list);
    fig.savefig(model_name_string+'_'+type+'.png', bbox_inches = 'tight')
    #plt.show()

# --- function to do TC verification 

def run_verify(work_dir, models, type, model_name_map, do_plot_only = False, model_sel=[], color_sel=[], atcf_dir = '../all/', sig_test_model_list=[]):

  if type == 'intensity':
    card_name = 'icard.'+models+'.al'
  elif type == 'track':
    card_name = 'tcard.'+models+'.al'
  elif type == 'radii':
    card_name = 'rcard.'+models+'.al'

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
    plot_stat(tc_dict, 'intensity_error', card_name, work_dir, model_name_map, model_sel, color_sel, sig_test_model_list)
    plot_stat(tc_dict, 'intensity_bias', card_name, work_dir, model_name_map, model_sel, color_sel)
  elif type == 'track':
    tc_dict = read_track_stat(work_dir+card_name+'.out')
    plot_stat(tc_dict, 'track_error', card_name, work_dir, model_name_map, model_sel, color_sel, sig_test_model_list)
    plot_stat(tc_dict, 'track_xbias', card_name, work_dir, model_name_map, model_sel, color_sel)
    plot_stat(tc_dict, 'track_ybias', card_name, work_dir, model_name_map, model_sel, color_sel)
  elif type == 'radii':
    tc_dict = read_radii_stat(work_dir+card_name+'.out2')
    plot_stat(tc_dict, 'R34',       card_name, work_dir, model_name_map, model_sel, color_sel)
    plot_stat(tc_dict, 'R34_error', card_name, work_dir, model_name_map, model_sel, color_sel)
    plot_stat(tc_dict, 'R34_bias',  card_name, work_dir, model_name_map, model_sel, color_sel)
    plot_stat(tc_dict, 'R64',       card_name, work_dir, model_name_map, model_sel, color_sel)
    plot_stat(tc_dict, 'R64_error', card_name, work_dir, model_name_map, model_sel, color_sel)
    plot_stat(tc_dict, 'R64_bias',  card_name, work_dir, model_name_map, model_sel, color_sel)

# ---

# The following tool is to gain insight on the error distribution, model differeces based on
# the intermediate results in 'xtr.dat' file

class stratify_TC_error():

  def __init__(self):
     self.col_list = ['b','g','r','m','orange','c','y']

  # generate a pd dataframe containing columns defined in name_list
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

  # show where the storms are located and their intensity distributions
  # assuming homogeneous model comparison
  # still work in progress
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

  # compare the error distributions in two models at lead times given in hh_list
  def compare_error_twoModels(self, modelA, modelB, hh_list):

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
 
         # do pdf 
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

  # The following functions are read_radii_statto 
  # obtain aggregated errors based on model and storm IDs at given lead time hh

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
      
  # plot storm-by-storm error in one model 
  def plot_by_storm_oneModel(self, type, modelID):

     stormID_list = []
     value_list = []
     count_list = []

     if type == 'intensity_bias':
         oneModelDict = self.bias_dict[modelID]
     elif 'error' in type:
         oneModelDict = self.error_dict[modelID]
     else:
         print 'wrong type!!!'
     for stormID, value in oneModelDict.iteritems():
         value_list.append(value)
         stormID_list.append(stormID)
         count_list.append(self.count_dict[modelID][stormID])

     percent_list = self.get_percent_contribution(value_list, count_list)
     stormID_list, value_list, count_list, percent_list = self.sort_lists(stormID_list, value_list, count_list, percent_list)  

     fig = plt.figure(figsize=(18,6))
     ax=plt.subplot(111)
     ft = 16
     ft1 = 12
     ft2 = 10

     x = np.arange(len(value_list)) 
     width = 0.35 

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
        for x1, p1 in zip(x, percent_list):
            plt.text(x1, bound, '%.1f' % p1 + '%', ha='center', va='bottom', fontsize=ft2)

     if 'bias' in type:
        plt.legend(loc='upper right', ncol=1,  fontsize=ft1)
     elif 'error' in type:
        plt.legend(loc='center right', ncol=1,  fontsize=ft1)

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
 
     ax.set_title(title_a + ' at '  + str(self.hh) + ' hr', fontsize=ft1) 
     ax.set_ylabel(unit, fontsize=ft1)

     filename = 'oneModel_'+modelID+'_'+type+'_'+str(self.hh)+'hr_by_storm'
     fig.savefig(filename+'.png',bbox_inches='tight')
     #plt.show()
                                         
  # plot storm-by-storm errors in two models                                  
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
         print 'wrong type!!!'
     for stormID, value1 in oneModelDict1.iteritems():
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

     ax.set_title(title_a + ' at '  + str(self.hh) + ' hr', fontsize=ft1)                   
     ax.set_ylabel(unit, fontsize=ft1)

     filename = 'twoModel_'+modelID1+'_vs_'+modelID2+'_'+type+'_'+str(self.hh)+'hr_by_storm'
     fig.savefig(filename+'.png',bbox_inches='tight')
     #plt.show()

  def plot_by_storm(self, type, selected_model_list):
     stormID_list = []

     fig = plt.figure(figsize=(14,6))
     ax=plt.subplot(111)
     ft = 16 
     ft1 = 12
 
     for i in range(len(selected_model_list)):
         modelID = selected_model_list[i]
         if type == 'intensity_bias':
            oneModelDict = self.bias_dict[modelID] 
         elif 'error' in type:
            oneModelDict = self.error_dict[modelID]
         else:
            print 'wrong type!!!'
         value_list = []
         for stormID, value in oneModelDict.iteritems():
             value_list.append(value)
             if i == 0:
                stormID_list.append(stormID)
         plt.scatter(range(len(value_list)), value_list, color=self.col_list[i], marker='o', s=40, edgecolors='w', label=modelID) 

     if 'bias' in type:
        ax.axhline(y=0, c='k', lw=1)
        low, high = plt.ylim()
        bound = max(abs(low), abs(high))
        plt.ylim(-bound, bound)

     plt.legend(loc='center right', ncol = 1,  fontsize=ft1)

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
     fig.savefig('test.png',bbox_inches='tight')

# ---

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
       tc_dict[L[0]] = map(int, L[2:])
    elif counter >= error_str and counter <= error_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_intensity_error'] = map(float,L[2:])
    elif counter >= std_str and counter <= std_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_error_std'] = map(float,L[2:])
    elif counter >= bias_str and counter <= bias_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_intensity_bias'] = map(float,L[2:])
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
       tc_dict[L[0]] = map(int, L[2:]) # Case number
    elif counter >= error_str and counter <= error_str+num_model-1: 
       L = line.split()
       tc_dict[L[0]+'_track_error'] = map(float,L[2:])
    elif counter >= std_str and counter <= std_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_error_std'] = map(float,L[2:])
    elif counter >= xbias_str and counter <= xbias_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_track_xbias'] = map(float,L[2:])
    elif counter >= ybias_str and counter <= ybias_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_track_ybias'] = map(float,L[2:])
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
       tc_dict[L[0]+'_R34'] = map(int, L[2:]) # Case number
    elif counter >= r34_str and counter <= r34_str+num_model-1: 
       L = line.split()
       tc_dict[L[0]+'_R34'] = map(float,L[2:])
    elif counter >= r34_err_str and counter <= r34_err_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_R34_error'] = map(float,L[2:])
    elif counter >= r34_bias_str and counter <= r34_bias_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_R34_bias'] = map(float,L[2:])

    # get r50 stat
    if counter == r50_str+num_model:
       L = line.split()
       tc_dict[L[0]+'_R50'] = map(int, L[2:]) # Case number
    elif counter >= r50_str and counter <= r50_str+num_model-1: 
       L = line.split()
       tc_dict[L[0]+'_R50'] = map(float,L[2:])
    elif counter >= r50_err_str and counter <= r50_err_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_R50_error'] = map(float,L[2:])
    elif counter >= r50_bias_str and counter <= r50_bias_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_R50_bias'] = map(float,L[2:])

    # get r64 stat
    if counter == r64_str+num_model:
       L = line.split()
       tc_dict[L[0]+'_R64'] = map(int, L[2:]) # Case number
    elif counter >= r64_str and counter <= r64_str+num_model-1: 
       L = line.split()
       tc_dict[L[0]+'_R64'] = map(float,L[2:])
    elif counter >= r64_err_str and counter <= r64_err_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_R64_error'] = map(float,L[2:])
    elif counter >= r64_bias_str and counter <= r64_bias_str+num_model-1:
       L = line.split()
       tc_dict[L[0]+'_R64_bias'] = map(float,L[2:])
    counter += 1

  #print r34_str, r34_end, r34_err_str, r34_bias_str
  #print num_model
  #print tc_dict['#CASES_R64']

  return tc_dict

# ---

def plot_stat(tc_dict, keyword, card_name, pic_dir, model_name_map, model_sel, color_sel, sig_test_model_list=[]):

    ms = 12
    mec = 'w'
    lw = 3

    ft = 22
    ft1 = 20

    time = np.array([12, 24, 36, 48, 72, 96, 120])
 
    if 'R34' in keyword or 'R64' in keyword or 'R50' in keyword:
        case_num = tc_dict['#CASES_'+keyword[:3]]
    else:
        case_num = tc_dict['#CASES']

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
       print 'doing significance test'
       print sig_test_model_list 
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
      print 'plotting', models[i]+'_'+keyword
      label = model_name_map[models[i]]
      ax.plot(time, tc_dict[models[i]+'_'+keyword], marker='o', lw=lw, ms=ms, mec=mec, label = label, c=colors[i])
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
    
    xticks = [1, 3, 4, 5, 6]
    xticklabels=[]
    for i in range(len(xticks)):
        xtick1 =  str(time[xticks[i]])+'h\n(' + str(case_num[xticks[i]]) + ')'
        xticklabels.append(xtick1)
    
    ax.set_xlim([0, 132])
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
       title = 'Mean 34-kt wind radii'
    elif keyword == 'R34_error':
       title = 'Mean 34-kt wind radii error'
    elif keyword == 'R34_bias':
       title = 'Mean 34-kt wind radii bias'
    elif keyword == 'R64':
       title = 'Mean 64-kt wind radii'
    elif keyword == 'R64_error':
       title = 'Mean 64-kt wind radii error'
    elif keyword == 'R64_bias':
       title = 'Mean 64-kt wind radii bias'

    if 'intensity' in keyword:
       yname = 'knots'
    elif 'track' in keyword or 'R34' in keyword or 'R64' in keyword or 'R50' in keyword:
       yname = 'nm' 

    plt.title(title, fontsize=ft)
    #plt.xlabel('Forecast hour', fontsize=ft)
    plt.ylabel(yname, fontsize=ft)

    model_name_string='.'.join(models);
    #fig.savefig('/work/Kun.Gao/trak_ver/run_2022/track_5model.png')
    fig.savefig(pic_dir+model_name_string+'_'+keyword+'.png', bbox_inches = 'tight')
    #plt.show()

# --- 

def flatten_list(listOflist):
  flat_list = []
  for sublist in listOflist:
    for item in sublist:
        flat_list.append(item)
  return flat_list

# ---

def sel_var(var_all, vmax_all, wind_th1, wind_th2):
    var_sel = []
    for i in range(len(vmax_all)):
        if vmax_all[i]  >= wind_th1 and vmax_all[i]  <= wind_th2:
           var_sel.append(var_all[i])
    return np.array(var_sel)

# --- function to average wind radii

def detect_wind_radii(wind_list, target):
   
    rad = [10,25,50,75,100,125,150,200,250,300,350,400,450,500]
    idx_max = wind_list.index(max(wind_list))

    # we assume RMW >=300 is a bad case 
    if wind_list[-1] > target or max(wind_list) < target or rad[idx_max] > 300:
       result = np.nan 
    else:
       for i in range(idx_max, len(rad)-1):
           if wind_list[i] == target:
              result = rad[i]
           elif wind_list[i] > target and wind_list[i+1] <= target:
              slope = (wind_list[i+1] - wind_list[i])/float((rad[i+1] - rad[i])) 
              result = rad[i] + 1./slope * (target - wind_list[i]) 
    return result 

def detect_wind_radii_from_dict(tc_dict, target):
      wind1 = tc_dict['wind_NWE']
      wind2 = tc_dict['wind_SEE']
      wind3 = tc_dict['wind_SWE']
      wind4 = tc_dict['wind_NWE']
      rad1 = []
      rad2 = []
      rad3 = []
      rad4 = []
      for i in range(len(wind1)):
          rad1.append(detect_wind_radii(wind1[i], target))
          rad2.append(detect_wind_radii(wind2[i], target))
          rad3.append(detect_wind_radii(wind3[i], target))
          rad4.append(detect_wind_radii(wind4[i], target))
      return rad1, rad2, rad3, rad4

def mean_wind_radii(rad1, rad2, rad3, rad4):
    r1 = np.array(rad1)
    r2 = np.array(rad2)
    r3 = np.array(rad3)
    r4 = np.array(rad4)

    rad_new = np.stack((r1,r2,r3,r4))

    rad_new[rad_new<1] = np.nan
    rad_new[rad_new>900] = np.nan
   
    return np.nanmean(rad_new, axis=0)

# --- function to get wind radii

def get_wind_radius(vel_azi, dr, wind_th):
    nt, nr = np.shape(vel_azi)
    radius = np.arange(nr)*dr
    rad = np.zeros(nt)
    for t in np.arange(nt):
      vel1 = vel_azi[t,:]
      if np.max(vel1) < wind_th:
          rad[t] = np.nan
      else:
        rind1 = find_nearest(vel1,np.max(vel1))
        rind = find_nearest(vel1[rind1:],wind_th)
        rad[t] = radius[rind1+rind]
    return rad

# --- function to selected atcf records

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

# --- function to read atcf

def read_atcf_obs(filename):

    print("Reading", filename)
    fo = open (filename, "r")

    lines = fo.readlines()

    forecast_hour, lat, lon, wind, pres = [], [], [], [], []
    counter = 0
    for line in lines:

        line = str(line)
        fields = line.split(',')

        latSingle = int(fields[6][:-1])/10.0
        lonSingle = 360.-(int(fields[7][:-1])/10.0)
        hourSingle = fields[2]
        forecast_hour.append(hourSingle)
        lat.append(latSingle)
        lon.append(lonSingle)
        wind.append(0.5144*int(fields[8]))
        pres.append(int(fields[9]))

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
               'forecast_hour': forecast_hour,
               'lat'       : lat,
               'lon'       : lon,
               'wind'      : wind,
               'pres'      : pres
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
         print 'Warning: invalid structure filename'
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

########################################################################
# funciton to read matching obs and mod tc records from 
# files like 2017.NAtl.11.txt 
########################################################################

def read_match(filename):

  all_pred={}

  if os.path.exists(filename):
 
    print("Reading", filename)

    # --- initialize var
    title = ''
    obs_date = []
    obs_lon = []
    obs_lat = []
    obs_wind = []
    obs_pres = []

    mod_date = []
    mod_lon = []
    mod_lat = []
    mod_wind = []
    mod_pres = []

    # --- start reading
 
    fo = open (filename, "r")
    lines = fo.readlines()

    for line in lines:

     if "+++" in line:
       L = line.split()
       title = str(L[1])
       if "forecast" in line:
          ini_time = str(L[2])
 
       # --- wrap one forecast 
       if title == "overlap_BT_for" and len(obs_lon)>0 and len(mod_lon)>0:

          tc_stack = np.zeros((10,len(obs_lon))) 
          tc_stack[0,:] = np.array(obs_date)
          tc_stack[1,:] = np.array(obs_lon)
          tc_stack[2,:] = np.array(obs_lat)
          tc_stack[3,:] = np.array(obs_wind)
          tc_stack[4,:] = np.array(obs_pres)
          tc_stack[5,:] = np.array(mod_date)
          tc_stack[6,:] = np.array(mod_lon)
          tc_stack[7,:] = np.array(mod_lat)
          tc_stack[8,:] = np.array(mod_wind)
          tc_stack[9,:] = np.array(mod_pres)
          all_pred[ini_time] = tc_stack
          
          # now let's start a new cycle
          obs_date = []
          obs_lon = []
          obs_lat = []
          obs_wind = []
          obs_pres = []

          mod_date = []
          mod_lon = []
          mod_lat = []
          mod_wind = []
          mod_pres = []
 
     else:
      if title == "overlap_BT_for":
         L = line.split()
         obs_date += [float(L[0])/100.] 
         obs_lon  += [float(L[1])]
         obs_lat  += [float(L[2])]
         obs_pres += [float(L[3])]
         obs_wind += [float(L[4])]

      if title == "forecast":
         L = line.split()
         mod_date += [float(L[0])/100.] 
         mod_lon += [float(L[1])]
         mod_lat += [float(L[2])]
         mod_pres += [float(L[3])]
         mod_wind += [float(L[4])]

    #Ok, finished reading all lines but not done yet
    if len(obs_lon)>0 and len(mod_lon)>0:

          tc_stack = np.zeros((10,len(obs_lon)))
          tc_stack[0,:] = np.array(obs_date)
          tc_stack[1,:] = np.array(obs_lon)
          tc_stack[2,:] = np.array(obs_lat)
          tc_stack[3,:] = np.array(obs_wind)
          tc_stack[4,:] = np.array(obs_pres)
          tc_stack[5,:] = np.array(mod_date)
          tc_stack[6,:] = np.array(mod_lon)
          tc_stack[7,:] = np.array(mod_lat)
          tc_stack[8,:] = np.array(mod_wind)
          tc_stack[9,:] = np.array(mod_pres)
          all_pred[ini_time] = tc_stack

    fo.close()
    
  return all_pred 

########################################################################
# funciton to plot forecasts 
########################################################################

def transfer_time(time_s, time_ref):

    hours = []

    #note time_s is ini array; time_ref is str
    date_ref = dt.datetime.strptime(time_ref, "%Y%m%d%H")

    for i in np.arange(len(time_s)):
        #print str(int(100*time_s[i]))
        date1 = dt.datetime.strptime(str(int(100*time_s[i])), "%Y%m%d%H")
        diff = date1 - date_ref
        hours+= [diff.days*24+1./3600*diff.seconds]
    #print time_s, time_ref
    #print hours
    return np.array(hours)

def plot_multi_forecast(stack1, stack2, stack3, stack4, ini_time, storm_id, exp1, exp2, exp3, exp4, pic_dir):

          # --- exact info
          t_obs=stack1[0,:]
          x_obs=stack1[1,:]
          y_obs=stack1[2,:]
          w_obs=stack1[3,:]
          p_obs=stack1[4,:]

          t_mod1=stack1[5,:]
          x_mod1=stack1[6,:]
          y_mod1=stack1[7,:]
          w_mod1=stack1[8,:]
          p_mod1=stack1[9,:]

          t_mod2=stack2[5,:]
          x_mod2=stack2[6,:]
          y_mod2=stack2[7,:]
          w_mod2=stack2[8,:]
          p_mod2=stack2[9,:]

          time1 = transfer_time(t_mod1, ini_time) 
          time2 = transfer_time(t_mod2, ini_time)

          do_tc3 = False
          do_tc4 = False

          if np.size(stack3)>1:
             do_tc3 = True
             t_mod3=stack3[5,:]
             x_mod3=stack3[6,:]
             y_mod3=stack3[7,:]
             w_mod3=stack3[8,:]
             p_mod3=stack3[9,:]
             time3 = transfer_time(t_mod3, ini_time)

          if np.size(stack4)>1:
             do_tc4 = True
             t_mod4=stack4[5,:]
             x_mod4=stack4[6,:]
             y_mod4=stack4[7,:]
             w_mod4=stack4[8,:]
             p_mod4=stack4[9,:]
             time4 = transfer_time(t_mod4, ini_time)

          # --- make plot

          ft = 22
          ft1= 18

          col1 = 'orange'
          col2 = 'red'
          col3 = 'g'
          col4 = 'b'

          marker = 'o'
          ms = 7
          mec = 'w'

          xmin = 0-3 
          xmax = 120+3 
          dx = 12
          xticks = np.arange(0, 120+dx, dx)

          title = storm_id + ' Forecast Initialized on ' + ini_time + '\n'
          
          plt.close('all')
          fig = plt.figure(figsize = (8,22))

          ax1 = plt.subplot(311)

          ax1.plot(x_obs,   y_obs,  'k', lw = 2., label = 'Obs',   marker = marker, ms = ms, mec = mec)
          ax1.plot(x_mod1, y_mod1, col1, lw = 2., label = exp1,    marker = marker, ms = ms, mec = mec)
          ax1.plot(x_mod2[:len(x_mod1)], y_mod2[:len(x_mod1)], col2, lw = 2., label = exp2,    marker = marker, ms = ms, mec = mec)
          if do_tc3:
           ax1.plot(x_mod3[:len(x_mod1)], y_mod3[:len(x_mod1)], col3, lw = 2., label = exp3,    marker = marker, ms = ms, mec = mec) 
          if do_tc4:
           ax1.plot(x_mod4[:len(x_mod1)], y_mod4[:len(x_mod1)], col4, lw = 2., label = exp4,    marker = marker, ms = ms, mec = mec)

          ax1.grid(True)
          ax1.legend(loc=0,frameon=False,fontsize=ft1)

          ax1.set_xlabel('Lon (deg)',fontsize=ft)
          ax1.set_ylabel('Lat (deg)',fontsize=ft)
          ax1.set_title(title, fontsize=ft)

          for tick in ax1.xaxis.get_major_ticks():
              tick.label.set_fontsize(ft1)
          for tick in ax1.yaxis.get_major_ticks():
              tick.label.set_fontsize(ft1)

          ax1 = plt.subplot(312)

          ax1.plot(time1, w_obs,   'k',  lw = 2., marker = marker, ms = ms, mec = mec)
          ax1.plot(time1, w_mod1,  col1, lw = 2., marker = marker, ms = ms, mec = mec)
          ax1.plot(time2, w_mod2,  col2, lw = 2., marker = marker, ms = ms, mec = mec)
          if do_tc3:
             ax1.plot(time3, w_mod3,  col3, lw = 2., marker = marker, ms = ms, mec = mec)
          if do_tc4:
             ax1.plot(time4, w_mod4,  col4, lw = 2., marker = marker, ms = ms, mec = mec)

          ax1.grid(True)
          ax1.set_ylabel('Wind (m/s)',fontsize=ft)

          ax1.set_xlim([xmin, xmax])
          ax1.set_xticks(xticks)

          for tick in ax1.xaxis.get_major_ticks():
              tick.label.set_fontsize(ft1)
          for tick in ax1.yaxis.get_major_ticks():
              tick.label.set_fontsize(ft1)

          ax1 = plt.subplot(313)
          ax1.plot(time1, p_obs,   'k',  lw = 2., marker = marker, ms = ms, mec = mec)
          ax1.plot(time1, p_mod1,  col1, lw = 2., marker = marker, ms = ms, mec = mec)
          ax1.plot(time2, p_mod2,  col2, lw = 2., marker = marker, ms = ms, mec = mec)
          if do_tc3:
             ax1.plot(time3, p_mod3,  col3, lw = 2., marker = marker, ms = ms, mec = mec)
          if do_tc4:
             ax1.plot(time4, p_mod4,  col4, lw = 2., marker = marker, ms = ms, mec = mec)

          ax1.grid(True)
          ax1.set_ylabel('Pres (mb)',fontsize=ft)

          ax1.set_xlim([xmin, xmax])
          ax1.set_xticks(xticks)

          for tick in ax1.xaxis.get_major_ticks():
              tick.label.set_fontsize(ft1)
          for tick in ax1.yaxis.get_major_ticks():
              tick.label.set_fontsize(ft1)
          
          fig.savefig(pic_dir + storm_id + '.' + ini_time + '.png',bbox_inches='tight')
          #plt.show()
