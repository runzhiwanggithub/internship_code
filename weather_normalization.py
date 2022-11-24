#import packages
import os
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import datetime
from dateutil.relativedelta import relativedelta

import random
from tqdm import tqdm

import matplotlib.pyplot as plt
tjd_blue = "#1F3A93"
tjd_red = "#CF000F"
tjd_gray = "#bfbfbf"
tjd_black = "#000000"
plt.style.use('seaborn')
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  

import warnings
warnings.filterwarnings('ignore')

from pprint import pprint
from chinese_calendar import is_holiday, is_workday

#functions:
def cac_humid(d,t):
    d = (d-32)/1.8
    t = (t-32)/1.8
    above = (17.625 * d)/(243.04 + d)
    beneath = (17.625 * t)/(243.04 + t)
    return 100 * np.exp(above)/np.exp(beneath)

def extract_month(date):
    if type(date) == str:
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
    date_month = str(date.month) +'月'
    return date_month

def julian_day(date):
    
    if type(date) == str:
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
    tt = date.timetuple()
    julian_day = tt.tm_yday
    
    return julian_day
    
def climate_fillna(df):
    #缺失值填补
    for col in df.columns:
        for ind in range(len(df)):
            if df.loc[ind,col] == 999.9:
                df.loc[ind,col] = np.nan
    
    df["WDSP"] = df["WDSP"].fillna(method="ffill")
    df["MXSPD"] = df["MXSPD"].fillna(method="ffill")
    return df

def gen_humidity(df):
    df["HUMID"] = np.nan
    #generate humidity
    for ind in range(len(df)):
        d_ind = float(df.loc[ind,"DEWP"])
        t_ind = float(df.loc[ind,"TEMP"])
        df.loc[ind,"HUMID"] = cac_humid(d_ind,t_ind)
    return df

def gen_month(df):
    #generate month
    df["month"] = np.nan
    for ind in range(len(df)):
        df.loc[ind,"month"] = extract_month(df.loc[ind,"DATE"])
    return df

def is_weekday(df):
    #generate weekday
    df["weekday"] = np.nan
    for ind in range(len(df)):
        df.loc[ind,"weekday"] = int(is_workday(df.loc[ind,"DATE"]))
    return df

def gen_julian_day(df):
    df["julian_day"] = np.nan
    for ind in range(len(df)):
        df.loc[ind,"julian_day"] = julian_day(df.loc[ind,"DATE"])
    return df

def is_covid(df):
    #generate pre_after
    df["pre_after"] = np.nan
    for ind in range(len(df)):
        if df.loc[ind,"DATE"] < datetime.datetime.strptime('2020-01-23', '%Y-%m-%d'):
            df.loc[ind,"pre_after"] = 0
        else:
            df.loc[ind,"pre_after"] = 1
    return df

def set_index(df):
    #set index
    df = df.set_index('DATE')
    df = df[["CITY","month","weekday","julian_day","pre_after","TEMP","WDSP","MXSPD","HUMID"]]
    return df

def get_dummy(df):
    df = pd.get_dummies(df, prefix='month', prefix_sep='.', 
                            columns=['month'])
    return df

def same_ind(df,no2_raw):
    all_index = list(set(df.index).intersection(set(no2_raw.index)))
    all_index = sorted(all_index)
    df = df.loc[all_index,:]
    return df

def intersection(df):
    all_index = list(set(df.index).intersection(set(no2_cities.index)))
    all_index = sorted(all_index)
    df = df.loc[all_index,:]
    return df
    
def process_climate(file_lst):
    df_lst = []
    for file in tqdm(file_lst):
        temp_path = flie_path + "\\" + file
        climate_raw = pd.read_excel(temp_path)
    #添加city
        climate_raw["CITY"] = np.nan
        climate_raw["CITY"] = climate_raw["CITY"].apply(lambda x : file[:-7]) 
    #处理缺失值
        climate_raw = climate_fillna(climate_raw)
    #generate humidity
        climate_raw = gen_humidity(climate_raw)
    #generate month
        climate_raw = gen_month(climate_raw)
    #is_workday
        climate_raw = is_weekday(climate_raw)
    #generate 
        climate_raw = gen_julian_day(climate_raw)
    #is_covid
        climate_raw = is_covid(climate_raw)
    #set_index
        climate_raw = set_index(climate_raw)
    #get_dummy
        climate_raw = get_dummy(climate_raw)
        df_lst.append(climate_raw)
    return df_lst

def merge_resd(resd_lst):
    data = resd_lst[0]
    for df in tqdm(resd_lst[1:]):
        
        all_index = list(set(df.index).intersection(set(data.index)))
        all_index = sorted(all_index)
        
        data = data.loc[all_index,:]
        df = df.loc[all_index,:]
        
        data = pd.concat([data,df],axis = 1)
    return data

def aq_DataProcess(df):
    monthly_no2 = df.resample('M').mean()
    no2_yoy = (monthly_no2 / monthly_no2.shift(12) - 1) * 100
    no2_yoy = no2_yoy["2014-12-31":]
    return no2_yoy
    
def rf_tuning(df,no2_temp,random_grid):
    
    rf = RandomForestRegressor(random_state = 42)
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter = 100, scoring='neg_mean_absolute_error',
                                   cv = 5, verbose=2, random_state=42, n_jobs=-1,
                                   return_train_score=True)
    rf_random.fit(df.iloc[:,1:], no2_temp)
    
    return rf_random.best_params_
    
def sample_met(df,ind):
    day_1 = df.loc[ind,"julian_day"]
    if day_1 <= 14 :
        subset = df[(df["julian_day"]>=365 + day_1-14)|(df["julian_day"]<= day_1 +14)]
        met_sample = subset.sample(n=1)
    if (day_1 > 14 and day_1<352):
        subset = df[(df["julian_day"]>= day_1-14) & (df["julian_day"]<= day_1 +14)]
        met_sample = subset.sample(n=1)
    if day_1 >=352:
        subset = df[(df["julian_day"]>= day_1-14) | (df["julian_day"]<= day_1 +14-365)]
        met_sample = subset.sample(n=1)
    return met_sample

def generate_met(df):
    df_met = pd.DataFrame(columns=df.columns)
    for ind in df.index:
        met_sample = sample_met(df,ind)
        df_met = pd.concat([df_met,met_sample])
    return df_met

def generate_resample(re_sample_df,df_met):
    
    re_sample_df["TEMP"] = df_met["TEMP"].values
    re_sample_df["WDSP"] = df_met["WDSP"].values
    re_sample_df["MXSPD"] = df_met["MXSPD"].values
    re_sample_df["HUMID"] = df_met["HUMID"].values
    
    return re_sample_df

def prediction(n,df,rf,pred,re_sample_df):    
    for i in tqdm(range(n)):
        df_met = generate_met(df)
        ### Replaced old MET by generated MET
        re_sample_df = generate_resample(re_sample_df,df_met)
        prediction = rf.predict(re_sample_df)
        prediction = pd.DataFrame(prediction,columns = [str(i)])
        pred = pd.concat([pred,prediction],axis = 1)
    pred = pred.set_index("DATE")
    column = df.loc["2014-01-01","CITY"]
    pred[column] = pred.mean(axis=1)
    pred = pred[[column]]
    return pred
    
#load data
cities = ['上海', '广州', '重庆', '西安', '武汉', '长沙', '杭州', '南昌', '天津']
no2_raw = pd.read_excel(r"C:\Users\DELL\Desktop\腾景\data_update\air_quality_no2.xlsx", index_col=0)
no2_cities = no2_raw[cities]

no2_cities = no2_cities.replace(' ', np.nan)
no2_cities = no2_cities.astype(float)
no2_cities = no2_cities.fillna(method = 'ffill')
no2_cities.index = pd.to_datetime(no2_cities.index)

print(no2_cities.shape)
print(no2_cities.isnull().sum())

flie_path = r'C:\Users\DELL\Desktop\climate_data'
file_lst = os.listdir(flie_path)
df_lst = process_climate(file_lst)

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest #default 200 2000
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 800, num = 11)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 30, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]#default 2 5 10
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
               
wnorm_lst = []
for df in tqdm(df_lst):
    #generate df
    df = df_lst[0]
    df = intersection(df)
    #reproduce df
    re_sample_df = df.drop("julian_day",axis = 1)
    re_sample_df = df.drop("CITY",axis = 1)
    #gennerate no2
    city = df.loc["2014-01-01","CITY"]
    no2_temp = no2_cities.loc[df.index,[city]]
    #generate pred
    pred = pd.DataFrame(df.index,columns = ["DATE"])
    #generate met
    df_met = generate_met(df)
    #search best_params
    best_params = rf_tuning(re_sample_df,no2_temp,random_grid)
    print("Random_Grid_Search_completed")
    #Using the best parameters
    rf = RandomForestRegressor(n_estimators = best_params["n_estimators"], max_features = 'sqrt', 
                               max_depth = best_params["max_depth"], min_samples_leaf = best_params["min_samples_leaf"],
                               min_samples_split = best_params["min_samples_split"],
                               random_state = 42)
    rf.fit(re_sample_df,no2_temp)
    #return pred
    pred = prediction(10,df,rf,pred,re_sample_df)
    wnorm_lst.append(pred)

data = merge_resd(wnorm_lst)
data.to_pickle(r'C:\Users\DELL\Desktop\resd_merge.pkl')
no2_yoy = aq_DataProcess(data)
no2_yoy.to_excel(r"C:\Users\DELL\Desktop\平减数据.xlsx")
