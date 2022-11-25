#import packages
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import datetime
from dateutil.relativedelta import relativedelta

import random
import itertools
from tqdm import tqdm

# from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
import matplotlib.pyplot as plt
tjd_blue = "#1F3A93"
tjd_red = "#CF000F"
tjd_gray = "#bfbfbf"
tjd_black = "#000000"
plt.style.use('seaborn')
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

from pprint import pprint

from chinese_calendar import is_holiday, is_workday

#function
####修改时间
def month_delta(date,delta):
    if type(date) == str:
        date = datetime.datetime.strptime(date, '%Y-%m-%d')+ relativedelta(months= delta)
    else:
        date = date + relativedelta(months=+ delta)
    if date.month in [1,3,5,7,8,10,12]:
        date = date.replace(day=31)
    elif date.month not in [1,3,5,7,8,10,12] and date.month != 2:
        date = date.replace(day=30)
    elif date.month == 2 and date.year % 4 == 0:
        date = date.replace(day=29)
    else : 
        date = date.replace(day=28)
    return date.strftime('%Y-%m-%d')
####模型评估
def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    actual = np.array(actual).reshape(actual.shape[0],)
    predicted = np.array(predicted).reshape(predicted.shape[0],)
    
    return actual - predicted

def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))

def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))

def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    actual = np.array(actual).reshape(actual.shape[0],)
    predicted = np.array(predicted).reshape(predicted.shape[0],)
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))
    
def aq_DataProcess(df):
    monthly_aq = df.resample('M').mean()
    aq_yoy = (monthly_aq / monthly_aq.shift(12) - 1) * 100
    aq_yoy = aq_yoy['2015-01-31':"2022-10-31"]
    return aq_yoy

def gen_train(no2_yoy,iav_yoy,train_date):
    x_train = no2_yoy[:train_date]
    y_train = iav_yoy[:train_date]
    return x_train,y_train

def gen_valid(no2_yoy,iav_yoy,train_date):
    valid_start = month_delta(train_date,1)
    valid_end = month_delta(valid_start,6)
    x_valid = no2_yoy[valid_start:valid_end]
    y_valid = iav_yoy[valid_start:valid_end]
    return x_valid,y_valid

def gen_test(no2_yoy,iav_yoy,valid_end,test_end):
    test_start = month_delta(valid_end,1)
    test_end = "2022-08-31"
    x_test = no2_yoy[test_start:test_end]
    y_test = iav_yoy[test_start:test_end]
    return x_test,y_test

def findsubsets(s, n):
    return list(itertools.combinations(s, n))
    
def gen_result_dic(no2_yoy,iav_yoy,no2_cities):
    result_dic = {"城市":[],"训练集RMSE":[],"验证集RMSE":[],"测试集RMSE":[],"训练集MDA":[],"验证集MDA":[],"测试集MDA":[]}
    #max_n = no2_cities.shape[1]
    for n in tqdm(range(1,10)):
        group_city = findsubsets(no2_cities,n)
        for g in group_city :
            no2_group_city = no2_yoy[list(g)]
            result_dic["城市"].append(list(g))
            #train_test_spilt
            train_date = '2021-01-31'#train
            x_train,y_train = gen_train(no2_group_city,iav_yoy,train_date)
            #valid
            x_valid,y_valid = gen_valid(no2_group_city,iav_yoy,train_date)
            #test
            test_start = month_delta(valid_end,1)
            test_end = "2022-08-31"
            x_test,y_test = gen_test(no2_group_city,iav_yoy,valid_end,test_end)
            
            #fit and predict
            model = linear_model.Lasso()
            model.fit(x_train, y_train)
            y_pred_train = model.predict(x_train)
            
            rmse_train = rmse(y_train,y_pred_train)
            mda_train = mda(y_train,y_pred_train)
            result_dic["训练集RMSE"].append(rmse_train)
            result_dic["训练集MDA"].append(mda_train)
            
            #vaild rmse
            y_pred_valid = model.predict(x_valid.loc[y_valid.index])
            rmse_valid = rmse(y_valid,y_pred_valid)
            #vaild mda
            mda_valid = mda(y_valid,y_pred_valid)
            result_dic["验证集RMSE"].append(rmse_valid)
            result_dic["验证集MDA"].append(mda_valid)

            #test rmse
            y_pred_test = model.predict(x_test.loc[y_test.index])
            rmse_test = rmse(y_test,y_pred_test)
            #test mda
            mda_test = mda(y_test,model.predict(x_test.loc[y_test.index]))

            result_dic["测试集RMSE"].append(rmse_test)
            result_dic["测试集MDA"].append(mda_test)
    return result_dic

cities = [
    "青岛","南京","扬州","常州",
    "连云港","镇江","上海","宁波",
    "杭州","惠州","福州","乌鲁木齐",
    '珠海','银川',"天津","北京","兰州",
         ]
no2_raw = pd.read_excel(r"C:\Users\DELL\Desktop\腾景\data_update\air_quality_no2.xlsx", index_col=0)
no2_cities = no2_raw[cities]

no2_cities = no2_cities.replace(' ', np.nan)
no2_cities = no2_cities.astype(float)
no2_cities = no2_cities.fillna(method = 'ffill')
no2_cities.index = pd.to_datetime(no2_cities.index)

no2_yoy = aq_DataProcess(no2_cities)
iav_yoy = pd.read_excel(r"C:\Users\DELL\Desktop\石油与天然气增加值.xlsx", index_col=0)['2015-01-31':]

result_dic = gen_result_dic(no2_yoy,iav_yoy,no2_cities)
result_dic = pd.DataFrame(result_dic)
result_dic.to_excel('C:\\Users\\DELL\\Desktop\\test.xlsx',index = False)

