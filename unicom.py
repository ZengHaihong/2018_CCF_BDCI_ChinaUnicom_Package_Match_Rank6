import os
import time
import copy
from scipy import sparse
from tqdm import *
from sklearn.feature_extraction.text import CountVectorizer

# add this feature function at 20:53pm ,2018-11-10
# ************************************************************
def dec(row):
    return row * 100 - int(row * 10) * 10
def dec1(row):
    return int(row * 10) - int(row) * 10
def dec2(row):
    return int(row) - int(row / 10) * 10
def dec3(row):
    return int(row / 10) - int(row / 100) * 10
def dec4(row):
    return int(row / 100) - int(row / 1000) * 10
def dec5(row):
    return int(row / 1000) - int(row / 10000) * 10


def is_trafficzero_time_notzero(online_time, month_traffic):
    if month_traffic == 0 and online_time != 0:
        return 1
    else:
        return 0


def traffic_fee(service_type, month_traffic):
    if service_type == 1:
        return month_traffic / 800
    else:
        return month_traffic * 0.3


def Rate_diff_1_fee_up(total_fee):
    if total_fee < 5:
        return (5 - total_fee) / 5
    elif total_fee >= 5 and total_fee < 19:
        return (19 - total_fee) / 19
    elif total_fee >= 19 and total_fee < 39:
        return (39 - total_fee) / 39
    elif total_fee >= 39 and total_fee < 49:
        return (49 - total_fee) / 49
    elif total_fee >= 49 and total_fee < 59:
        return (59 - total_fee) / 59
    elif total_fee >= 59 and total_fee < 99:
        return (99 - total_fee) / 99
    elif total_fee >= 99 and total_fee < 139:
        return (139 - total_fee) / 139
    elif total_fee >= 139 and total_fee < 199:
        return (199 - total_fee) / 199
    elif total_fee >= 199 and total_fee < 398:
        return (398 - total_fee) / 398
    elif total_fee >= 398:
        return (total_fee - 398) / 398


def Rate_diff_1_fee_down(total_fee):
    if total_fee < 5:
        return 0
    elif total_fee >= 5 and total_fee < 19:
        return (total_fee - 5) / 5
    elif total_fee >= 19 and total_fee < 39:
        return (total_fee - 19) / 19
    elif total_fee >= 39 and total_fee < 49:
        return (total_fee - 39) / 39
    elif total_fee >= 49 and total_fee < 59:
        return (total_fee - 49) / 49
    elif total_fee >= 59 and total_fee < 99:
        return (total_fee - 59) / 59
    elif total_fee >= 99 and total_fee < 139:
        return (total_fee - 99) / 99
    elif total_fee >= 139 and total_fee < 199:
        return (total_fee - 139) / 139
    elif total_fee >= 199 and total_fee < 398:
        return (total_fee - 199) / 199
    elif total_fee >= 398:
        return (total_fee - 398) / 398


def Rate_diff_1_traffic_up(traffic):
    if traffic < 1024:
        return (1024 - traffic) / 1024
    elif traffic >= 1024 and traffic < 1024 * 2:
        return (1024 * 2 - traffic) / (1024 * 2)
    elif traffic >= 1024 * 2 and traffic < 1024 * 3:
        return (1024 * 3 - traffic) / (1024 * 3)
    elif traffic >= (1024 * 3) and traffic < 1024 * 4:
        return (1024 * 4 - traffic) / (1024 * 4)
    elif traffic >= (1024 * 4) and traffic < (1024 * 5):
        return (1024 * 5 - traffic) / (1024 * 5)
    elif traffic >= (1024 * 5) and traffic < (1024 * 10):
        return (1024 * 10 - traffic) / (1024 * 10)
    elif traffic >= (1024 * 10) and traffic < (1024 * 20):
        return ((1024 * 20) - traffic) / (1024 * 20)
    elif traffic >= (1024 * 20) and traffic < (1024 * 30):
        return ((1024 * 30) - traffic) / (1024 * 30)
    elif traffic >= (1024 * 30) and traffic < (1024 * 40):
        return ((1024 * 40) - traffic) / (1024 * 40)
    elif traffic >= (1024 * 40) and traffic < (1024 * 100):
        return ((1024 * 100) - traffic) / (1024 * 100)
    elif traffic > (1024 * 100):
        return (traffic - 1024 * 100) / (1024 * 100)


def Rate_diff_1_traffic_down(traffic):
    if traffic < 1024:
        return 0
    elif traffic >= 1024 and traffic < 1024 * 2:
        return (traffic - 1024) / (1024)
    elif traffic >= 1024 * 2 and traffic < 1024 * 3:
        return (traffic - 1024 * 2) / (1024 * 2)
    elif traffic >= (1024 * 3) and traffic < 1024 * 4:
        return (traffic - 1024 * 3) / (1024 * 3)
    elif traffic >= (1024 * 4) and traffic < (1024 * 5):
        return (traffic - 1024 * 4) / (1024 * 4)
    elif traffic >= (1024 * 5) and traffic < (1024 * 10):
        return (traffic - 1024 * 5) / (1024 * 5)
    elif traffic >= (1024 * 10) and traffic < (1024 * 20):
        return (traffic - 1024 * 10) / (1024 * 10)
    elif traffic >= (1024 * 20) and traffic < (1024 * 30):
        return (traffic - (1024 * 20)) / (1024 * 20)
    elif traffic >= (1024 * 30) and traffic < (1024 * 40):
        return (traffic - (1024 * 30)) / (1024 * 30)
    elif traffic >= (1024 * 40) and traffic < (1024 * 100):
        return (traffic - (1024 * 40)) / (1024 * 40)
    elif traffic > (1024 * 100):
        return (traffic - 1024 * 100) / (1024 * 100)




def Rate_diff_online_up(online):
    if online < 9:
        return (9 - online) / 9
    elif online >= 9 and online < 14:
        return (14 - online) / 14
    elif online >= 14 and online < 17:
        return (17 - online) / 17
    elif online >= 17 and online < 34:
        return (34 - online) / 34
    elif online >= 34 and online < 40:
        return (40 - online) / 40
    elif online >= 40 and online < 62:
        return (62 - online) / 62
    elif online >= 69 and online < 116:
        return (116 - online) / 116
    elif online >= 116 and online < 136:
        return (136 - online) / 136
    elif online > 136:
        return (online - 136) / 136


def Rate_diff_online_down(online):
    if online < 9:
        return 0
    elif online >= 9 and online < 14:
        return (online - 9) / 9
    elif online >= 14 and online < 17:
        return (online - 14) / 14
    elif online >= 17 and online < 34:
        return (online - 17) / 17
    elif online >= 34 and online < 40:
        return (online - 34) / 34
    elif online >= 40 and online < 62:
        return (online - 40) / 40
    elif online >= 62 and online < 116:
        return (online - 62) / 62
    elif online >= 116 and online < 136:
        return (online - 116) / 116
    elif online > 136:
        return (online - 136) / 136


def other(data):
    data['gender'] = data['gender'].apply(lambda x: -1 if x == r'\N' or x == r'\\N'  else x)
    data['1_total_fee'] = data['1_total_fee'].apply(lambda x: -1 if x == r'\N' or x == r'\\N'  else x)
    data['2_total_fee'] = data['2_total_fee'].apply(lambda x: -1 if x == r'\N' or x == r'\\N'  else x)
    data['3_total_fee'] = data['3_total_fee'].apply(lambda x: -1 if x == r'\N' or x == r'\\N'  else x)
    data['4_total_fee'] = data['4_total_fee'].apply(lambda x: -1 if x == r'\N' or x == r'\\N'  else x)
    data['2_total_fee'] = data['2_total_fee'].astype(float)
    data['3_total_fee'] = data['3_total_fee'].astype(float)
    fee_mean = data[['user_id']]
    fee_mean['fee_mean'] = data.loc[:, '1_total_fee':'4_total_fee'].mean(axis=1)
    fee_mean['1_fee_diff'] = (data['1_total_fee'] - fee_mean['fee_mean']) / fee_mean['fee_mean']
    fee_mean['2_fee_diff'] = (data['2_total_fee'] - fee_mean['fee_mean']) / fee_mean['fee_mean']
    fee_mean['3_fee_diff'] = (data['3_total_fee'] - fee_mean['fee_mean']) / fee_mean['fee_mean']
    fee_mean['4_fee_diff'] = (data['4_total_fee'] - fee_mean['fee_mean']) / fee_mean['fee_mean']
    fee_u1 = fee_mean.loc[int(len(fee_mean['1_fee_diff']) * 3 / 4), '1_fee_diff']
    fee_l1 = fee_mean.loc[int(len(fee_mean['1_fee_diff']) / 4), '1_fee_diff']
    iqr1 = fee_u1 - fee_l1
    upper_bound1 = fee_u1 + 1.5 * iqr1
    lower_bound1 = fee_l1 - 1.5 * iqr1
    fee_u2 = fee_mean.loc[int(len(fee_mean['2_fee_diff']) * 3 / 4), '2_fee_diff']
    fee_l2 = fee_mean.loc[int(len(fee_mean['2_fee_diff']) / 4), '2_fee_diff']
    iqr2 = fee_u2 - fee_l2
    upper_bound2 = fee_u2 + 1.5 * iqr2
    lower_bound2 = fee_l2 - 1.5 * iqr2
    fee_u3 = fee_mean.loc[int(len(fee_mean['3_fee_diff']) * 3 / 4), '3_fee_diff']
    fee_l3 = fee_mean.loc[int(len(fee_mean['3_fee_diff']) / 4), '3_fee_diff']
    iqr3 = fee_u3 - fee_l3
    upper_bound3 = fee_u3 + 1.5 * iqr3
    lower_bound3 = fee_l3 - 1.5 * iqr3
    fee_u4 = fee_mean.loc[int(len(fee_mean['4_fee_diff']) * 3 / 4), '4_fee_diff']
    fee_l4 = fee_mean.loc[int(len(fee_mean['4_fee_diff']) / 4), '4_fee_diff']
    iqr4 = fee_u4 - fee_l4
    upper_bound4 = fee_u4 + 1.5 * iqr4
    lower_bound4 = fee_l4 - 1.5 * iqr4
    fee1_upout = fee_mean[fee_mean['1_fee_diff'] > upper_bound1].index.tolist()
    fee1_lowout = fee_mean[fee_mean['1_fee_diff'] < lower_bound1].index.tolist()
    fee2_upout = fee_mean[fee_mean['2_fee_diff'] > upper_bound2].index.tolist()
    fee2_lowout = fee_mean[fee_mean['2_fee_diff'] < lower_bound2].index.tolist()
    fee3_upout = fee_mean[fee_mean['3_fee_diff'] > upper_bound3].index.tolist()
    fee3_lowout = fee_mean[fee_mean['3_fee_diff'] < lower_bound3].index.tolist()
    fee4_upout = fee_mean[fee_mean['4_fee_diff'] > upper_bound4].index.tolist()
    fee4_lowout = fee_mean[fee_mean['4_fee_diff'] < lower_bound4].index.tolist()
    data['1_out_fee'] = 0
    data['2_out_fee'] = 0
    data['3_out_fee'] = 0
    data['4_out_fee'] = 0
    data['1_out_fee'][fee1_upout] = 1
    data['2_out_fee'][fee2_upout] = 1
    data['3_out_fee'][fee3_upout] = 1
    data['4_out_fee'][fee4_upout] = 1
    data['1_out_fee'][fee1_lowout] = -1
    data['2_out_fee'][fee2_lowout] = -1
    data['3_out_fee'][fee3_lowout] = -1
    data['4_out_fee'][fee4_lowout] = -1
    return data



def zuheFeature(data_source):
    data = data_source.copy()
    
    data['fee_min'] = data.loc[:, '1_total_fee':'4_total_fee'].min(axis=1)
    data['fee_max'] = data.loc[:, '1_total_fee':'4_total_fee'].max(axis=1)
    data['last_month_traffic_rest'] = data['month_traffic'] - data['last_month_traffic']
    data['pay_num_pertime'] = data['pay_num'] / data['pay_times']
    
    data['range'] = data['fee_max'] - data['fee_min']
    data['range_fee1'] = data['1_total_fee'] / data['range']
    data['range_fee2'] = data['2_total_fee'] / data['range']
    data['range_fee3'] = data['3_total_fee'] / data['range']
    data['range_fee4'] = data['4_total_fee'] / data['range']
    
    data['online_fee'] = data['online_time'] * data['fee_min']

    data['ratio_service2_online'] = data['service2_caller_time'] / data['online_time']

    data['ratio_month_online'] = data['month_traffic'] / data['online_time']

    
    
    tmp = []
    for i in tqdm(range(len(data))):
        if (data['month_traffic'][i] - (data['last_month_traffic'][i] + data['local_trafffic_month'][i])) < 0:
            tmp.append(data['1_total_fee'][i] - (data['service1_caller_time'][i] * 0.15))
        elif (data['month_traffic'][i] - (data['last_month_traffic'][i] + data['local_trafffic_month'][i])) <= 200:
            tmp.append(data['1_total_fee'][i] - (data['service1_caller_time'][i] * 0.15) - (
            data['month_traffic'][i] - (data['last_month_traffic'][i] + data['local_trafffic_month'][i])) * 0.3)
        else:
            tmp.append(data['1_total_fee'][i] - (data['service1_caller_time'][i] * 0.15) - (200 * 0.3 + (
            data['month_traffic'][i] - (
            data['last_month_traffic'][i] + data['local_trafffic_month'][i]) - 200) * 60 / 1024))
    data['base_fee'] = tmp
    
    
    isint_feature=['1_total_fee','2_total_fee','3_total_fee','4_total_fee','month_traffic','pay_num','last_month_traffic','local_trafffic_month','local_caller_time',
              'service1_caller_time','service2_caller_time']
    for col in isint_feature:
        data[col+'xx']=data[col].apply(get_str4)
        data[col+'x1']=data[col+'xx'].map(lambda x:x[0])
        data[col+'x1']=data[col+'x1'].astype(int)
        data[col+'x2']=data[col+'xx'].map(lambda x:x[1])
        data[col+'x2']=data[col+'x2'].astype(int)
        data[col+'x3']=data[col+'xx'].map(lambda x:x[2])
        data[col+'x3']=data[col+'x3'].replace('e',0)
        data[col+'x3']=data[col+'x3'].astype(int)
        data[col+'x4']=data[col+'xx'].map(lambda x:x[3])
        data[col+'x4']=data[col+'x4'].replace('-',0)
        data[col+'x4']=data[col+'x4'].astype(int)
        data.pop(col+'xx')
    
    
    
    
    # add 20:53pm 2018-11-10
    # *******************************************************
    data['1decimal'] = data['1_total_fee'].apply(dec)
    data['1decimal1'] = data['1_total_fee'].apply(dec1)
    data['1decimal2'] = data['1_total_fee'].apply(dec2)
    data['1decimal3'] = data['1_total_fee'].apply(dec3)
    data['1decimal4'] = data['1_total_fee'].apply(dec4)
    
    data['mindecimal2'] = data['fee_min'].apply(dec2)
    data['mindecimal4'] = data['fee_min'].apply(dec4)

    data['lmtdecimal5'] = data['last_month_traffic'].apply(dec5)
    data['lmtdecimal4'] = data['last_month_traffic'].apply(dec4)
    data['lmtdecimal2'] = data['last_month_traffic'].apply(dec2)

    data['1calldecimal3'] = data['service1_caller_time'].apply(dec3)
    data['2calldecimal4'] = data['service2_caller_time'].apply(dec4)
    
    
    for i in tqdm(list(range(1,5))):
        data[str(i)+'_total_fee_max'] = data[str(i)+'_total_fee']/data[str(i)+'_total_fee'].max()
        data[str(i)+'_total_fee_min'] = data[str(i)+'_total_fee']/data[str(i)+'_total_fee'].min()
        data[str(i)+'_total_fee_median'] = data[str(i)+'_total_fee']/data[str(i)+'_total_fee'].median()
        data[str(i)+'_total_fee_mean'] = data[str(i)+'_total_fee']/data[str(i)+'_total_fee'].mean()
    
    
    LT_count=['1_total_fee','2_total_fee','3_total_fee','4_total_fee','online_time','age','last_month_traffic_rest','pay_num']
    for col in LT_count:
        lt=data.groupby(col).size().reset_index()
        lt.columns=[col,col+'_count']
        data=pd.merge(data,lt,on=col,how='left')
        
        
    # data['fee_mean'] = data.loc[:, '1_total_fee':'4_total_fee'].mean(axis=1)

    # data['mean_fee_1'] = (data['1_total_fee'] - data['fee_mean']).abs()

    # data['mean_fee_2'] = (data['2_total_fee'] - data['fee_mean']).abs()

    # data['mean_fee_3'] = (data['3_total_fee'] - data['fee_mean']).abs()

    # data['mean_fee_4'] = (data['4_total_fee'] - data['fee_mean']).abs()
    data['rate_diff_1_fee_up'] = data['1_total_fee'].apply(Rate_diff_1_fee_up)

    data['rate_diff_1_fee_down'] = data['1_total_fee'].apply(Rate_diff_1_fee_down)

    data['rate_diff_2_fee_up'] = data['2_total_fee'].apply(Rate_diff_1_fee_up)

    data['rate_diff_2_fee_down'] = data['2_total_fee'].apply(Rate_diff_1_fee_down)

    data['rate_diff_3_fee_up'] = data['3_total_fee'].apply(Rate_diff_1_fee_up)

    data['rate_diff_3_fee_down'] = data['3_total_fee'].apply(Rate_diff_1_fee_down)

    data['rate_diff_4_fee_up'] = data['4_total_fee'].apply(Rate_diff_1_fee_up)

    data['rate_diff_4_fee_down'] = data['4_total_fee'].apply(Rate_diff_1_fee_down)
    
    data["Rate_diff_online_up"] = data["online_time"].map(Rate_diff_online_up)
    
    data["Rate_diff_online_down"] = data["online_time"].map(Rate_diff_online_down)
    
    #data['rate_diff_local_traffic_up']=data['local_trafffic_month'].apply(Rate_diff_1_traffic_up)
    
    #data['rate_diff_local_traffic_down']=data['local_trafffic_month'].apply(Rate_diff_1_traffic_down)
    
    #data['rate_diff_month_traffic_up']=data['month_traffic'].apply(Rate_diff_1_traffic_up)
    
    #data['rate_diff_month_traffic_down']=data['month_traffic'].apply(Rate_diff_1_traffic_down)
    
    #data['rate_diff_lastmonth_traffic_up']=data['last_month_traffic'].apply(Rate_diff_1_traffic_up)
    
    #data['rate_diff_lastmonth_traffic_down']=data['last_month_traffic'].apply(Rate_diff_1_traffic_down)
    # --------------------------add on 11_4/21:48
    # data['pay_1'] = (data['pay_num']-data['1_total_fee']).abs()
    # data['pay_2'] = (data['pay_num']-data['2_total_fee']).abs()
    # data['pay_3'] = (data['pay_num']-data['3_total_fee']).abs()
    # data['pay_4'] = (data['pay_num']-data['4_total_fee']).abs()
    # data['1_sum'] = data['online_time']+data['contract_time']
    # data['2_sum'] = data['last_month_traffic']+data['month_traffic']

    
    data['month_traffic_fee'] = data.apply(lambda row: traffic_fee(row['service_type'], row['month_traffic']), axis=1)
   
    # *************************************************************

    data['service1_caller_time_fee'] = data['service1_caller_time'] * 0.15
    data['service2_caller_time_fee'] = data['service2_caller_time'] * 0.15
    data['21_total'] = (data['2_total_fee'] - data['1_total_fee']).abs()
    
    data['32_total'] = (data['3_total_fee'] - data['2_total_fee']).abs()
    
    data['43_total'] = (data['4_total_fee'] - data['3_total_fee']).abs()
    

    data['out1_fee1'] = data['1_total_fee'] - data['service1_caller_time_fee']
    data['out1_fee2'] = data['2_total_fee'] - data['service1_caller_time_fee']
    data['out1_fee3'] = data['3_total_fee'] - data['service1_caller_time_fee']
    data['out1_fee4'] = data['4_total_fee'] - data['service1_caller_time_fee']
  
    data['out1_fee1_rate'] = data['out1_fee1'] / data['1_total_fee']
    data['out1_fee2_rate'] = data['out1_fee2'] / data['2_total_fee']
    data['out1_fee3_rate'] = data['out1_fee3'] / data['3_total_fee']
    data['out1_fee3_rate'] = data['out1_fee4'] / data['4_total_fee']

    data['out2_fee1'] = data['1_total_fee'] - data['service2_caller_time_fee']
    data['out2_fee2'] = data['2_total_fee'] - data['service2_caller_time_fee']
    data['out2_fee3'] = data['3_total_fee'] - data['service2_caller_time_fee']
    data['out2_fee4'] = data['4_total_fee'] - data['service2_caller_time_fee']
   
    data['out2_fee1_rate'] = (data['1_total_fee'] - data['service2_caller_time_fee']) / data['1_total_fee']
    data['out2_fee2_rate'] = (data['2_total_fee'] - data['service2_caller_time_fee']) / data['2_total_fee']
    data['out2_fee3_rate'] = (data['3_total_fee'] - data['service2_caller_time_fee']) / data['3_total_fee']
    data['out2_fee3_rate'] = (data['4_total_fee'] - data['service2_caller_time_fee']) / data['4_total_fee']

    

    

    # -----------------------------------------------------add new features-------------------------
    data['call_time_sub1'] = data['local_caller_time'] - data['service1_caller_time']

    data['fee_1'] = data['1_total_fee'] / data['month_traffic']

    data['fee_2'] = data['1_total_fee'] / data['last_month_traffic_rest']

    # data['fee_3'] = data['1_total_fee'] / data['local_trafffic_month']

    # data['fee_4'] = data['1_total_fee'] / data['local_caller_time']

    # data['fee_5'] = data['1_total_fee'] / data['service1_caller_time']

    data['fee_6'] = data['1_total_fee'] / data['call_time_sub1']

    data['1_fee'] = data['1_total_fee'] * data['month_traffic']

    data['2_fee'] = data['1_total_fee'] * data['last_month_traffic_rest']

    data['3_fee'] = data['1_total_fee'] * data['local_trafffic_month']

    # data['4_fee'] = data['1_total_fee']*data['local_caller_time']

    # data['5_fee'] = data['1_total_fee']*data['service1_caller_time']

    # data['6_fee'] = data['1_total_fee']*data['call_time_sub1']
    
    return data


def getOneHot(data):
    cat_feat = ['service_type', 'net_service', 'gender', 'complaint_level', 'contract_type']
    for cat in cat_feat:
        data[cat] = pd.Categorical(data[cat])
    return data


def get_cv(data):
    num_feature = data.columns
    data['traffic_month_min'] = data[['local_trafffic_month', 'month_traffic']].min(axis=1)
    data['traffic_month_max'] = data[['local_trafffic_month', 'month_traffic']].max(axis=1)
    data['new_con'] = data['1_total_fee'].astype(str)
    for i in ['2_total_fee', '3_total_fee', '4_total_fee']:
        data['new_con'] = data['new_con'].astype(str) + '_' + data[i].astype(str)
    data['new_con'] = data['new_con'].apply(lambda x: ' '.join(x.split('_')))

    # print(len(data))
    total_feature = sparse.csr_matrix((len(data), 0))
    cv = CountVectorizer(min_df=2)
    # print(data['new_con'])
    for feature in ['new_con']:
        data[feature] = data[feature].astype(str)
        # print(data[feature])
        cv.fit(data[feature])
        total_feature = sparse.hstack((total_feature, cv.transform(data[feature].astype(str))), 'csr', 'bool')
    print('CountVectorizer_over!')
    total_feature = sparse.hstack((sparse.csr_matrix(data[num_feature].astype('float32')), total_feature),
                                  'csr').astype('float32')
    #     print(total_feature)
    return total_feature


def match(x=None):
    if x["local_caller_time"] < 60:
        s1_local = 0
        s1_non_local = x["service1_caller_time"]
        return s1_local, s1_non_local
    elif x["local_caller_time"] > 60 and x["local_caller_time"] <= 100:
        if (x["local_caller_time"] - 60) <= x["service1_caller_time"]:
            s1_local = x["local_caller_time"] - 60
            s1_non_local = x["service1_caller_time"] - s1_local
        else:
            s1_local = x["local_caller_time"] - 60
            s1_non_local = x["service1_caller_time"]
        return s1_local, s1_non_local

    elif x["local_caller_time"] > 100 and x["local_caller_time"] <= 200:
        if (x["local_caller_time"] - 100) <= x["service1_caller_time"]:
            s1_local = x["local_caller_time"] - 100
            s1_non_local = x["service1_caller_time"] - s1_local
        else:
            s1_local = x["local_caller_time"] - 100
            s1_non_local = x["service1_caller_time"]
        return s1_local, s1_non_local

    elif x["local_caller_time"] > 200 and x["local_caller_time"] <= 300:
        if (x["local_caller_time"] - 200) <= x["service1_caller_time"]:
            s1_local = x["local_caller_time"] - 200
            s1_non_local = x["service1_caller_time"] - s1_local
        else:
            s1_local = x["local_caller_time"] - 200
            s1_non_local = x["service1_caller_time"]
        return s1_local, s1_non_local

    elif x["local_caller_time"] > 300 and x["local_caller_time"] <= 500:
        if (x["local_caller_time"] - 300) <= x["service1_caller_time"]:
            s1_local = x["local_caller_time"] - 300
            s1_non_local = x["service1_caller_time"] - s1_local
        else:
            s1_local = x["local_caller_time"] - 300
            s1_non_local = x["service1_caller_time"]
        return s1_local, s1_non_local

    elif x["local_caller_time"] > 500 and x["local_caller_time"] <= 1000:
        if (x["local_caller_time"] - 500) <= x["service1_caller_time"]:
            s1_local = x["local_caller_time"] - 500
            s1_non_local = x["service1_caller_time"] - s1_local
        else:
            s1_local = x["local_caller_time"] - 500
            s1_non_local = x["service1_caller_time"]
        return s1_local, s1_non_local

    elif x["local_caller_time"] > 1000 and x["local_caller_time"] <= 2000:
        if (x["local_caller_time"] - 1000) <= x["service1_caller_time"]:
            s1_local = x["local_caller_time"] - 1000
            s1_non_local = x["service1_caller_time"] - s1_local
        else:
            s1_local = x["local_caller_time"] - 1000
            s1_non_local = x["service1_caller_time"]
        return s1_local, s1_non_local

    elif x["local_caller_time"] > 2000 and x["local_caller_time"] <= 3000:
        if (x["local_caller_time"] - 2000) <= x["service1_caller_time"]:
            s1_local = x["local_caller_time"] - 2000
            s1_non_local = x["service1_caller_time"] - s1_local
        else:
            s1_local = x["local_caller_time"] - 2000
            s1_non_local = x["service1_caller_time"]
        return s1_local, s1_non_local

    else:
        if (x["local_caller_time"] - 3000) <= x["service1_caller_time"]:
            s1_local = x["local_caller_time"] - 3000
            s1_non_local = x["service1_caller_time"] - s1_local
        else:
            s1_local = x["local_caller_time"] - 3000
            s1_non_local = x["service1_caller_time"]
        return s1_local, s1_non_local


def get_Local_S1(df=None):
    data = copy.deepcopy(df)
    data = data[["user_id", "local_caller_time", "service1_caller_time"]]
    print(data.shape)

    data["s1_local"] = 0
    data["s1_non_local"] = 0

    zero_local = data[data["local_caller_time"] == 0]
    nozero_local = data[data["local_caller_time"] != 0]

    zero_local["s1_local"] = 0
    zero_local["s1_non_local"] = zero_local["service1_caller_time"].values

    temp = nozero_local.apply(match, axis=1)
    nozero_local["s1_local"] = temp.map(lambda x: x[0]).values
    nozero_local["s1_non_local"] = temp.map(lambda x: x[1]).values

    data = pd.concat([zero_local, nozero_local], axis=0, ignore_index=True)
    print(data.shape)
    return data


def generateFeature(data):
    # dropList=['gender','complaint_level','pay_times','contract_time','service_type','contract_type','net_service','former_complaint_num']
    dropList = ['former_complaint_fee', 'complaint_level']

    data = other(data)

    data = zuheFeature(data)

    data = getOneHot(data)

    data.drop(dropList, axis=1, inplace=True)

    return data


def readData():
    train = pd.read_csv("train_2.csv", sep=",", encoding="utf-8")
    train = train[train["service_type"] != 3].reset_index(drop=True)
    print(train.shape)
    test = pd.read_csv("test_2.csv", sep=",", encoding="utf-8")
    test["current_service"] = -1
    print(test.shape)
    data = pd.concat([train, test[train.columns]], axis=0, ignore_index=True)
    data = data[data["current_service"] != 999999].reset_index(drop=True)
    return data


def baseClean(data):
    print("preprocessing!")
    columns = ["2_total_fee", "3_total_fee"]
    for i in columns:
        data[i] = data[i].map(lambda x: np.nan if str(x) == '\\N' else float(x))
    return data


# *********************************************
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(11, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'f1_score', score_vali ** 2, True

train_data1 = pd.read_csv('train_2.csv')
train_data2 = pd.read_csv('train_all.csv')
'''
#test
train_data1 = train_data1.sample(n=100).reset_index(drop=True)
train_data2 = train_data2.sample(n=100).reset_index(drop=True)
'''
print(train_data1.shape, train_data2.shape)

# train_data = pd.concat([train_data1,train_data2],axis=0,ignore_index=True)
train_data1 = train_data1[train_data1['current_service'] != 999999].reset_index(drop=True)
train_data1 = train_data1[train_data1['service_type'] != 3].reset_index(drop=True)

test_data = pd.read_csv('test_2.csv')
test_data["current_service"] = -1
data1 = pd.concat([train_data1, test_data], axis=0, ignore_index=True)
# data2 = pd.concat([train_data2,test_data],axis=0,ignore_index=True)
data1 = generateFeature(data1)
data2 = generateFeature(train_data2)
data = pd.concat([data1, data2], axis=0, ignore_index=True)
# *********************************************


train_feature = data[data["current_service"] != -1].reset_index(drop=True)
test_feature = data[data["current_service"] == -1].reset_index(drop=True)
print(train_feature.shape)
print(test_feature.shape)

train = train_feature
test = test_feature

print('train data shape', train.shape)
print('train data of user_id shape', len(set(train['user_id'])))
print('train data of current_service shape', (set(train['current_service'])))

print('train data shape', test.shape)
print('train data of user_id shape', len(set(test['user_id'])))

label2current_service = dict(
    zip(range(0, len(set(train['current_service']))), sorted(list(set(train['current_service'])))))
current_service2label = dict(
    zip(sorted(list(set(train['current_service']))), range(0, len(set(train['current_service'])))))

y = train_feature.pop('current_service').map(current_service2label)

# y = train.pop('current_service')
train_id = train.pop('user_id')

X = train
train_col = train.columns

X_test = test[train_col]
test_id = test['user_id']

'''
for i in train_col:
    X[i] = X[i].replace("\\N",-1)
    X_test[i] = X_test[i].replace("\\N",-1)
'''
for i in train_col:
    X[i] = X[i].map(lambda x: -1 if str(x) == '\\N' else x)
    X_test[i] = X_test[i].map(lambda x: -1 if str(x) == '\\N' else x)
# test_id = test_data['user_id']
# y = train_feature.pop('current_service')

t = pd.concat([train_feature, test_feature], axis=0)
print(t.dtypes)
print(t.shape)
user_id = t.pop('user_id')
t = get_cv(t)
# t = pd.concat([user_id, t], axis=1)
train = t[:train.shape[0]]
test = t[train.shape[0]:]

X, y, X_test = train, y, test

print("please check the dimesion of each dataset")
print("******************************************")
print(X.shape, y.shape, X_test.shape)
print("******************************************")

import pandas as pd
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

n_splits = 10
seed = 42

params = {
    "learning_rate": 0.1,
    "boosting": 'gbdt',
   
    "lambda_l2": 0.1,
    "max_depth": -1,
    "num_leaves": 128,
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "max_bin": 1500,
    "metric": None,
    "objective": "multiclass",
    "num_class": 11,
    "silent": True,
    "nthread": 10,
}

xx_score = []
cv_pred = []

temp = pd.DataFrame()

skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
for index, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(index)

    X_train, X_valid, y_train, y_valid = X[train_index], X[test_index], y[train_index], y[test_index]

    train_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params, train_data, num_boost_round=1900, valid_sets=[validation_data], feval=f1_score_vali,
                    verbose_eval=1, early_stopping_rounds=50)

    xx_pred = clf.predict(X_valid, num_iteration=clf.best_iteration)

    xx_pred = [np.argmax(x) for x in xx_pred]

    xx_score.append(f1_score(y_valid, xx_pred, average='macro') ** 2)

    y_test = clf.predict(X_test, num_iteration=clf.best_iteration)
    print(y_test)

    if index == 0:
        temp = y_test
    else:
        temp += y_test

    y_test = [np.argmax(x) for x in y_test]
    if index == 0:
        cv_pred = np.array(y_test).reshape(-1, 1)
    else:
        cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))

submit = []
for line in cv_pred:
    submit.append(np.argmax(np.bincount(line)))

submit1 = [np.argmax(x) for x in temp]

print("*******************************")
print("temp")
print(temp.shape)
print(temp)

print("cv_pred")
print(cv_pred.shape)
print(cv_pred)
print("********************************")

pd.DataFrame(temp).to_csv("sub2_prob.csv", index=False)

df_test = pd.DataFrame()
df_test['id'] = list(test_id.unique())
df_test['predict1'] = submit
df_test['predict1'] = df_test['predict1'].map(label2current_service).astype(int)
df_test['predict2'] = submit1
df_test['predict2'] = df_test['predict2'].map(label2current_service).astype(int)

print(xx_score, np.mean(xx_score))

df_test[['id', 'predict2']].to_csv('sub2_gailv.csv', index=False)
print(xx_score, np.mean(xx_score))
