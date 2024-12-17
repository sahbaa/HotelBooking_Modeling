import pandas as pd 
from ydata_profiling import ProfileReport
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_regression,f_classif,mutual_info_classif,mutual_info_regression
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV




df = pd.read_csv('preprocessingOnkaggle\\booking_hotel\hotel_bookings.csv')
report = ProfileReport(df)
# report.to_file('./EDA.html')
df.set_index('id')
target = df.iloc[:,2]
target.astype('str')
print(target.index)
data = df.drop(['is_canceled','reservation_status','reservation_status_date'],axis=1)
x_train , x_test , y_train , y_test = train_test_split (data , target , test_size= 0.3)
cat_col = ['typeOfHotel','arrival_date_year','arrival_date_month','previous_bookings_not_canceled'
            ,'stays_in_weekend_nights','stays_in_week_nights' ,'adults','children','babies'
            ,'meal','country','market_segment','distribution_channel','is_repeated_guest'
            ,'previous_cancellations', 'reserved_room_type'
            ,'assigned_room_type', 'booking_changes', 'deposit_type', 'agent'
            , 'company', 'days_in_waiting_list', 'customer_type'
            , 'required_car_parking_spaces', 'total_of_special_requests']
            
cont_col = ['lead_time','arrival_date_week_number','arrival_date_day_of_month', 'adr']

def edaCleaning(data):
    in_data = data.copy()
    in_data['arrival_date_year'] = in_data['arrival_date_year'].replace({'2016': 0,'2017': 1,'2018': 2}).astype('str')
    in_data['arrival_date_month'] = in_data['arrival_date_month'].replace({'January': '0',
                                                                    'February': '1','March': '2'
                                                                    ,'April' : '3','May':'4'
                                                                    , 'June' :'5' ,'July' : '6'
                                                                    , 'August': '7','September': '8'
                                                                    , 'October': '9' , 'November':'10'
                                                                    , 'December' : '11'
                                                                    })
    in_data['stays_in_weekend_nights'] = in_data['stays_in_weekend_nights'].apply(lambda x:'0' if x== '0' else '1')
    in_data['stays_in_week_nights'] = in_data['stays_in_week_nights'].apply(lambda x : '0' if x == '0' else '1'if x == '1'
                                                                    else '2' if x == '2' else '3' if x == '3' else x == '4'
                                                                    if x == '4' else '5' if x == '5' else '6')
    in_data['adults'] = in_data['adults'].apply(lambda x : '0' if x == '1' else '1' if x == '2' else '2')
    in_data['children'] = in_data['children'].apply(lambda x : '0' if x == '0' else '1')
    in_data['babies'] = in_data['babies'].apply(lambda x : '0' if x == '0' else '1')
    in_data['meal'] = in_data['meal'].replace({'Undefined' : np.nan})
    in_data['country'] = in_data['country'].apply(lambda x : 'PRT' if x == 'PRT' else 'GBR' if x == 'GBR' else 'FRA' if x == 'FRA' 
                                            else 'ESP' if x == 'ESP' else 'DEU' if x == 'DEU' else 'other')                                    
    in_data['market_segment'] = in_data['market_segment'].replace({'Undefined' : np.nan,'Complementary':'other','Aviation':'other'})
    in_data['distribution_channel'] = in_data['distribution_channel'].replace({'Undefined' : np.nan})
    in_data['previous_bookings_not_canceled'] = in_data['previous_bookings_not_canceled'].apply(lambda x: '0' if x == '0' else '1')
    in_data['days_in_waiting_list'] = in_data['days_in_waiting_list'].apply(lambda x: '0' if x == '0' else '1')
    in_data['reserved_room_type'] = in_data['reserved_room_type'].apply(lambda x : 'A' if x == 'A' else 'NotA')
    in_data['assigned_room_type'] = in_data['assigned_room_type'].apply(lambda x : 'A' if x== 'A' else 'NotA')
    in_data['booking_changes'] = in_data['booking_changes'].apply(lambda x : '0' if x == '0' else '1')
    in_data['agent'] = in_data['agent'].apply(lambda x : '9' if x == '9.0' else '240' if x == '240.0' else '1' 
                                        if x == '1.0' else 'others' )
    in_data['required_car_parking_spaces'] = in_data['required_car_parking_spaces'].apply(lambda x : '0' if x == 0 else '1')
    # #40.0 :40  #223.0 :223  [# 67.0 # 45.0 , #153.0] :seconds
    in_data['company'] = in_data['company'].apply(lambda x : '40' if x =='40.0' else '223' if x == '223.0' else 'others'                                        if x == '67.0' or x =='153.0' or x == '45.0' else 'others')
    in_data['total_of_special_requests'] = in_data['total_of_special_requests'].apply(lambda x : '0' if x == '0' 
                                                                                else '1' if x == '1' else '2' if x == '2'
                                                                                else '>=3')
    in_data['deposit_type'] = in_data['deposit_type'].replce({'Non Refund':'Deposit','Refundable':'Deposit'})
    # data['reservation_status_date'] = data['reservation_status_date'].apply(lambda x : [x.split('-')[1],x.split('-')[0]])
    


    return in_data



def frequntyTbl(data): 

    cats , counts = np.unique (data , return_counts = True)
    result = pd.DataFrame({'categories' : cats , 'counts' : counts})
    return result


def featuring(data ,cv=0.1 ,diverCat=0.9 ,lowVar = 0.94):
    in_data = data.copy()
    maxVal = in_data[cat_col].apply(lambda x : x.value_counts().max()/len(data))
    resultCat_1 = maxVal[maxVal>lowVar].index

    maxDiver = in_data[cat_col].apply(lambda x : x.nunique()/len(data))
    resultCat_2 = maxDiver[maxDiver>diverCat].index

    cv = in_data[cont_col].apply(lambda x: x.mean()/x.std())
    low_cv_col = cv[cv < 0.1].index

    deleted_cols = set(list(resultCat_1)+list(resultCat_2)+list(low_cv_col))
    in_data = in_data.drop(deleted_cols,axis =1)
    return (resultCat_1,resultCat_2,low_cv_col)



def checkLogicalVal(data): 

    in_data = data.copy()

    logical_val = {'TypeOfHotel':['0','1'], 'arrival_date_year':['0','1','2'],
       'arrival_date_month':['0','1','2','3','4','5','6','7','8','9','10','11'],
       'arrival_date_day_of_month':(1,30), 'stays_in_weekend_nights':['0','1'],
       'stays_in_week_nights':['0','1','2','3','4','5','6'], 'adults':['0','1','2'], 'children':['0','1'],
       'meal':['BB','HB','SC','FB','nan'],'country':['PRT','GBR','DEU','ESP','FRA','other'],
       'market_segment':['Online TA','Offline TA/TO','Groups','Direct','Corporate','nan','other'],
       'distribution_channel':['TA/TO','Direct','Corporate','GDS','nan'],
       'reserved_room_type':['A','NotA'],
       'assigned_room_type':['A','NotA'], 'booking_changes':['0','1'], 'deposit_type':['Deposit','No Deposit'], 
       'agent':['9','240','1','others'],
       'company':['40','223','others'], 'customer_type':['Transient','Transient-Party','Contract','Group'],
       'required_car_parking_spaces':['0','1'], 'total_of_special_requests':['0','1','0','1']}
    
    logical_val_cont = {'lead_time':(0,737),'adr':(0,5400),'arrival_date_week_number':(1,53)}

    for feature,(minval,maxval) in logical_val_cont.items():
        in_data[feature] = in_data[feature].apply(lambda x : x if x<maxval or x>minval else None)

    for feature,items in zip(in_data.select_dtypes(include=['category','object']).columns,[val for val in logical_val.values() if feature in in_data.select_dtypes(include=['category','object']).columns]):
        in_data[feature] = in_data[feature].apply(lambda x : x if x in items else None)
   

    return in_data



def outlier_detection(data,target , catcol,contcol):
    data = data.set_index('id')
    in_data = data.copy()
    #onehotencoding and scaling 
    # all cats are nominal 
    for cols in in_data.columns : 
        if cols in catcol:
            in_data[cols] = in_data[cols].fillna(data[cols].mode().iloc[0])
        else:
            in_data[cols] = in_data[cols].fillna(np.mean(data[cols]))
    
    ohe = OneHotEncoder (sparse_output=False , handle_unknown='ignore')
    encoded = ohe.fit_transform(in_data[catcol])
    encoded_data = pd.DataFrame(encoded,columns=ohe.get_feature_names_out())
    
    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(in_data[contcol]),columns=contcol)
    
    temp_data_for_out_detecting = pd.concat([encoded_data,scaled_data],axis=1)
    
    iso = IsolationForest()
    
    iso.fit(temp_data_for_out_detecting)
    temp_data_for_out_detecting['outs'] = iso.predict(temp_data_for_out_detecting)
    print(temp_data_for_out_detecting.index)
    to_remove = temp_data_for_out_detecting[temp_data_for_out_detecting['outs']==-1].index
    print(to_remove)

    data = data.drop(to_remove,axis =0 ,errors='ignore')
    target = target.drop(to_remove,axis=0,errors='ignore')

    return data,target,to_remove



def missingByRec(data,target):

    in_data = data.copy()
    sum_of_nan = in_data.isna().sum(axis=1)
    high_miss_recs = sum_of_nan[sum_of_nan >11].index
    
    in_data = in_data.drop(high_miss_recs)
    target = target.drop(high_miss_recs)
    
    return in_data,target

def missingByCol(data,test,catcol):
    #catcol=nominal 
    #iterative does not need one hot encoding just scaling
    in_data = data.copy()
    in_test = test.copy()
    
    nas_in_cols = in_data.isna().sum()
    nan_per_col = nas_in_cols/len(data)
    upperthan50 =  nan_per_col[nan_per_col>0.5].index
    fast_imp_indx = nan_per_col[(nan_per_col <0.05)].index
    iter_imp_indx = nan_per_col[(nan_per_col<0.5) & (nan_per_col >0.05)].index
    in_data = in_data.drop(upperthan50,axis = 1)
    in_test = in_test.drop(upperthan50 ,axis = 1)
    
    for cols in fast_imp_indx :

        if cols in catcol:
            imputer = SimpleImputer(strategy='most_frequent')

        elif np.abs(in_data[cols].skew())>1:            
            imputer = SimpleImputer(strategy='median')

        else :
            imputer = SimpleImputer(strategy='mean')
    
#numpy orizontal and series vertical for assigning we should transpose one of them and imputers outputs are numpy() 
        
        imputer.fit(in_data[[cols]])
        in_data[cols] = (imputer.transform(in_data[[cols]])).ravel()
    contcol = [cols for cols in in_data.columns if cols not in catcol]
    scaler = MinMaxScaler()

    for cols in iter_imp_indx :
        if cols in catcol:     
            imputer = IterativeImputer(initial_strategy='most_frequent')
        else :
            in_data[cols] = scaler.fit_transform(in_data[[cols]]) 
            imputer = IterativeImputer(initial_strategy='median')

        in_data[cols] = imputer.fit_transform(in_data[[cols]])
    
    return in_data,test





def setting_up_dtypes(data):
    list_of_modified = ['adults','arrival_date_year','stays_in_week_nights','stays_in_weekend_nights','children','booking_changes','agent','required_car_parking_spaces','total_of_special_requests']
    data[list_of_modified] = data[list_of_modified].astype('str')
    return data




def discritizer (data,test , contcol):
    in_data = data.copy()
    print(in_data[contcol].isna().sum())
    # x_train[contcol] = x_train[contcol].apply(pd.to_numeric)
    # scaler = MinMaxScaler()
    # tmp_sclr = scaler.fit_transform(in_data[contcol])
    # in_data[contcol] = pd.DataFrame(tmp_sclr,columns=contcol)
    # print(in_data[contcol].isna().sum())

    disc = KBinsDiscretizer(n_bins=5,strategy='kmeans',encode='ordinal')
    temp_disc = disc.fit_transform(in_data[contcol])
    in_data[contcol] = pd.DataFrame(temp_disc,columns=disc.get_feature_names_out())
    in_data[contcol] = in_data[contcol].astype('str')
    test[contcol] = test[contcol].astype('str')
    # for col in contcol : 
    #     print(frequntyTbl(data[col]))

    return in_data,test




def feature_selection(data,target):
    
    cat_cat = SelectKBest(score_func = chi2 , k = 'all')
    ig_cat = SelectKBest(score_func = mutual_info_classif)

    oe = OrdinalEncoder()
    encoder = oe.fit_transform(data[data.columns])
    tempencoder = pd.DataFrame(encoder,columns = oe.get_feature_names_out())

    cat_cat.fit(tempencoder[data.columns],target)
    ig_cat.fit(tempencoder[data.columns],target)

    res1 = pd.DataFrame({'name':cat_cat.get_feature_names_out(),'scores':cat_cat.scores_})
    # res2 = pd.DataFrame({'name':ig_cat.get_feature_names_out(),'scores':ig_cat.scores_})
    print(res1.shape)
    return list(res1.sort_values(by='scores',ascending=False).iloc[:10,0])




def encoding(data):
    catcol = data.select_dtypes(include=['object']).columns.to_list()
    oe = OrdinalEncoder()
    data[catcol] = pd.DataFrame(oe.fit_transform(data[catcol]),columns=oe.get_feature_names_out())
    return data[catcol]



catcol1,catcol2,contcol = featuring(x_train)
dropped_cols = set(list(catcol1)+list(catcol2)+list(contcol))
x_train = x_train.drop(dropped_cols,axis =1)
x_test = x_test.drop(dropped_cols,axis =1)

x_train = checkLogicalVal(x_train)
x_test = checkLogicalVal(x_test)

x_train,y_train,indxes  = outlier_detection(x_train,y_train,[cat for cat in cat_col if cat not in dropped_cols],[cont for cont in cont_col if cont not in dropped_cols])


x_train , y_train = missingByRec(x_train,y_train)
x_test , y_test = missingByRec(x_test,y_test)


last_cols = x_train.columns
x_train,x_test = missingByCol(x_train,x_test,[cat for cat in cat_col if cat in x_train.columns])

x_train = setting_up_dtypes (x_train)
x_test = setting_up_dtypes (x_test)

should_add = set(last_cols) ^ set(x_train.columns)
dropped_cols.update(list(should_add))

x_train,x_test = discritizer(x_train,x_test , [cont for cont in cont_col if cont  not in dropped_cols])


cols_after_selection = feature_selection(x_train,y_train)   
print(x_test.columns)
x_train = x_train.drop([cols for cols in x_train.columns if cols not in cols_after_selection],axis = 1)
x_test = x_test.drop([cols for cols in x_test.columns if cols not in cols_after_selection], axis = 1)

x_train = encoding(x_train)
x_test = encoding(x_test)
x_train.to_csv('train_preprocessed.csv')
x_test.to_csv('test_preprocessed.csv')

params = {'criterion': ['entropy','log_loss','gini'],'max_depth':[2,3,4,5] ,'min_impurity_decrease':[0.02,0.05,0.1,0.12] , 'ccp_alpha':[0.05,0.1,0.15,0.2] , 'min_samples_split':[10,20,30,40,50] }
tree_mdl = DecisionTreeClassifier()
# tree_mdl.fit(x_train,y_train)
# y_pred = tree_mdl.predict(x_test)
# cnf = confusion_matrix(y_test,y_pred)
# acc = accuracy_score(y_test,y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cnf,display_labels=tree_mdl.classes_)
# disp.plot()


bestmodel_finder = GridSearchCV(tree_mdl,param_grid=params,scoring='accuracy',cv=5)
bestmodel_finder.fit(x_train,y_train)
pred = bestmodel_finder.predict(x_test)
cnf2 = confusion_matrix(y_test,pred).T
acc2 = accuracy_score(y_test,pred)
print(acc2)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cnf2,display_labels=bestmodel_finder.best_estimator_.classes_)
print(bestmodel_finder.best_estimator_)
disp2.plot()
plt.show()