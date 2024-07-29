# %%
from netCDF4 import Dataset
from wrf import getvar, interplevel, ALL_TIMES, extract_times
import numpy as np
import xarray as xr
import glob 
import datetime
import pandas as pd
from k_index import calc_LI_TT_SWI_Q
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KDTree
from time import time 
import joblib
import pickle 
# %%
def get_wrf(name):
    """""
    In: year, month, day, time
    Out: WRF dataset for specified time
    """""
    start = time()
    path_wrf = "/Users/jmceachern/wrf_data/"
    end = time()
    print('get_wrf: ',end-start)
    return Dataset(path_wrf+name)

def get_glm(yy,mm, dd, tt):
    """""
    In: year, month, day, time
    Out: 1 hour of GLM data and returns list of netCDF4 files.
    There is one files for every 20s 
    """""

    # j_day = days since Jan 1st of yy
    start = time()
    j_day = (datetime.datetime(int(yy), int(mm), int(dd)) - datetime.datetime(int(yy),1,1)).days + 1
    path_glm = '/Users/jmceachern/data/noaa-goes16/GLM-L2-LCFA/'+str(yy)+'/'+str(j_day)+'/'+str(tt)+'/'

    list_of_paths = glob.glob(path_glm+'*.nc', recursive=True)

    ds_lis = []
    for name in list_of_paths:
        ds = Dataset(name)
        ds_lis.append(ds)
    end = time()

    print('get_glm: ',end-start)
    return ds_lis

def extract_vars(ds_w):
    """""
    In: list of WRF datasets (1 item per timestep) and extracts / calculates all desired variables 
    Out: 2D array where each column is a flat (1D) version of the WRF variables (matrix X for ML models)
    """""
    start = time()
    all_vars_lis = []
    for i in range(len(extract_times(ds_w,timeidx=ALL_TIMES))):
        pw = getvar(ds_w,'pw', timeidx = i)
        tt = np.full(pw.shape,np.sin(extract_times(ds_w,timeidx=i,do_xtime='XTIME')/60*np.pi/23)) # cyclic time
        cape_cin_LCL = getvar(ds_w,'cape_2d', timeidx = i)
        cape2 = cape_cin_LCL[0]
        cin2 = cape_cin_LCL[1]
        LCL = cape_cin_LCL[2]
        pblh = getvar(ds_w,'PBLH',timeidx = i)
        td_2 = getvar(ds_w,'td2',units='K', timeidx = i)
        Td = getvar(ds_w,'td',units='K', timeidx = i)
        z = getvar(ds_w,"height_agl", units = "m", timeidx = i)
        P = getvar(ds_w,"p",units='Pa', timeidx = i)
        t = getvar(ds_w, 'temp',units='K', timeidx = i)
        T2 = getvar(ds_w,'T2', timeidx = i)
        w = getvar(ds_w, "wa", units = "m s-1", timeidx = i) 
        wspd_wdir = getvar(ds_w, 'wspd_wdir', timeidx = i)
        wspd, wdir = wspd_wdir[0], wspd_wdir[1]
        bulk_shear = 6000 # m
        wspd_bulk = interplevel(wspd,z,bulk_shear)
        wspd_pbl = interplevel(wspd,z,pblh)
        wspd_surf = wspd[0,:,:]
        shear_1 = wspd_pbl - wspd_surf
        shear_2 = wspd_bulk - wspd_pbl
        shear_3 = wspd_bulk - wspd_surf
        qi = np.abs(getvar(ds_w,"QICE", timeidx = i))*1000 # ice mixing ratio (g/kg)
        qs = np.abs(getvar(ds_w, "QSNOW", timeidx = i))*1000 # snow mixing ratio (g/kg)
        qg = np.abs(getvar(ds_w, "QGRAUP", timeidx = i))*1000 # graupel mixing ratio (g/kg)
        lat = getvar(ds_w,'lat', timeidx = i)
        lon = getvar(ds_w,'lon', timeidx = i)
        K, LI, TT, SWI, Q = calc_LI_TT_SWI_Q(z,P,t,T2,LCL,Td,td_2,w,wspd,wdir,qi,qs,qg,lat)
        mdbz = getvar(ds_w,'mdbz', timeidx = i)
        ctt = getvar(ds_w,'ctt',units = 'K', timeidx = i) # cloud top temperature
        td_depress = t-Td
        td_depress_sur = td_depress[0,:,:]
        td_pbl = interplevel(Td,z,pblh)
        slp = getvar(ds_w,'slp',units = 'atm', timeidx = i)
        ter = getvar(ds_w,'ter',units = 'm', timeidx = i) # terrain model height
        rain = getvar(ds_w,'RAINC') # convective precipitation
        tsk = getvar(ds_w,'TSK') # surface skin temperature
        rh2 = getvar(ds_w,'rh2') # 2m relative humidity
        cld = getvar(ds_w,'cloudfrac')
        low_cld = cld[0]
        mid_cld = cld[1]

        vars_li = [pw,cape2,cin2,LCL,pblh,td_pbl,K,TT,td_2,LI,T2,SWI,Q,mdbz,lat,lon,ctt,td_depress_sur,slp,ter,rain,tsk,rh2,low_cld,mid_cld,shear_1,shear_2,shear_3,np.sum(qi,axis=0),np.sum(qs,axis=0)]
        processed_li = []
        processed_li.append(np.array(tt).flatten()) # we cant normalize time bc every value is the same for 1 hr of data, so it stays out of the loop
        for i,var in enumerate(vars_li):
            flat = np.array(var).flatten()
            norm = (flat - np.sort(flat)[0]) / (-np.sort(-flat)[0] - np.sort(flat)[0])
            nanless = np.where(norm>-999,norm,0)
            processed_li.append(nanless)
        all_vars_lis.append(np.array(processed_li).T)
    maxtix_X = np.concatenate(all_vars_lis,axis=0)

    end = time()
    print("extract wrf vars: ", end-start)
    return maxtix_X

def make_array(wrf_ds, glm_lis):
    """""
    In: WRF data (used only to duplicate its format) and list of GLM data where each item is one hours worth of data.
    Out: GLM data in 2D array of same shape as WRF array with each flash being placed at its specified lat and lon
    """""
    start = time()
    latlis = []
    lonlis = []
    for ds in glm_lis:
        # extract lats and lons from GLM array and add to lists
        latlis.append(ds.variables['flash_lat'][:])
        lonlis.append(ds.variables['flash_lon'][:])
    # concatinate lists to get one long column representing 1 hours of GLM obs. 
    lat = np.concatenate(latlis)
    lon = np.concatenate(lonlis)

    lats = getvar(wrf_ds,"XLAT")
    lons = getvar(wrf_ds,"XLONG")
    shape = lats.shape # shape of wrf array

    try:
        ## try and open kdtree for domain
        lightning_tree, lightning_loc = pickle.load(open('/Users/jmceachern/lightning/KDTree/lightning_tree.p', "rb"))
        print('Found lightning Tree')
    except:
        ## build a kd-tree 
        print("Could not find KDTree building....")
        ## create dataframe with columns of all lat/long in the domian...rows are cord pairs 
        lightning_locs = pd.DataFrame({"XLAT": lats.values.ravel(), "XLONG": lons.values.ravel()})
        ## build kdtree
        lightning_tree = KDTree(lightning_locs)
        ## save tree
        pickle.dump([lightning_tree, lightning_locs], open('/Users/jmceachern/lightning/KDTree/lightning_tree.p', "wb"))
        print("KDTree built")

    df = pd.DataFrame()
    df['lon']=lon
    df['lat']=lat

    south_north,  west_east = [], []
    for loc in df.itertuples(index=True, name='Pandas'):
        ## arange lightning lat and long in a formate to query the kdtree
        single_loc = np.array([loc.lat, loc.lon]).reshape(1, -1)

        ## query the kdtree retuning the distacne of nearest neighbor and the index on the raveled grid
        flash_dist, flash_ind = lightning_tree.query(single_loc, k=1)

        ## set condition to pass on flshes outside model domian 
        if flash_dist > 0.5:
            pass
        else:
            ## if condition passed reformate 1D index to 2D indexes
            ind = np.unravel_index(flash_ind[0][0], shape)
            ## append the indexes to lists
            south_north.append(ind[0])
            west_east.append(ind[1])

    tup_lis = tuple(zip(south_north,west_east))
    count = []
    for tup in tup_lis:
        count.append(tup_lis.count(tup)) # this counts duplicate tuples (gets the count for lightning in each gridcell)

    new_lis = tuple(zip(tup_lis,count))
    no_repeats = list(dict.fromkeys(new_lis)) # this gets rid of repeats

    rows, cols = list(zip(*list(zip(*no_repeats))[0]))[0], list(zip(*list(zip(*no_repeats))[0]))[1]
    count = list(zip(*no_repeats))[1]

    ds_final = xr.DataArray(np.zeros_like(lats))
    for i,num in enumerate(count):
        ds_final[rows[i],cols[i]] = num

    end = time()
    print("make glm array with KDTree: ",end-start)
    return ds_final.to_numpy()

def get_X_y(name,ytr,mtr,dtr,time_lis):
    """""
    In: testing year, month, day, and hours (as a list)
    Out: the X,y array of calculated WRF features, the GLM training data as a list where each item in the list is 1 hour worth of GLM data
    """""
    wrf_ds_train = get_wrf(name)
    X = extract_vars(wrf_ds_train)

    glm_train_lis = []
    for t in time_lis:
        glm_lis_temp = get_glm(ytr, mtr, dtr, t)
        glm_ds_train = make_array(wrf_ds_train,glm_lis_temp)
        glm_flat = glm_ds_train.flatten()
        glm_train_lis.append(glm_flat)
    y = np.concatenate(glm_train_lis,axis=0)

    return y, X

def make_models(i,X_train,y_train):
    """""
    In: the training data from wrf and glm. It creates the classification and regression models
    Out: dictionary of models where key= model name, value= model
    """""
    path = '/Users/jmceachern/lightning/experimental_models/'

    # if decision == 't':
    y_train_bi = np.where(y_train>0, 1,0)
    #training all models and calculate time it takes 
    # s = time()
    # clf_LogReg = LogisticRegression().fit(X_train, y_train_bi)
    # e = time()
    # print(f'LogisticRegression train time: {e-s} seconds')
    # s = time()
    # clf_Gauss = GaussianNB().fit(X_train,y_train_bi) # pros: fast to train, high AUC, cons: overpredicts, have to set threshold
    # e = time()
    # print(f'GaussianNB train time: {e-s} seconds')
    # s = time()
    # clf_DecTre = DecisionTreeClassifier().fit(X_train,y_train_bi) # pros: not biased, not setting thresh, cons: slow
    # e = time()
    # print(f'DecisionTreeClassifier train time: {e-s} seconds')
    s = time()
    clf_RanFor = RandomForestClassifier(n_estimators=52).fit(X_train,y_train_bi) # pros: high CSI, good performance, cons: super slow.
    e = time()
    print(f'RandomForestClassifier{i} train time: {e-s} seconds')
    # s = time()
    # clf_HistGB = HistGradientBoostingClassifier().fit(X_train,y_train_bi) 
    # e = time()
    # print(f'HistGradientBoostingClassifier train time: {e-s} seconds')
    # s = time()
    # we are only training regressor on flash data. No class 0
    # find indexes where training GLM data shows a flash
    idx_flash_train = np.where(y_train_bi == 1)[0]
    # Index the gridcells where the flashes are occuring
    X_train_flash = X_train[idx_flash_train]
    y_train_flash = y_train[idx_flash_train]
    s = time()
    est = HistGradientBoostingRegressor().fit(X_train_flash,y_train_flash)
    e = time()
    print(f'HistGradientBoostingRegressor{i} train time: {e-s} seconds')
    # s = time()
    # clf_Perc = Perceptron().fit(X_train, y_train_bi)
    # e = time()
    # print(f'Perceptron train time: {e-s} seconds')
    # s = time()
    # clf_Pass = PassiveAggressiveClassifier().fit(X_train, y_train_bi)
    # e = time()
    # print(f'PassiveAggressiveClassifier train time: {e-s} seconds')

    # save models to models directory
    # joblib.dump(clf_LogReg, open(path + 'LogisticRegression.pkl','wb'), 9) #the 9 here is the highest compression. Slows the time of saving/loading but saves disk space
    # joblib.dump(clf_Gauss, open(path + 'GaussianNB.pkl','wb'), 9)
    # joblib.dump(clf_DecTre, open(path + 'DecisionTreeClassifier.pkl','wb'), 9)
    joblib.dump(clf_RanFor, open(path + f'RandomForestClassifier{i}.pkl','wb'), 9)
    # joblib.dump(clf_HistGB, open(path + 'HistGradientBoostingClassifier.pkl','wb'), 9)
    joblib.dump(est, open(path + f'HistGradientBoostingRegressor{i}.pkl','wb'), 9)
    # joblib.dump(clf_Perc, open(path + 'PerceptronClassifier.pkl','wb'), 9)
    # joblib.dump(clf_Pass, open(path + 'PassiveAggressiveClassifier.pkl','wb'), 9)

    # return {'LogisticRegression':clf_LogReg,'GaussianNB':clf_Gauss,'DecisionTreeClassifier':clf_DecTre,'RandomForestClassifier':clf_RanFor,'HistGradientBoostingClassifier':clf_HistGB,'HistGradientBoostingRegressor':est,'PerceptonClassifier':clf_Perc,'PassiveAgressiveClassifier':clf_Pass}
    return {'RandomForestClassifier':clf_RanFor,'HistGradientBoostingRegressor':est}

def retreive_model(i):
    """""
    if models are already trained, this function retreives them
    In: index that specifies model
    Out: dictionary of models where key= model name, value= model
    """""
    path = '/Users/jmceachern/lightning/experimental_models/'
    name = f'RandomForestClassifier{i}.pkl'
    savefile = open(path + name,'rb')
    clf_RanFor = joblib.load(savefile)
    # name = 'LogisticRegression.pkl'
    # savefile = open(path + name,'rb')
    # clf_LogReg = joblib.load(savefile)
    # name = 'GaussianNB.pkl'
    # savefile = open(path + name,'rb')
    # clf_Gauss = joblib.load(savefile)
    # name = 'DecisionTreeClassifier.pkl'
    # savefile = open(path + name,'rb')
    # clf_DecTre = joblib.load(savefile)
    # name = 'HistGradientBoostingClassifier.pkl'
    # savefile = open(path + name,'rb')
    # clf_HistGB = joblib.load(savefile)
    name = f'HistGradientBoostingRegressor{i}.pkl'
    savefile = open(path + name,'rb')
    est = joblib.load(savefile)
    # name = 'PerceptronClassifier.pkl'
    # savefile = open(path + name,'rb')
    # clf_Perc = joblib.load(savefile)
    # name = 'PassiveAggressiveClassifier.pkl'
    # savefile = open(path + name,'rb')
    # clf_Pass = joblib.load(savefile)
    return {'RandomForestClassifier':clf_RanFor,'HistGradientBoostingRegressor':est}

def make_decision(ds_w, model_dict, key, X_test, y_test = None):

    # arbitrary variable used for shape
    t = getvar(ds_w,'T2')
    shape = t.shape

    # setting threshold for classifiers (determined by hyperparameter tuning) and then reshaping using random variable
    # thresh_LogReg = find_optimal_threshold(model_dict['LogisticRegression'].predict_proba(X_test)[:,1], y_test)
    # thresh_Gauss = find_optimal_threshold(model_dict['GaussianNB'].predict_proba(X_test)[:,1], y_test)
    # thresh_DecTre = find_optimal_threshold(model_dict['DecisionTreeClassifier'].predict_proba(X_test)[:,1], y_test)
    # thresh_RanFor = find_optimal_threshold(model_dict['RandomForestClassifier'].predict_proba(X_test)[:,1], y_test)
    # thresh_HistGB = find_optimal_threshold(model_dict['HistGradientBoostingClassifier'].predict_proba(X_test)[:,1], y_test)

    # cla_LogReg = np.reshape(np.where(model_dict['LogisticRegression'].predict_proba(X_test)[:,1]>0.041,1,0),shape)
    # cla_Gauss = np.reshape(np.where(model_dict['GaussianNB'].predict_proba(X_test)[:,1]>0.99,1,0),shape)
    # cla_DecTre = np.reshape(np.where(model_dict['DecisionTreeClassifier'].predict_proba(X_test)[:,1]>0.020408,1,0),shape)
    # cla_RanFor = np.reshape(np.where(model_dict['RandomForestClassifier'].predict_proba(X_test)[:,1]>0.12,1,0),shape)
    # cla_HistGB = np.reshape(np.where(model_dict['HistGradientBoostingClassifier'].predict_proba(X_test)[:,1]>0.076,1,0),shape)
    # cla_LogReg = np.reshape(np.where(model_dict['LogisticRegression'].predict_proba(X_test)[:,1]>thresh_LogReg,1,0),shape)
    # cla_Gauss = np.reshape(np.where(model_dict['GaussianNB'].predict_proba(X_test)[:,1]>thresh_Gauss,1,0),shape)
    # cla_DecTre = np.reshape(np.where(model_dict['DecisionTreeClassifier'].predict_proba(X_test)[:,1]>thresh_DecTre,1,0),shape)
    cla_RanFor = np.reshape(np.where(model_dict['RandomForestClassifier'].predict_proba(X_test)[:,1]>0.1423,1,0),shape)
    # cla_HistGB = np.reshape(np.where(model_dict['HistGradientBoostingClassifier'].predict_proba(X_test)[:,1]>thresh_HistGB,1,0),shape)

    # # for perceptron model, there is no predict_proba API so we normalize the decision function
    # decision = model_dict['PerceptonClassifier'].decision_function(X_test)
    # decision_norm = (decision - np.min(decision)) / (np.max(decision) - np.min(decision))
    # thresh_Perc = find_optimal_threshold(decision_norm, y_test)
    # cla_Perc = np.reshape(np.where(decision_norm>0.79,1,0),shape)

    # decision = model_dict['PassiveAgressiveClassifier'].decision_function(X_test)
    # decision_norm = (decision - np.min(decision)) / (np.max(decision) - np.min(decision))
    # thresh_Pass = find_optimal_threshold(decision_norm, y_test)
    # cla_Pass = np.reshape(np.where(decision_norm>0.77,1,0),shape)

    # all the models will be compared to this array of random values
    # create array of 0's and 1's that has the same ratio of 1's as the target values
    # random = np.zeros_like(y_test)
    # rng = np.random.default_rng(12345)
    # rints = rng.integers(low=0, high=len(y_test), size=(y_test>0).sum())
    # random[rints] = 1
    # y_random = np.reshape(random,shape) # reshape array of random values to 2d

    # decision_dict = {'LogisticRegression':cla_LogReg,'GaussianNB':cla_Gauss,'DecisionTreeClassifier':cla_DecTre,'RandomForestClassifier':cla_RanFor,'HistGradientBoostingClassifier':cla_HistGB,'PerceptronClassifier':cla_Perc,'PassiveAgressiveClassifier':cla_Pass, 'Random':y_random}
    # thresh_dict = {'LogisticRegression':thresh_LogReg,'GaussianNB':thresh_Gauss,'DecisionTreeClassifier':thresh_DecTre,'RandomForestClassifier':thresh_RanFor,'HistGradientBoostingClassifier':thresh_HistGB,'PerceptronClassifier':thresh_Perc,'PassiveAgressiveClassifier':thresh_Pass}

    # decision_dict = {'RandomForestClassifier':cla_RanFor, 'Random':y_random}
    decision_dict = {'RandomForestClassifier':cla_RanFor}
    # thresh_dict = {'RandomForestClassifier':thresh_RanFor}

    # use the new regression model where the classifier == 1
    idx_flash_test = np.where(decision_dict[key].flatten() == 1)[0]
    combined = model_dict['HistGradientBoostingRegressor'].predict(X_test[idx_flash_test])

    # consruct 2d array where regression values are used where classifier == 1 and zero everywhere else
    combined_1d = decision_dict[key].flatten() 
    for i,index in enumerate(idx_flash_test):
        combined_1d[index] = combined[i]

    combined_2d = np.reshape(combined_1d,shape)

    # add combined to dictionary
    decision_dict['Combination'] = combined_2d

    return decision_dict

def find_optimal_threshold(y_preds, glm_target):

    threshs = np.linspace(0,1) # defines thresholds between 0 and 1 (default is 0.5)
    csis = np.zeros(len(threshs))

    for i,t in enumerate(threshs):
        #make a dummy binary array full of 0s
        y_preds_bi = np.zeros(y_preds.shape)
        
        #find where the prediction is greater than or equal to the threshold
        #set those indices to 1
        y_preds_bi[np.where(y_preds >= t)] = 1

        try:
            #find statistics for the predictions at this specific threshold
            acc, POD, POFD, SR, CSI, bias, continge_dict = contingency_table(y_preds_bi, glm_target.flatten())
        except ZeroDivisionError:
            break
        # print(POD,t)

        csis[i] = CSI

    idx = np.where(csis == np.max(csis))[0][0] # index of max csi value 

    return threshs[idx]

def contingency_table(y_hat, y):
    """""
    In: y_hat = model prediction, y = observations
    Out = true positives, false positive, false negatives, true negatives as ditionary
    """""
    TP_KEY = 'num_true_positives'
    FP_KEY = 'num_false_positives'
    FN_KEY = 'num_false_negatives'
    TN_KEY = 'num_true_negatives'

    reg_binary = np.where(y_hat>=1, 1,0).flatten() # is 1 when there is 1 or more strikes, zero when less than 1 strike
    glm_binary = np.where(y>=1, 1,0).flatten()

    true_pos_indices = np.where(np.logical_and(reg_binary==1,glm_binary==1))[0]
    false_pos_indices = np.where(np.logical_and(reg_binary==1,glm_binary==0))[0]
    false_neg_indices = np.where(np.logical_and(reg_binary==0,glm_binary==1))[0]
    true_neg_indices = np.where(np.logical_and(reg_binary==0,glm_binary==0))[0]

    continge_dict = {TP_KEY: len(true_pos_indices), FP_KEY: len(false_pos_indices), FN_KEY: len(false_neg_indices), TN_KEY: len(true_neg_indices)}

    accuracy = (continge_dict['num_true_positives'] + continge_dict['num_true_negatives']) / (continge_dict['num_true_positives'] + continge_dict['num_true_negatives'] + continge_dict['num_false_positives'] + continge_dict['num_false_negatives']) * 100
    POD = continge_dict['num_true_positives'] / (continge_dict['num_true_positives'] + continge_dict['num_false_negatives']) # probability of detedtion (POD)
    POFD = continge_dict['num_false_positives'] / (continge_dict['num_false_positives'] + continge_dict['num_true_negatives']) # probability of false detection (POFD)
    SR = continge_dict['num_true_positives'] / (continge_dict['num_true_positives'] + continge_dict['num_false_positives']) # Success ratio (SR)
    CSI = continge_dict['num_true_positives'] / (continge_dict['num_true_positives'] + continge_dict['num_false_positives'] + continge_dict['num_false_negatives'])
    bias = (continge_dict['num_true_positives'] + continge_dict['num_false_positives']) / (continge_dict['num_true_positives'] + continge_dict['num_false_negatives'])

    return accuracy, POD, POFD, SR, CSI, bias, continge_dict


# %%
