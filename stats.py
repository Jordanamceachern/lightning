# %%
import numpy as np
from run import make_models, make_decision, retreive_model, get_wrf, get_X_y
from wrf import getvar, to_np, getvar, latlon_coords, get_basemap
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, roc_auc_score
from time import time 
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
import pickle 
import imageio as io
# %%
def permute_import(model_dict, all_vars, glm_target):
    """""
    In: dictionaly of models with key=name of model, value=model, and glm list
    Out: plot of permutation importance for the model 
    """""
    feature_names = np.array(['LPI','pw','cape','cin','LCL','LFC','abs_vort','K','TT','td_2','LI','T2','SWI','Q','mdbz','lat','lon','ctt','omega','slp','ter','conv_precip','skin_temp','rh2','low_cld','mid_cld'])

    for key in model_dict.keys():
        print(key)
        result = permutation_importance(model_dict[key], all_vars, np.where(glm_target.flatten()>0,1,0), n_repeats=5,random_state=0)

        fig, ax = plt.subplots()

        indices = result["importances_mean"].argsort()
        plt.barh(
            range(len(indices)),
            result["importances_mean"][indices],
            xerr=result["importances_std"][indices],
        )
        plt.title(f'Permutation Importance (model: {key})')
        ax.set_yticks(range(len(indices)))
        _ = ax.set_yticklabels(feature_names[indices])
    
    return

def make_figs(ds_w,decision_dict,key,date,tt,y_test = None):

    t = getvar(ds_w, "T2") 

    glm = np.reshape(y_test,t.shape)
    cla = decision_dict[key]
    both = decision_dict['Combination']

    lats, lons = latlon_coords(t)
    bm = get_basemap(t)
    x, y = bm(to_np(lons), to_np(lats))

    fig = plt.figure(figsize = (30,30))

    ax1 = fig.add_subplot(221)
    bm.drawcoastlines(linewidth=0.25)
    bm.drawstates(linewidth=0.25)
    bm.drawcountries(linewidth=0.25)
    levels = np.array([0,1,2,5,10,15,20,25,30,50,100])
    norm = matplotlib.colors.BoundaryNorm(levels,len(levels))
    colors = list(plt.cm.Greys(np.linspace(0,1,len(levels)-1)))
    colors[0] = "w"
    colors[1] = "dodgerblue"
    colors[2] = "mediumseagreen"
    colors[3] = "seagreen"
    colors[4] = "greenyellow"
    colors[5] = "yellow"
    colors[6] = "orange"
    colors[7] = "red"
    colors[8] = "magenta"
    colors[9] = "blueviolet"
    contours = bm.contourf(x, y, both, levels=levels, colors="dodgerblue")
    cmap = matplotlib.colors.ListedColormap(colors,"", len(colors))
    # cmap = matplotlib.colors.ListedColormap(colors[::-1],"", len(colors))
    im = bm.contourf(x, y, to_np(both), levels, cmap=cmap, norm=norm)
    plt.colorbar(ticks=levels,shrink=0.2)
    ax1.set_title(f'Experimental Lightning Forecast \ninit: 00z July 10 2024                                     valid at: {date[0]}/{date[1]}/{date[2]} {tt}:00:00 UTC',fontsize=20)
    ax1.set_xlabel('flashes/gridcell/hr',fontsize=20)

    ax2 = fig.add_subplot(222)
    bm.drawcoastlines(linewidth=0.25)
    bm.drawstates(linewidth=0.25)
    bm.drawcountries(linewidth=0.25)
    levels = np.array([0,1,2,5,10,15,20,25,30,50,100])
    norm = matplotlib.colors.BoundaryNorm(levels,len(levels))
    colors = list(plt.cm.Greys(np.linspace(0,1,len(levels)-1)))
    colors[0] = "w"
    colors[1] = "dodgerblue"
    colors[2] = "mediumseagreen"
    colors[3] = "seagreen"
    colors[4] = "greenyellow"
    colors[5] = "yellow"
    colors[6] = "orange"
    colors[7] = "red"
    colors[8] = "magenta"
    colors[9] = "blueviolet"
    contours = bm.contourf(x, y, glm, levels=levels, colors="dodgerblue")
    cmap = matplotlib.colors.ListedColormap(colors,"", len(colors))
    im = bm.contourf(x, y, glm, levels, cmap=cmap, norm=norm)
    ax2.set_title(f'Goestationary Lightning Mapper Satellite Obs. \nvalid at: {date[0]}/{date[1]}/{date[2]} {tt}:00:00 UTC',fontsize=20)
    ax2.set_xlabel('flashes/gridcell/hr',fontsize=20)
    plt.colorbar(ticks=levels, shrink=0.2, format='%.0f')

    # ax3 = fig.add_subplot(121)
    # bm.drawcoastlines(linewidth=0.25)
    # bm.drawstates(linewidth=0.25)
    # bm.drawcountries(linewidth=0.25)
    # levels = np.linspace(-1,2,1)
    # # levels = np.array([-1,0,1,2])
    # levels = np.array([0,0.5,1])
    # norm = matplotlib.colors.BoundaryNorm(levels,len(levels))
    # colors = list(plt.cm.Greys(np.linspace(0,1,len(levels)-1)))
    # cmap = matplotlib.colors.ListedColormap(colors,"", len(colors))
    # im = bm.contourf(x, y, to_np(cla), levels, cmap=cmap, norm=norm)
    # plt.colorbar(ticks=levels, shrink=0.15)
    # ax3.set_title(f'Classifier: {key}',fontsize=20)

    # ax4 = fig.add_subplot(122)
    # bm.drawcoastlines(linewidth=0.25)
    # bm.drawstates(linewidth=0.25)
    # bm.drawcountries(linewidth=0.25)
    # # levels = np.array([-2,np.mean(reg),1,2,3,4,5,6,7,8,9,10,np.max(reg)+1]) 
    # # levels = np.array([np.min(reg),np.mean(reg),1,2,4,8,10,np.max(reg)])
    # # levels = np.array([0,1,2,5,10,15,20,25,30,50,100,np.max(reg)+1]) 
    # # norm = matplotlib.colors.BoundaryNorm(levels,len(levels))
    # # colors = list(plt.cm.inferno(np.linspace(0,1,len(levels)-1)))
    # # colors[-1] = "w"
    # levels = np.array([0,1,2,5,10,15,20,25,30,50,100])
    # norm = matplotlib.colors.BoundaryNorm(levels,len(levels))
    # colors = list(plt.cm.Greys(np.linspace(0,1,len(levels)-1)))
    # colors[0] = "w"
    # colors[1] = "dodgerblue"
    # colors[2] = "mediumseagreen"
    # colors[3] = "seagreen"
    # colors[4] = "greenyellow"
    # colors[5] = "yellow"
    # colors[6] = "orange"
    # colors[7] = "red"
    # colors[8] = "magenta"
    # colors[9] = "blueviolet"
    # # cmap = matplotlib.colors.ListedColormap(colors[::-1],"", len(colors))
    # contours = bm.contourf(x, y, to_np(reg), levels=levels, colors="dodgerblue")
    # cmap = matplotlib.colors.ListedColormap(colors,"", len(colors))
    # im = bm.contourf(x, y, to_np(reg), levels, cmap=cmap, norm=norm)
    # plt.colorbar(ticks=levels,shrink=0.15)
    # ax4.set_title('HBGB Regression',fontsize=20)
    # plt.ylim(lim)

    # ax5 = fig.add_subplot(122)
    # data=[['The threat of lighting is low or negligible.'],
    #     ['Weather conditions are conducive to thunderstorm formation.'],
    #     ['Thunderstorms are very likely in this region.'],
    #     ['There is a high risk of dangerous lighting activity.']]
    # column_labels=["Colour Key and Explanation"]

    # #creating a 2-dimensional dataframe out of the given data
    # df=pd.DataFrame(data,columns=column_labels)

    # ax5.axis('tight') #turns off the axis lines and labels
    # ax5.axis('off') #changes x and y axis limits such that all data is shown

    # #plotting data
    # table = ax5.table(cellText=df.values,
    #         colLabels=df.columns,
    #         rowLabels=["Low Risk","Moderate Risk","High Risk","Extreme Risk"],
    #         rowColours =['w','greenyellow','yellow','gold'],
    #         colColours =["red"],
    #         loc="center")
    # table.set_fontsize(40)
    # table.scale(0.70,1.8)


    plt.tight_layout(pad=5, w_pad=0.2)
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
    #             hspace = 0, wspace = 0)

    plt.savefig(f'forecast_img/forecast{date[0]}{date[1]}{date[2]}{tt}',bbox_inches='tight')
    plt.show()

    return fig 

def make_performance_diagram_axis(ax=None,figsize=(5,5),CSIBOOL=True,FBBOOL=True,csi_cmap='Greys_r'):
    import matplotlib.patheffects as path_effects
    pe = [path_effects.withStroke(linewidth=2,
                                 foreground="k")]
    pe2 = [path_effects.withStroke(linewidth=2,
                                 foreground="w")]

    if ax is None:
        fig=plt.figure(figsize=figsize)
        fig.set_facecolor('w')
        ax = plt.gca()
    
    
    if CSIBOOL:
        sr_array = np.linspace(0.001,1,200)
        pod_array = np.linspace(0.001,1,200)
        X,Y = np.meshgrid(sr_array,pod_array)
        csi_vals = (X ** -1 + Y ** -1 - 1.) ** -1
        pm = ax.contourf(X,Y,csi_vals,levels=np.arange(0,1.1,0.1),cmap=csi_cmap)
        plt.colorbar(pm,ax=ax,label='CSI')
    
    if FBBOOL:
        fb = Y/X
        bias = ax.contour(X,Y,fb,levels=[0.25,0.5,1,1.5,2,3,5],linestyles='--',colors='Grey')
        plt.clabel(bias, inline=True, inline_spacing=5,fmt='%.2f', fontsize=10,colors='LightGrey')
    
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel('SR')
    ax.set_ylabel('POD')

    return ax

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

def performance_diagram_points(model_dict,y_test,date,t,ax):
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(model_dict.keys()))))
    for key in model_dict.keys():
        print(key)
        acc, POD, POFD, SR, CSI, bias, continge_dict = contingency_table(model_dict[key], y_test)
        ax.plot(SR,POD,'o',color=next(colors),markerfacecolor='w',label=str(key));
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          fancybox=True, shadow=True, ncol=2)
    ax.set_title(f'Performance Diagram \n valid at: {date[0]}/{date[1]}/{date[2]} {t}:00:00 UTC');
    plt.savefig(f'performance_img/performance{date[0]}{date[1]}{date[2]}{t}',bbox_inches='tight')
    return

def tune_prediction_threshold(model_dict, X_test, y_test):
    ax = make_performance_diagram_axis()
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(model_dict.keys()))))
    for key in model_dict.keys():

        yhat_proba = model_dict[key].predict_proba(X_test) # finds the probability of each class
        y_preds = yhat_proba[:,1] # focusing only on the probability of class 1 (lightning occuring)

        threshs = np.linspace(0,1) # defines thresholds between 0 and 1 (default is 0.5)

        pods = np.zeros(len(threshs))
        srs = np.zeros(len(threshs))
        csis = np.zeros(len(threshs))

        for i,t in enumerate(threshs):
            #make a dummy binary array full of 0s
            y_preds_bi = np.zeros(y_preds.shape,dtype=int)
            
            #find where the prediction is greater than or equal to the threshold
            idx = np.where(y_preds >= t)
            #set those indices to 1
            y_preds_bi[idx] = 1
            try:
                #find statistics for the predictions at this specific threshold
                acc, POD, POFD, SR, CSI, bias, continge_dict = contingency_table(y_preds_bi, y_test)
            except ZeroDivisionError:
                break

            pods[i] = POD
            srs[i] = SR
            csis[i] = CSI

        ax.plot(srs,pods,'-',color=next(colors),markerfacecolor='w',lw=2,label=key)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
            fancybox=True, shadow=True, ncol=2);
    return

def make_ROC(model_dict,y,date,t):
    fig = plt.figure(figsize=(5,5))
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(model_dict.keys()))))
    y = np.where(y>=1,1,0).flatten()
    auc_dict = {}
    for key in model_dict.keys():
        yhat = np.where(model_dict[key]>=1,1,0).flatten()
        fpr, tpr, thresholds = roc_curve(y,yhat)
        AUC = roc_auc_score(y,yhat)
        auc_dict[key] = AUC
        plt.plot(fpr,tpr,color=next(colors),label=key+' (AUC='+str(round(AUC,3))+')')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve \n valid at: {date[0]}/{date[1]}/{date[2]} {t}:00:00 UTC')
    plt.plot(fpr,fpr,color='b',linestyle='dashed',label='no skill=0.5')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          fancybox=True, shadow=True, ncol=2)
    plt.savefig(f'ROC_curve_img/ROC_curve{date[0]}{date[1]}{date[2]}{t}',bbox_inches='tight');
    return auc_dict

def calc_FSS(key,decision_dict,y_1d,n):
    """""
    In: y_hat = model prediction (2d array), y = observations (2d array), n = window shape (2n+1 x 2n+1)
    Out: FSS
    """""
    p_y = []
    p_r = []
    p_o = []
    y_hat = decision_dict[key]
    y_test = np.reshape(y_1d,y_hat.shape) # reshapes target values to 2d array

    # create array of 0's and 1's that has the same ratio of 1's as the target values
    random = np.zeros_like(y_test)
    rng = np.random.default_rng(12345)
    rints = rng.integers(low=0, high=len(y_test), size=(y_test>0).sum())
    random[rints] = 1
    y_random = np.reshape(random,y_hat.shape) # reshape array of random values to 2d
    y_zeros = np.zeros_like(y_hat)

    for i in range(y_hat.shape[0]): #  i = row
        i_min = max(0, i - n) # chooses i - n unless near the bound
        i_max = min(y_hat.shape[0], i + n +1) # chooses i + n unless near the bound
        for ii in range(y_hat.shape[1]): # ii = column
            ii_min = max(0, ii - n)
            ii_max = min(y_hat.shape[1], ii + n + 1)

            window_num = (i_max-i_min)*(ii_max-ii_min) # number of obs in the window
            p_y.append((y_hat[i_min:i_max,ii_min:ii_max]).sum() / window_num)
            p_o.append((y_test[i_min:i_max,ii_min:ii_max]).sum() / window_num)
            p_r.append((y_random[i_min:i_max,ii_min:ii_max]).sum() / window_num)

    N = y_test.shape[0] * y_test.shape[1] # total number of observations in array
    p_y, p_o, p_r = np.array(p_y), np.array(p_o), np.array(p_r)
    FSS_withnans = 1/N*(p_y-p_o)**2 / (p_y**2/N + p_o**2/N)
    FSS_array = np.where(FSS_withnans>-1,FSS_withnans,0) # is 0/0, then == 0

    FSS_withnans_r = 1/N*(p_r-p_o)**2 / (p_r**2/N + p_o**2/N)
    FSS_array_r = np.where(FSS_withnans_r>-1,FSS_withnans_r,0) # is 0/0, then == 0

    return 1-np.mean(FSS_array), 1-np.mean(FSS_array_r)

def FSS_table(decision_dict,y_test,n):
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(decision_dict.keys()))))
    FSS_lis = []
    key_lis = []
    colour_lis = []
    for key in decision_dict.keys():
        FSS, FSS_thresh = calc_FSS(key,decision_dict,y_test,n)
        FSS_lis.append(round(FSS,3))
        key_lis.append(key)
        colour_lis.append(next(colors))

    df = pd.DataFrame(FSS_lis, index=key_lis, columns=['FSS'])

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df.values,
            colLabels=df.columns,
            rowLabels=key_lis,
            rowColours =colour_lis,
            colColours =["red"],
            loc="center")
    table.set_fontsize(40)
    table.scale(0.60,5)
    return

def plot_vars(var,var_dict,X,y):
    """""
    In: variable of choice, training data features (regular and log scaled), glm target values as list
    Out: histograms showing the values of the feature when there is no lightning (blue),
    and with lightning (red). The second figure shows the logscaled value (for the case of CAPE and CIN)
    """""

    idx_flash = np.where(y >= 1)[0]
    idx_noflash = np.where(y == 0)[0]

    var_idx = var_dict[var]
    var_flash = X[idx_flash][:,var_idx]
    var_noflash = X[idx_noflash][:,var_idx]
    sample_flash = np.random.choice(var_flash,size=500)
    sample_noflash = np.random.choice(var_noflash,size=500)

    fig, ax = plt.subplots(figsize=(10, 10))
    bins = np.linspace(0,1)

    ax.hist(sample_flash,bins=bins,alpha=0.5,color='red',zorder=0,label='flash = Y',edgecolor="black")
    ax.hist(sample_noflash,bins=bins,alpha=0.5,color='blue',zorder=0,label='flash = N',edgecolor="black")
    ax.set_title(var)
    ax.grid('on') 
    ax.legend()
    return

def plot_timeseries(key,testing_dates,times):
    FSS_lis = []
    FSS_thresh_lis = []
    AUC_lis = []
    CSI_lis = []
    t_lis = []

    for date in testing_dates:
        for t in times:
            t_lis.append(t)
            try:
                y_test, X_test = retreive_X_y_test(date,t)
                decision_dict = make_decision(wrf_ds, model_dict, key, X_test)
                # figx4 = make_figs(wrf_ds, decision_dict, key, date, t, y_test)
                small_dic = {key:decision_dict[key]}
                # ax = make_performance_diagram_axis()
                # performance_diagram_points(small_dic,y_test,date,t,ax)
                AUC_dict = make_ROC(small_dic,y_test,date,t)
                # plt.show()
                AUC_lis.append(AUC_dict[key])
                # n = 2
                FSS, FSS_thresh = calc_FSS(key,small_dic,y_test,n)
                FSS_lis.append(FSS)
                FSS_thresh_lis.append(FSS_thresh)
                accuracy, POD, POFD, SR, CSI, bias, continge_dict = contingency_table(y_test,small_dic[key])
                CSI_lis.append(CSI)

            except FileNotFoundError:
                break

    leadtime = np.arange(0,len(CSI_lis))
    new_tick_locations = np.arange(0,len(leadtime),6)
    # t_arr = np.array([0,6,12,18,0,6,12,18,0,12,18,0,6,12])

    # fig = plt.figure(figsize=(15,5))
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(111)
    # ax2 = ax1.twiny()
    ax1.plot(leadtime,CSI_lis,label='Random Forest Classifier',c='orchid')
    ax1.axhline(y=0,label='no skill',c='lightseagreen')
    ax1.set_title('Critical Success Index',fontsize=17)
    ax1.set_xlabel('Forecast Hour (hours)')
    ax1.set_ylim(-0.01,0.15) 
    # ax2.set_xlim(ax1.get_xlim())
    # ax2.set_xticks(new_tick_locations)
    # ax2.set_xticklabels(t_arr)
    ax1.set_xticks(new_tick_locations)
    # ax2.set_xlabel('Time of Day (UTC)')
    ax1.legend()


    fig = plt.figure(figsize=(11.3,5))
    # fig = plt.figure(figsize=(30,5))
    ax1 = fig.add_subplot(121)
    # ax2 = ax1.twiny()
    ax1.plot(leadtime,FSS_lis,label='Random Forest Classifier',c='orchid')
    ax1.plot(leadtime,FSS_thresh_lis,label='no skill',c='lightseagreen')
    ax1.set_ylim(0,1)
    # ax2.set_xlim(ax1.get_xlim())
    # ax2.set_xticks(new_tick_locations)
    # ax2.set_xticklabels(t_arr)
    ax1.set_xticks(new_tick_locations)
    # ax2.set_xlabel('Time of Day (UTC)')
    ax1.set_title('Fractional Skill Score',fontsize=17)
    ax1.set_xlabel('Forecast Hour (hours)')
    ax1.legend()

    fig = plt.figure(figsize=(17.5,5))
    # fig = plt.figure(figsize=(45,5))
    ax1 = fig.add_subplot(131)
    # ax2 = ax1.twiny()
    ax1.plot(leadtime,AUC_lis,label='Random Forest Classifier',c='orchid')
    ax1.axhline(y=0.5,label='no skill',c='lightseagreen')
    ax1.set_title('Area Under the ROC Curve',fontsize=17)
    ax1.set_xlabel('Forecast Hour (hours)')
    ax1.set_ylim(0.45,1)
    # ax2.set_xlim(ax1.get_xlim())
    # ax2.set_xticks(new_tick_locations)
    # ax2.set_xticklabels(t_arr)
    # ax2.set_xlabel('Time of Day (UTC)')
    ax1.set_xticks(new_tick_locations)
    ax1.legend()
    return

def permutation_calculator(times,date):
    decision = input('Retreiving or creating dictionary? ("r" or "c")?')
    
    if decision == 'r':
        # retreive CSI dictionary from pickle file 
        with open('CSI_dict.pkl', 'rb') as f:
            CSI_dict = pickle.load(f)

    if decision == 'c':
        CSI_dict = {'model':[],'tt':[]}
        for i,t in enumerate(times):
            print(t)
            y_test, X_test = retreive_X_y_test(date,t)
            decision_dict = make_decision(wrf_ds, model_dict, key, X_test, y_test)
            accuracy, POD, POFD, SR, CSI_0, bias, continge_dict = contingency_table(y_test,decision_dict[key])
            CSI_dict['model'].append(CSI_0)
            CSI_dict['tt'].append(int(t))
            for var in var_dict.keys():
                y_test, X_test = retreive_X_y_test(date,t)
                if var == 'time':
                    X_test[:,var_dict["time"]] = np.random.uniform(low=0, high=1, size=(len(y_test),))
                else:
                    np.random.shuffle(X_test[:,var_dict[var]])
                decision_dict = make_decision(wrf_ds, model_dict, key, X_test, y_test)
                accuracy, POD, POFD, SR, CSI, bias, continge_dict = contingency_table(y_test,decision_dict[key])
                if i == 0:
                    CSI_dict[var] = []
                CSI_dict[var].append(CSI)

        with open('CSI_dict.pkl', 'wb') as f:
            pickle.dump(CSI_dict, f)
    return CSI_dict

def permute_timeseries(CSI_dict,var):
    color = (0.2, # redness
            0.4, # greenness
            0.3, # blueness
            0.6 # transparency
            ) 
    plt.plot(np.array(CSI_dict['tt']),np.array(CSI_dict['model']) - np.array(CSI_dict[var]), label=var, color=color)
    plt.axhline(y = 0, color = 'pink', linestyle = '--',alpha = 1, label = 'no contribution') 
    plt.title(f'Permutation Importance of {var}')
    plt.ylabel('CSI contribution')
    plt.xlabel('Time (UTC)')
    plt.legend()
    return

def permutation_plot(CSI_dict):
    CSI_mean = {'idx':[],'mean':[]}
    for i,var in enumerate(var_dict.keys()):
        CSI_mean['mean'].append(np.mean(np.array(CSI_dict['model']) - np.array(CSI_dict[var])))
        CSI_mean['idx'].append(i)

    ordered_idx = np.array(CSI_mean['mean']).argsort()
    ordered_means = np.array(CSI_mean['mean'])[ordered_idx]

    color = (0.2, # redness
            0.4, # greenness
            0.3, # blueness
            0.6 # transparency
            ) 
    fig, ax = plt.subplots(figsize=(10, 11))
    plt.barh(range(len(ordered_idx)),ordered_means,color=color)
    ax.set_yticks(range(len(ordered_idx)))
    _ = ax.set_yticklabels(np.array(list(var_dict.keys()))[ordered_idx])
    plt.xlabel('CSI contribution', fontsize=15)
    plt.title("Permutation Importance", fontsize=20)
    return

def train_preprocess(training_dates,times):
    for date in training_dates:
        s = time()
        name_train = 'wrfout_d02_'+date[0]+date[1]+date[2]+'_merged.nc'
        y_train, X_train = get_X_y(name_train,date[0],date[1],date[2],times)
        np.save('preprocessed/y_train/y_train'+date[0]+date[1]+date[2]+'.npy',y_train)
        np.save('preprocessed/X_train/X_train'+date[0]+date[1]+date[2]+'.npy',X_train)
        e = time()
        print(f'time for one day: {e-s}')
    return

def test_preprocess(testing_dates,times):
    for date in testing_dates:
        for t in times:
            try:
                name_test = 'wrfout_d02_'+date[0]+'-'+date[1]+'-'+date[2]+'_'+t+':00:00'
                y_test = get_X_y(name_test,date[0],date[1],date[2],[t])
                # y_test, X_test = get_X_y(name_test,date[0],date[1],date[2],[t])
                np.save('preprocessed/y_test/y_test'+date[0]+date[1]+date[2]+t+'.npy',y_test)
                # np.save('preprocessed/X_test/X_test'+date[0]+date[1]+date[2]+t+'.npy',X_test)
            except FileNotFoundError:
                break
    return

def retreive_X_y(dates):
    y_lis = []
    X_lis = []
    for date in dates:
        y_lis.append(np.load('preprocessed/y_train/y_train'+date[0]+date[1]+date[2]+'.npy'))
        X_lis.append(np.load('preprocessed/X_train/X_train'+date[0]+date[1]+date[2]+'.npy'))
    return np.concatenate(y_lis), np.concatenate(X_lis)

def retreive_X_y_test(date,time):
    y = np.load('preprocessed/y_test/y_test'+date[0]+date[1]+date[2]+time+'.npy')
    X = np.load('preprocessed/X_test/X_test'+date[0]+date[1]+date[2]+time+'.npy')
    return y, X

# %%
times = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
# training_dates = [('2024','06','06'),('2024','06','07'),('2024','06','09'),('2024','06','10'),('2024','06','11'),('2024','06','12'),('2024','06','13'),('2024','06','14'),('2024','06','15'),('2024','06','16'),('2024','06','18'),('2024','06','19'),('2024','06','20'),('2024','06','21')]
training_dates = [('2024','07','23'),('2024','07','24'),('2024','07','25'),('2024','07','26'),('2024','07','27')]
testing_dates = [('2023','06','06')]
# testing_dates = [('2024','07','10'),('2024','07','11'),('2024','07','12'),('2024','07','13')]
# %%
train_preprocess(training_dates,times)
# test_preprocess(testing_dates,times)
# %%
y_train_all, X_train_all = retreive_X_y(training_dates)
# %%
# model_dict = make_models(X_train_all, y_train_all)
test_preprocess(testing_dates,times)
# %%
model_dict = make_models()
# %%
key = 'RandomForestClassifier' # the classifier model that will be combined with the regression
wrf_ds = get_wrf('wrfout_d02_20240605_merged.nc') # this is just to reconstruct the flat arrays.
# wrf_ds = get_wrf('wrfout_d02_20240612_merged.nc')
# %%
date = ('2023','06','06')
t = '23'
y_test, X_test = retreive_X_y_test(date,t)
key = 'RandomForestClassifier'
# %%
decision_dict = make_decision(wrf_ds, model_dict, key, X_test, y_test)
# %%
figx4 = make_figs(wrf_ds, decision_dict, key, date, t, y_test)
# %%
# decision_dict.pop('Combination')
# decision_dict.pop('GaussianNB')
ax = make_performance_diagram_axis()
performance_diagram_points(decision_dict,y_test,date,t,ax)
# %%
n = 2
AUC_dict = make_ROC(decision_dict,y_test,date,t)
# FSS, FSS_thresh = calc_FSS(key,decision_dict,y_test,n)
# %%
n = 2
FSS_table(decision_dict,y_test,n)
# %%
tune_prediction_threshold(model_dict,X_test,y_test)
# %%
n=2
key = 'RandomForestClassifier'
plot_timeseries(key,testing_dates,times)
# %%
var_dict = {'time':0,'pw':1,'cape':2,'cin':3,'LCL':4,'PBLH':5,'td_PBL':6,'K':7,'TT':8,'td_2':9,'LI':10,'T2':11,'SWI':12,'Q':13,'mdbz':14,'lat':15,'lon':16,'ctt':17,'td_dep':18,'slp':19,'ter':20,'rain':21,'tsk':22,'rh2':23,'low_cld':24,'mid_cld':25,'shear_1':26,'shear_2':27,'shear_3':28,'q_ice':29,'q_snow':30}
var = 'rh2'
plot_vars(var,var_dict,X_train_all,y_train_all)
# %%
date = ('2024','07','10')
CSI_dict = permutation_calculator(times,date)
permute_timeseries(CSI_dict,var)
permutation_plot(CSI_dict)
# %%

"""
to retreive the data from bluesky (when in wrf_data directory): scp 'jmceachern@bluesky2:/nfs/kitsault/archives/forecasts/WAN00CG-01/24060700/wrfout_d02_2024-06*' .
to merge all hours into 1 file: ncrcat wrfout_d02_2024-06-13* wrfout_d02_20240613_merged.nc
to extract all times when getting var: timeidx = ALL_TIMES
loop in command line: for i in 'seq 10 12'; do scp 'jmceachern@bluesky2:/nfs/kitsault/archives/forecasts/WAN00CG-01/24$i0700/wrfout_d02_2024-06-$i*' .; done
to make mp4 movie in command line: ffmpeg -framerate 4 -pattern_type glob -i 'forecast202306*.png' -c:v libx264 -pix_fmt yuv420p movie20230606.mp4
"""
# %%
frames = []
for date in testing_dates:
    for time in times:
        img_path = f'/Users/jmceachern/lightning/performance_img/performance{date[0]}{date[1]}{date[2]}{time}.png'
        frames.append(io.imread(img_path))
# %%
io.mimsave(f'/Users/jmceachern/lightning/performance_img/gif{testing_dates[0][0]}{testing_dates[0][1]}{testing_dates[0][2]}.gif',
           frames,loop=0,fps=6)
# %%
# for i,arr in enumerate(frames):
#     if arr.shape[1] != 494:
#         fill = np.zeros((520,494-arr.shape[1],4))+255
#         frames[i] = np.append(arr,fill,1)
# %%
t = getvar(wrf_ds,'T2')
shape = t.shape
cla_RanFor = np.reshape(model_dict['RandomForestClassifier'].predict_proba(X_test)[:,1],shape)
# %%
n = 3
smoothed = []
for i in range(cla_RanFor.shape[0]): #  i = row
    i_min = max(0, i - n) # chooses i - n unless near the bound
    i_max = min(cla_RanFor.shape[0], i + n +1) # chooses i + n unless near the bound
    for ii in range(cla_RanFor.shape[1]): # ii = column
        ii_min = max(0, ii - n)
        ii_max = min(cla_RanFor.shape[1], ii + n + 1)

        window_num = (i_max-i_min)*(ii_max-ii_min) # number of obs in the window
        smoothed.append((cla_RanFor[i_min:i_max,ii_min:ii_max]).sum() / window_num)
smoothed = np.reshape(smoothed,shape)
# %%
lats, lons = latlon_coords(t)
bm = get_basemap(t)
x, y = bm(to_np(lons), to_np(lats))

fig = plt.figure(figsize = (30,30))

ax1 = fig.add_subplot(221)
bm.drawcoastlines(linewidth=0.25)
bm.drawstates(linewidth=0.25)
bm.drawcountries(linewidth=0.25)
levels = np.array([np.percentile(cla_RanFor,50),np.percentile(cla_RanFor,80),np.percentile(cla_RanFor,90),np.percentile(cla_RanFor,95),np.percentile(cla_RanFor,98),np.percentile(cla_RanFor,99),np.max(cla_RanFor)])
norm = matplotlib.colors.BoundaryNorm(levels,len(levels))
colors = list(plt.cm.BuPu(np.linspace(0,1,len(levels)-1)))
colors[0] = "w"
# colors[1] = "dodgerblue"
# colors[2] = "mediumseagreen"
# colors[3] = "seagreen"
# colors[4] = "greenyellow"
# colors[5] = "yellow"
# colors[6] = "orange"
# colors[7] = "red"
# colors[8] = "magenta"
# colors[9] = "blueviolet"
contours = bm.contourf(x, y, cla_RanFor, levels=levels, colors="dodgerblue")
cmap = matplotlib.colors.ListedColormap(colors,"", len(colors))
# cmap = matplotlib.colors.ListedColormap(colors[::-1],"", len(colors))
im = bm.contourf(x, y, to_np(cla_RanFor), levels, cmap=cmap, norm=norm)
cax = ax1.imshow(cla_RanFor, cmap=cmap)
bar = plt.colorbar(cax, shrink=0.2, spacing='uniform', ticks=levels, boundaries=levels)
bar.set_ticklabels(['50', '80', '90', '95', '98', '99', '100'])
# ax1.set_title(f'Probablistic Lightning Forecast \ninit: 00z July 10 2024                                     valid at: {date[0]}/{date[1]}/{date[2]} {tt}:00:00 UTC',fontsize=20)
ax1.set_xlabel('percentile',fontsize=20)
# %%
# For each model trained on a different amount of data (1,3,5,7,10,12,14,20,28)
train_dates7 = [('2024','06','05'),('2024','06','06'),('2024','06','07'),('2024','06','09'),('2024','06','10'),('2024','06','11'),('2024','06','12'),('2024','06','13'),('2024','06','14'),('2024','06','15'),('2024','06','16'),('2024','06','18'),('2024','06','19'),('2024','06','20'),('2024','06','21'),('2024','07','14'),('2024','07','15'),('2024','07','16'),('2024','07','17'),('2024','07','19')]
train_dates8 = [('2024','06','05'),('2024','06','06'),('2024','06','07'),('2024','06','09'),('2024','06','10'),('2024','06','11'),('2024','06','12'),('2024','06','13'),('2024','06','14'),('2024','06','15'),('2024','06','16'),('2024','06','18'),('2024','06','19'),('2024','06','20'),('2024','06','21'),('2024','07','14'),('2024','07','15'),('2024','07','16'),('2024','07','17'),('2024','07','19'),('2024','07','20'),('2024','07','21'),('2024','07','22'),('2024','07','23'),('2024','07','24'),('2024','07','25'),('2024','07','26'),('2024','07','27')]
train_date_lis = [train_dates7,train_dates8]
i = 7

for training_dates in train_date_lis:
    y_train_all, X_train_all = retreive_X_y(training_dates) # retreive and concatinate the preprocessed training data
    model_dict = make_models(i,X_train_all,y_train_all) # make the model with the given amount of training data and save to experimental_models directory
    i += 1

# %%
# calculate the average CSI over one day (06/06/2023)
key = 'RandomForestClassifier' # the classifier model that will be combined with the regression
wrf_ds = get_wrf('wrfout_d02_20240605_merged.nc') # this is just to reconstruct the flat arrays.
date = ('2023','06','06')
key = 'RandomForestClassifier'

CSI_dict = {'ave':[]}
i_lis = np.arange(0,7)
for i in i_lis:
    print(f'model {i}')
    model_dict = retreive_model(i)
    CSI_dict[f'CSI_{i}'] = []
    for tt in times:
        y_test, X_test = retreive_X_y_test(date,tt)
        decision_dict = make_decision(wrf_ds, model_dict, key, X_test, y_test)
        accuracy, POD, POFD, SR, CSI, bias, continge_dict = contingency_table(y_test,decision_dict[key])
        CSI_dict[f'CSI_{i}'].append(CSI)

    CSI_dict['ave'].append(np.mean(np.array(CSI_dict[f'CSI_{i}'])[-4:-1]))
# %%
train_length = [1,3,5,7,10,12,14]
plt.plot(train_length, CSI_dict['ave'])
plt.ylim(0.05,0.1)
plt.xlabel('Amount of training data (days)')
plt.ylabel('Average CSI')
# %%

