###########################################################################################
# Libraries
###########################################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, confusion_matrix, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from models import denseOutput
from tensorflow import keras
from keras.utils import np_utils
import tensorflow as tf
from my_plotsMixed import plot_trainingProcess
from numpy.random import seed
from sklearn import preprocessing
seed(1)


########################################################################################################################
# Global parameters
########################################################################################################################

wd = os.getcwd()

########################################################################################################################
# Main
########################################################################################################################
def main():

    # Labels and partitions for each cv fold (dictionaries)
    partition_suffix = 'myPartitionfileName'
    partition_path   = ''
    label_path       = ''

    cv    = [1,2,3,4]   # CV folds to be analysed
    runs  = [0,1,2,3]   # Repeat runs for each CV fold
    patch = 91          # 3D image patch ID
    parameterID     = [84, 136, 80] # Hyperparameter IDs for TDA 2DCNN
    parameterID3D   = 41# Hyperparameter patch-based 3DCNN
    parameterIDtail = 1# Hyperparameter for combining patch-based 3DCNN and TDA 2DCNN


    # Folders from which the output will be used

    # Ensemble model 1
    outputForEnsemble1 = ''
    savetoEnsemble1    = ''

    # Ensemble model 2
    outputForEnsemble2 = ''
    loadFeaturesFrom_TDA = ''
    loadFeaturesFrom_3D  = ''

    # Evaluation of all models
    outputfolderEvalEnsembles  = ''
    saveToEvalEnsembles  = ''

    output3D_CNN = loadFeaturesFrom_3D+ '/run_overviews'
    outputEnsem1 = savetoEnsemble1 + '/run_overviews'

    ensemble1(cv, runs, outputForEnsemble1, partition_path, label_path, partition_suffix, savetoEnsemble1)

    ensemble2(cv, runs, patch,partition_path, label_path,partition_suffix, outputForEnsemble2,
              loadFeaturesFrom_TDA, loadFeaturesFrom_3D, normalize=True, parameterID=parameterID,
              parameterID3D=parameterID3D, parameterIDtail=parameterIDtail)
    evalEnsembles(cv, runs, patch, outputfolderEvalEnsembles, saveToEvalEnsembles, output3D_CNN, outputEnsem1)

    return 0

########################################################################################################################
# Subfucntions
########################################################################################################################

def ensemble1(cv, runs, output,partition_path,label_path,partition_suffix,saveto):

    # Fixed parameters
    name = 'None_CNNflexi_dense_test'
    patches = list(np.arange(0, 216))  # list(2+6*np.arange(36))+list(3+6*np.arange(36))
    n_patches = 216
    ensembleModel = 'LogisticRegression'

    # loop over all cv folds
    for c in range(0, len(cv)):

        print('Working on fold ' + str(c))


        # Load labels
        partition_file = partition_path + partition_suffix+ str(cv[c]) + '.npy'
        label_file = label_path + partition_suffix+ str(cv[c]) + '.npy'
        partition = np.load(partition_file, allow_pickle='TRUE').item()
        labels = np.load(label_file, allow_pickle='TRUE').item()

        # Loop over all runs
        for r in range(0, len(runs)):

            X_val, y_val = predictAllPatchesPerRun(partition, 'validation', labels, output, name, [cv[c]], [r],
                                                   patches, n_patches, False)
            X_trn, y_trn = predictAllPatchesPerRun(partition, 'train', labels, output, name, [cv[c]], [runs[r]],
                                                   patches, n_patches, True)


            # Ensemble Prediction
            bestacc_val = np.max(X_val[0, -216:])
            print('Best validation acc: ' + str(bestacc_val))

            X_trn_df = pd.DataFrame(X_trn[:, :216])
            X_val_df = pd.DataFrame(X_val[:, :216])


            # Run logRegression
            y_val, y_pred_class, acc, auc_val, aps, recall, F1, precision, tn, tp, fp, fn \
                = run_LR(X_trn_df, y_trn, X_val_df, y_val, saveto, cv[c], runs[r])


            # Create data frame
            data_df = {'runID': str(r),
                       'model': ensembleModel,
                       'n_epochs': 0,
                       'tn': tn,
                       'fp': fp,
                       'fn': fn,
                       'tp': tp,
                       'acc': acc,
                       'precision': precision,
                       'recall': recall,
                       'auc': auc_val,
                       'aps': aps,
                       }
            df = pd.DataFrame(data=data_df, index=[r])
            df.to_csv(os.path.join(saveto, ensembleModel + str(91) + '_run' + str(runs[r]) + 'CV_' + str(
                cv[c]) + '_results_overview.csv'))

    return 0

def ensemble2(cv,runs, patch, partition_path, label_path,partition_suffix,  outputfolder,
              loadFeaturesFrom_TDA, loadFeaturesFrom_3D, normalize=True, parameterID = [84, 136, 80],
              parameterID3D = 41, parameterIDtail = 1):

    tf.random.set_seed(1)

    ensembleModel   = 'Dense' #'RandomForest'#'LogisticRegression'
    output = outputfolder + '/figures'

    print('CV: ',str(cv))
    print('run: ', str(runs))

    # Loop over all cv folds
    for c in range(0,len(cv)):

        print('Working on fold '+str(c))

        # Load labels
        partition_file = partition_path + partition_suffix + str(cv[c])+ '.npy'
        label_file = label_path + partition_suffix+ str(cv[c]) + '.npy'
        partition  = np.load(partition_file, allow_pickle='TRUE').item()
        labels     = np.load(label_file, allow_pickle='TRUE').item()
        pats       = np.array(partition['validation'])

        # Get labels for this cv fold
        y_trn_in =  [labels[p] for p in partition['train']]
        y_val_in = [labels[p] for p in partition['validation']]

        # Loop over all repeat runs per fold
        for r in runs:

            # Load TDA features
            for j in range(0,len(parameterID)):
                filepath_enc_trn = os.path.join(os.path.join(loadFeaturesFrom_TDA, 'run_hdf5'),
                                                'encoding_trn_' + 'PI_CNN_model_None_dense' + '_patch' + str(patch) +
                                                '_' + str(r) + 'CV_' + str(cv[c]) + '_pID' + str(parameterID[j]) +
                                                'Dim' + str(j) + '_nnet_run_test.npy')
                filepath_enc_val = os.path.join(os.path.join(loadFeaturesFrom_TDA, 'run_hdf5'),
                                                'encoding_val_' + 'PI_CNN_model_None_dense' + '_patch' + str(patch) +
                                                '_' + str(r) + 'CV_' + str(cv[c]) + '_pID' + str(parameterID[j]) +
                                                'Dim' + str(j) + '_nnet_run_test.npy')
                with open(filepath_enc_trn, 'rb') as f:
                    X_trn_TDA_dim = np.load(f, allow_pickle=True)
                with open(filepath_enc_val, 'rb') as f:
                    X_val_TDA_dim = np.load(f, allow_pickle=True)

                if j == 0:
                    X_trn_TDA = X_trn_TDA_dim
                    X_val_TDA = X_val_TDA_dim
                else:
                    X_trn_TDA = np.concatenate((X_trn_TDA,X_trn_TDA_dim),axis=1)
                    X_val_TDA = np.concatenate((X_val_TDA,X_val_TDA_dim),axis=1)

            # Create TDA Dataframe:
            X_trn_df_TDA = pd.DataFrame(X_trn_TDA)
            X_val_df_TDA = pd.DataFrame(X_val_TDA)


            # Optional: Normalize
            if normalize:
                scalerTDA = preprocessing.StandardScaler().fit(X_trn_df_TDA)
                X_trn_df_TDA_norm = pd.DataFrame(scalerTDA.transform(X_trn_df_TDA))
                X_val_df_TDA_norm = pd.DataFrame(scalerTDA.transform(X_val_df_TDA))


                # Run ensemble model on TDA
                y_valTDA, y_pred_classTDA, accTDA, auc_valTDA, apsTDA, recallTDA,F1TDA,precisionTDA, tnTDA, tpTDA,fpTDA,fnTDA \
                        = run_model(X_trn_df_TDA_norm, y_trn_in, X_val_df_TDA_norm, y_val_in, ensembleModel,output, cv[c],r,'TDA')
            else:

                # Run ensemble model on TDA
                y_valTDA, y_pred_classTDA, accTDA, auc_valTDA, apsTDA, recallTDA, F1TDA, precisionTDA, tnTDA, tpTDA, fpTDA, fnTDA \
                    = run_model(X_trn_df_TDA, y_trn_in, X_val_df_TDA, y_val_in, ensembleModel, output, cv[c], r, 'TDA')


            # Load 3D features
            filepath_enc_trn = os.path.join(os.path.join(loadFeaturesFrom_3D, 'run_hdf5'),
                                            'encoding_trn_' + 'None_CNNflexi_dense' + '_patch' + str(patch) + '_run' + str(r) + 'CV_' + str(
                                                cv[c]) + '_pIDtda' +str(parameterID)+'_pID3D'+ str(parameterID3D)+'_pIDco'+ str(parameterIDtail) + '_tdadim[0, 1, 2].npy')
            filepath_enc_val = os.path.join(os.path.join(loadFeaturesFrom_3D, 'run_hdf5'),
                                            'encoding_val_' + 'None_CNNflexi_dense' + '_patch' + str(patch) + '_run' + str(r) + 'CV_' + str(
                                                cv[c]) + '_pIDtda' +str(parameterID)+'_pID3D'+ str(parameterID3D)+'_pIDco'+ str(parameterIDtail) + '_tdadim[0, 1, 2].npy')
            with open(filepath_enc_trn, 'rb') as f:
                X_trn_3D = np.load(f, allow_pickle=True)
            with open(filepath_enc_val, 'rb') as f:
                X_val_3D = np.load(f, allow_pickle=True)

            # Create data frame
            X_trn_df_3D = pd.DataFrame(X_trn_3D)  # sel_patches])#
            X_val_df_3D = pd.DataFrame(X_val_3D)

            if normalize:

                # Normalize
                scaler3D = preprocessing.StandardScaler().fit(X_trn_df_3D)
                X_trn_df_3D_norm = pd.DataFrame(scaler3D.transform(X_trn_df_3D))
                X_val_df_3D_norm = pd.DataFrame(scaler3D.transform(X_val_df_3D))

                # Run ensemble model on 3D
                y_val3D, y_pred_class3D, acc3D, auc_val3D, aps3D, recall3D, F13D, precision3D, tn3D, tp3D, fp3D, fn3D \
                    = run_model(X_trn_df_3D_norm, y_trn_in, X_val_df_3D_norm, y_val_in, ensembleModel, output, cv[c], r, '3D')
            else:

                # Run ensemble model on 3D
                y_val3D, y_pred_class3D, acc3D, auc_val3D, aps3D, recall3D, F13D, precision3D, tn3D, tp3D, fp3D, fn3D \
                    = run_model(X_trn_df_3D, y_trn_in, X_val_df_3D, y_val_in, ensembleModel, output,
                                cv[c], r, '3D')


            # Create combined Dataframe:
            if normalize:
                X_trn_df_norm = pd.concat([X_trn_df_3D_norm, X_trn_df_TDA_norm],axis = 1)
                X_val_df_norm = pd.concat([X_val_df_3D_norm, X_val_df_TDA_norm],axis = 1)


                # Run ensemble model on combination
                y_val, y_pred_class, acc, auc_val, aps, recall,F1,precision, tn, tp,fp,fn \
                        = run_model(X_trn_df_norm, y_trn_in, X_val_df_norm, y_val_in, ensembleModel,output, cv[c],r,'Combi')
            else:
                # Not normalized
                X_trn_df = pd.DataFrame(np.concatenate((X_trn_3D,X_trn_TDA),axis = 1))  # sel_patches])#
                X_val_df = pd.DataFrame(np.concatenate((X_val_3D,X_val_TDA),axis = 1))
                y_val, y_pred_class, acc, auc_val, aps, recall, F1, precision, tn, tp, fp, fn \
                    = run_model(X_trn_df, y_trn_in, X_val_df, y_val_in, ensembleModel, output, cv[c], r,'Combi')


            # Create dataframes to save results
            # TDA only results
            data_dfTDA = {'runID': str(runs),
                       'model': ensembleModel,
                       'n_epochs': 0,
                       'tn': tnTDA,
                       'fp': fpTDA,
                       'fn': fnTDA,
                       'tp': tpTDA,
                       'acc': accTDA,
                       'precision': precisionTDA,
                       'recall': recallTDA,
                       'auc': auc_valTDA,
                       'aps': apsTDA,
                       }
            dfTDA = pd.DataFrame(data=data_dfTDA, index = [cv[c]])
            dfTDA.to_csv(os.path.join(output, ensembleModel +'TDA'+str(patch)+ '_run' + str(r) + 'CV_' + str(cv[c]) + '_results_overview.csv'))

            # Combination results
            data_df = {'runID': str(runs),
                       'model': ensembleModel,
                       'n_epochs': 0,
                       'tn': tn,
                       'fp': fp,
                       'fn': fn,
                       'tp': tp,
                       'acc': acc,
                       'precision': precision,
                       'recall': recall,
                       'auc': auc_val,
                       'aps': aps,
                       }
            df = pd.DataFrame(data=data_df, index=[cv[c]])
            df.to_csv(os.path.join(output, ensembleModel + 'Combi' + str(patch) + '_run' + str(r) + 'CV_' + str(
                cv[c]) + '_results_overview.csv'))

            # 3D only results
            data_df3D = {'runID': str(runs),
                       'model': ensembleModel,
                       'n_epochs': 0,
                       'tn': tn3D,
                       'fp': fp3D,
                       'fn': fn3D,
                       'tp': tp3D,
                       'acc': acc3D,
                       'precision': precision3D,
                       'recall': recall3D,
                       'auc': auc_val3D,
                       'aps': aps3D,
                       }
            df3D = pd.DataFrame(data=data_df3D, index=[cv[c]])
            df3D.to_csv(os.path.join(output, ensembleModel + '3D' + str(patch) + '_run' + str(r) + 'CV_' + str(
                cv[c]) + '_results_overview.csv'))

    return 0

def evalEnsembles(cv, runs, patch, outputfolder, saveTo, output3D_CNN, outputEnsem1):

    ensembleModel = 'Dense'  # 'RandomForest'#'LogisticRegression'

    output = outputfolder + '/figures'
    for c in range(0, len(cv)):

        print('Working on fold ' + str(c))
        for r in runs:

            # load dataframes
            dfTDA_LR   = pd.read_csv(os.path.join(output, ensembleModel + 'TDA' + str(patch) + '_run' + str(r) +
                                                'CV_' + str(cv[c]) + '_results_overview.csv'))
            dfCombi_LR = pd.read_csv(os.path.join(output, ensembleModel + 'Combi' + str(patch) + '_run' + str(r) +
                                                'CV_' + str(cv[c]) + '_results_overview.csv'))
            df3D_LR    = pd.read_csv(os.path.join(output, ensembleModel + '3D' + str(patch) + '_run' + str(r) +
                                                'CV_' + str(cv[c]) + '_results_overview.csv'))
            df3D_CNN   = pd.read_csv(os.path.join(output3D_CNN, 'None_CNNflexi_dense_test' + str(patch) + '_run' + str(r)
                                                + '_CV' + str(cv[c]) + '_results_overview.csv'))
            dfEns_LR   = pd.read_csv(os.path.join(outputEnsem1, 'LogisticRegression' + str(patch) + '_run' + str(r) +
                                                  'CV_' + str(cv[c]) + '_results_overview.csv'))

            # collect
            if r == runs[0]:
                dfTDA_LR_run   = dfTDA_LR
                dfCombi_LR_run = dfCombi_LR
                df3D_CNN_run   = df3D_CNN
                df3D_LR_run    = df3D_LR
                dfEns_LR_run   = dfEns_LR
            else:
                dfTDA_LR_run   = pd.concat([dfTDA_LR_run, dfTDA_LR], axis=0)
                dfCombi_LR_run = pd.concat([dfCombi_LR_run, dfCombi_LR], axis=0)
                df3D_CNN_run   = pd.concat([df3D_CNN_run, df3D_CNN], axis=0)
                df3D_LR_run    = pd.concat([df3D_LR_run, df3D_LR], axis=0)
                dfEns_LR_run   = pd.concat([dfEns_LR_run, dfEns_LR], axis=0)

        # Average over runs
        av_dfTDA_LR_run   = dfTDA_LR_run.mean()
        av_dfCombi_LR_run = dfCombi_LR_run.mean()
        av_df3D_CNN_run   = df3D_CNN_run.mean()
        av_df3D_LR_run    = df3D_LR_run.mean()
        av_dfEns_LR_run   = dfEns_LR_run.mean()
        print(df3D_LR_run)

        # Collect for CV
        if c == 0:
            dfTDA_LR_cv   = av_dfTDA_LR_run
            dfCombi_LR_cv = av_dfCombi_LR_run
            df3D_CNN_cv   = av_df3D_CNN_run
            df3D_LR_cv    = av_df3D_LR_run
            dfEns_LR_cv   = av_dfEns_LR_run
        else:
            dfTDA_LR_cv   = pd.concat([dfTDA_LR_cv, av_dfTDA_LR_run], axis=1)
            dfCombi_LR_cv = pd.concat([dfCombi_LR_cv, av_dfCombi_LR_run], axis=1)
            df3D_CNN_cv   = pd.concat([df3D_CNN_cv, av_df3D_CNN_run], axis=1)
            df3D_LR_cv    = pd.concat([df3D_LR_cv, av_df3D_LR_run], axis=1)
            dfEns_LR_cv   = pd.concat([dfEns_LR_cv, av_dfEns_LR_run], axis=1)

    # Average over all CV folds
    if len(cv) > 1:
        av_dfTDA_LR_cv   = dfTDA_LR_cv.mean(axis=1)
        av_dfCombi_LR_cv = dfCombi_LR_cv.mean(axis=1)
        av_df3D_CNN_cv   = df3D_CNN_cv.mean(axis=1)
        av_df3D_LR_cv    = df3D_LR_cv.mean(axis=1)
        av_dfEns_LR_cv   = dfEns_LR_cv.mean(axis=1)

        std_dfTDA_LR_cv  = dfTDA_LR_cv.std(axis=1)
        std_dfCombi_LR_cv= dfCombi_LR_cv.std(axis=1)
        std_df3D_CNN_cv  = df3D_CNN_cv.std(axis=1)
        std_df3D_LR_cv   = df3D_LR_cv.std(axis=1)
        std_dfEns_LR_cv  = dfEns_LR_cv.std(axis=1)
    else:
        av_dfTDA_LR_cv   = dfTDA_LR_cv
        av_dfCombi_LR_cv = dfCombi_LR_cv
        av_df3D_CNN_cv   = df3D_CNN_cv
        av_df3D_LR_cv    = df3D_LR_cv
        av_dfEns_LR_cv   = dfEns_LR_cv

        std_dfTDA_LR_cv  = dfTDA_LR_cv
        std_dfCombi_LR_cv= dfCombi_LR_cv
        std_df3D_CNN_cv  = df3D_CNN_cv
        std_df3D_LR_cv   = df3D_LR_cv
        std_dfEns_LR_cv  = dfEns_LR_cv

    # Save to data frame
    rows_names = ['ensemble 2 TDA', 'ensemble 2 combi','3D CNN P*','ensemble 2 3D','av_dfEns1_LR_run']

    # Averages
    df_acc = saveResults('acc', rows_names, av_dfTDA_LR_cv, av_dfCombi_LR_cv,av_df3D_CNN_cv, av_df3D_LR_cv, av_dfEns_LR_cv)
    df_auc = saveResults('auc', rows_names, av_dfTDA_LR_cv, av_dfCombi_LR_cv,av_df3D_CNN_cv, av_df3D_LR_cv, av_dfEns_LR_cv)
    df_aps = saveResults('aps', rows_names, av_dfTDA_LR_cv, av_dfCombi_LR_cv,av_df3D_CNN_cv, av_df3D_LR_cv, av_dfEns_LR_cv)
    df_rec = saveResults('recall', rows_names, av_dfTDA_LR_cv, av_dfCombi_LR_cv,av_df3D_CNN_cv, av_df3D_LR_cv, av_dfEns_LR_cv)
    df_pre = saveResults('precision', rows_names, av_dfTDA_LR_cv,av_dfCombi_LR_cv,av_df3D_CNN_cv, av_df3D_LR_cv, av_dfEns_LR_cv)
    df_aps.to_csv(os.path.join(saveTo, ensembleModel + '_cv' + str(cv) + '_runs' + str(runs) + 'APS_allPI.csv'))
    df_acc.to_csv(os.path.join(saveTo, ensembleModel + '_cv' + str(cv) + '_runs' + str(runs) + 'Acc_allPI.csv'))
    df_auc.to_csv(os.path.join(saveTo, ensembleModel + '_cv' + str(cv) + '_runs' + str(runs) + 'Auc_allPI.csv'))
    df_rec.to_csv(os.path.join(saveTo, ensembleModel + '_cv' + str(cv) + '_runs' + str(runs) + 'Rec_allPI.csv'))
    df_pre.to_csv(os.path.join(saveTo, ensembleModel + '_cv' + str(cv) + '_runs' + str(runs) + 'Pre_allPI.csv'))

    # Standard deviations
    df_dacc = saveResults('acc', rows_names, std_dfTDA_LR_cv, std_dfCombi_LR_cv,std_df3D_CNN_cv, std_df3D_LR_cv, std_dfEns_LR_cv)
    df_dauc = saveResults('auc', rows_names, std_dfTDA_LR_cv, std_dfCombi_LR_cv,std_df3D_CNN_cv, std_df3D_LR_cv, std_dfEns_LR_cv)
    df_daps = saveResults('aps', rows_names, std_dfTDA_LR_cv, std_dfCombi_LR_cv,std_df3D_CNN_cv, std_df3D_LR_cv, std_dfEns_LR_cv)
    df_drec = saveResults('recall', rows_names, av_dfTDA_LR_cv,std_dfCombi_LR_cv,std_df3D_CNN_cv, std_df3D_LR_cv, std_dfEns_LR_cv)
    df_dpre = saveResults('precision', rows_names, av_dfTDA_LR_cv,std_dfCombi_LR_cv,std_df3D_CNN_cv, std_df3D_LR_cv, std_dfEns_LR_cv)
    df_daps.to_csv(os.path.join(saveTo, ensembleModel + '_cv' + str(cv) + '_runs' + str(runs) + '_stdAPS_allPI.csv'))
    df_dacc.to_csv(os.path.join(saveTo, ensembleModel + '_cv' + str(cv) + '_runs' + str(runs) + '_stdAcc_allPI.csv'))
    df_dauc.to_csv(os.path.join(saveTo, ensembleModel + '_cv' + str(cv) + '_runs' + str(runs) + '_stdAuc_allPI.csv'))
    df_drec.to_csv(os.path.join(saveTo, ensembleModel + '_cv' + str(cv) + '_runs' + str(runs) + '_stdRec_allPI.csv'))
    df_dpre.to_csv(os.path.join(saveTo, ensembleModel + '_cv' + str(cv) + '_runs' + str(runs) + '_stdPre_allPI.csv'))

    return 0

def run_LR(X_train, y_train, X_test, y_test, clf_choice, mydir, cv,r):

    # Use logistic regression
    clf = LogisticRegression(max_iter=10000, class_weight='balanced', solver='liblinear')
    param_grid = {'C': [0.001, 0.005, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], 'penalty': ['l1', 'l2']}
    randomizer = False


    ####Get classifier:
    print('Getting classifier')
    classifier = get_classifier(X_train, y_train, clf, param_grid, randomizer)


    #### Print coefficients (weights):
    for coef, name in zip(classifier.best_estimator_.coef_.ravel(), X_train.columns):
        print(name, coef)

    ####
    # Evaluate validation performance
    plot_title = 'Evaluation of ' + clf_choice + ' model_CV'+str(cv)+'_run'+ str(r)
    y_val, y_pred_class, acc, auc_val, aps, recall,F1,precision,tn, tp,fp,fn  = \
        evalBinaryClassifier_SimpleModel(classifier.best_estimator_, X_test, y_test, mydir,
                                          plot_title=plot_title + '_test', doPlots=True)

    return y_val, y_pred_class, acc, auc_val, aps, recall,F1,precision,tn, tp,fp,fn

def run_model(X_train, y_train, X_test, y_test, clf_choice, mydir, cv,r,name_suf):

    ####Get classifier:
    print('Getting classifier')

    # Optimize l1 regularization
    acc = 0
    aps = 0
    l1 = [0,0.1,0.001,0.0001,0.00001]
    shape = X_test.values.shape[1]
    for l1_i in l1:

        # Simple single layer serves as model
        model = denseOutput(l1_i,shape)

        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.Adam(lr=0.0001),
                      metrics=None)


        # Train with early stopping
        filepath = os.path.join(os.path.join(mydir,'Dense', 'run_hdf5'), 'Dense' +
                                '_' + str(r) + 'CV_' + str(cv) + '_l1'+str(l1_i) + name_suf+'_nnet_run_test.hdf5')
        check = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True,
                                                   mode='auto')
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,
                                                         mode='auto')

        # Make sure labels are categorical
        try:
            y_train.shape
            y_trn = y_train
        except:
            y_trn = np_utils.to_categorical(y_train, 2)
        try:
            y_test.shape
            y_val = y_test
        except:
            y_val = np_utils.to_categorical(y_test, 2)

        val_data = (X_test.values, y_val)
        val_data_X = X_test.values

        # Fit for this l1 parameter
        history = model.fit(x=X_train.values, y=y_trn,
                            batch_size=20,
                            epochs=500,
                            verbose=1,
                            callbacks=[check, earlyStopping],
                            validation_data=val_data)

        # Load the best weights
        model.load_weights(filepath)

        # Save
        name = 'Dense' + '_run' + str(r) + 'CV_'+ str(cv)+ '_l1'+str(l1_i) +name_suf
        plot_trainingProcess(history, os.path.join(mydir,'Dense', 'run_losses'), name, 0)


        # Evaluate validation performance
        plot_title = 'Evaluation of ' + clf_choice + ' model_CV' + str(cv) + '_run' + str(r) + name_suf


        tn_out, fp_out, fn_out, tp_out, acc_out, precision_out, recall_out, roc_auc_out, aps_out, dist_out,\
        meanDist_out, stdDist_out, thresh_opt_out, y_pred_out, y_pred_class_out \
        = evalBinaryClassifierCNN(model, val_data_X, y_val, batchsize = 20)

        # Clear keras
        tf.keras.backend.clear_session()

        # Update results
        if acc_out+aps_out>acc+aps:

            print('L1 '+ str(l1_i) + ' superior')
            tn = tn_out
            fp = fp_out
            fn = fn_out
            tp = tp_out
            acc = acc_out
            precision = precision_out
            recall = recall_out
            roc_auc = roc_auc_out
            aps = aps_out
            y_pred_class = y_pred_class_out
            F1 = 2 * (precision * recall) / (precision + recall)


    return y_val, y_pred_class, acc, roc_auc, aps, recall,F1,precision,tn, tp,fp,fn

def evalBinaryClassifierCNN(model, x,y, batchsize = 20):
    '''
    Visualize the performance of a binary Classifier.

    Displays a labelled Confusion Matrix, distributions of the predicted
    probabilities for both classes, the Recall-Precision curve, the ROC curve, and F1 score of a fitted
    Binary Logistic Classifier. Author: gregcondit.com/articles/logr-charts

    Parameters
    ----------
    model : fitted scikit-learn model with predict_proba & predict methods
        and classes_ attribute. Typically LogisticRegression or
        LogisticRegressionCV

    x : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples
        in the data to be tested, and n_features is the number of features

    y : array-like, shape (n_samples,)
        Target vector relative to x.

    batchsize: int, optional
        batchsize used for model training


    Returns
    ----------
    tn:        float; True Negatives
    fp:        float; True Positives
    fn:        float; False Negatives
    tp:        float; True Positives
    acc:       float; Accuracy
    precision: float; Precision
    recall:    float; Recall
    F1:        float; F1 score
    roc_auc:   float; Receiver-Operator area under the curve
    aps:       float; Average Precision score
    y_pred_class: array; predicted class per subject
    y_val:        array; correct class per subject
    '''

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(x, y, batch_size=batchsize)
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    y_pred = model.predict(x)[:,1]


    # Optimize the threshold:
    F1_opt = 0
    thresh_opt = 0
    for t in np.arange(0,1,0.01):
        y_pred_class  = np.where(y_pred > t, 1,0)
        cm = confusion_matrix(y[:, 1], y_pred_class)
        tn, fp, fn, tp = [i for i in cm.ravel()]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = 2 * (precision * recall) / (precision + recall)
        if F1>F1_opt:
            thresh_opt = t
            F1_opt = F1

    # Class allocation
    y_pred_class = np.where(y_pred > thresh_opt, 1, 0)

    # Get mean realtive distance to descision boundary
    dist = []
    for this_y in list(y_pred):
        if this_y-thresh_opt>0:
            distance = (this_y-thresh_opt)/(1-thresh_opt)
        else:
            distance = -(this_y - thresh_opt) / (thresh_opt)
        dist.append(distance)
    meanDist = np.mean(dist)
    stdDist = np.std(dist)

    # 1 -- Confusion matrix
    cm = confusion_matrix(y[:,1], y_pred_class)

    # 2 -- Distributions of Predicted Probabilities of both classes
    df = pd.DataFrame({'probPos': np.squeeze(y_pred), 'target': y[:,1]})

    # 3 -- PRC:
    precision, recall, _ = precision_recall_curve(y[:,1], y_pred, pos_label=1)
    aps = average_precision_score(y[:,1], y_pred)

    # 4 -- ROC curve with annotated decision point
    fp_rates, tp_rates, _ = roc_curve(y[:,1], y_pred, pos_label=1)
    roc_auc = auc(fp_rates, tp_rates)
    tn, fp, fn, tp = [i for i in cm.ravel()]

    # 5 -- F1 score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * (precision * recall) / (precision + recall)

    # 6 -- Accuracy:
    acc = (tp + tn) / (tp + tn + fp + fn)

    # Print
    printout = (
        f'Precision: {round(precision, 2)} | '
        f'Recall: {round(recall, 2)} | '
        f'F1 Score: {round(F1, 2)} | '
        f'Accuracy: {round(acc, 2)} | '
    )
    print(printout)

    return tn, fp, fn, tp,acc, precision,recall,roc_auc,aps, dist,meanDist,stdDist, thresh_opt, y_pred, y_pred_class

def evalBinaryClassifier_SimpleModel(model, x,y, mydir, labels=['Positives', 'Negatives'],
                         plot_title='Evaluation of model', doPlots=False,filename = ''):
    '''
    Visualize the performance of  a Logistic Regression Binary Classifier.
    '''



    # Generate predictions (probabilities -- the output of the last layer)
    y_pred = model.predict_proba(x)[:,1]


    # Optimize the threshold:
    F1_opt = 0
    thresh_opt = 0
    for t in np.arange(0,1,0.001):
        y_pred_class  = np.where(y_pred > t, 1,0)
        cm = confusion_matrix(y, y_pred_class)
        tn, fp, fn, tp = [i for i in cm.ravel()]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = 2 * (precision * recall) / (precision + recall)
        if F1>F1_opt:
            thresh_opt = t
            F1_opt = F1

    y_pred_class = np.where(y_pred > thresh_opt, 1, 0)


    # 1 -- Confusion matrix
    cm = confusion_matrix(y, y_pred_class)

    # 2 -- Distributions of Predicted Probabilities of both classes
    df = pd.DataFrame({'probPos': np.squeeze(y_pred), 'target': y})

    # 3 -- PRC:
    #y_score = model.predict_proba(x)[:, pos_ind]
    precision, recall, _ = precision_recall_curve(y, y_pred, pos_label=1)
    aps = average_precision_score(y, y_pred)

    # 4 -- ROC curve with annotated decision point
    fp_rates, tp_rates, _ = roc_curve(y, y_pred, pos_label=1)
    roc_auc = auc(fp_rates, tp_rates)
    tn, fp, fn, tp = [i for i in cm.ravel()]

    # FIGURE
    if doPlots:
        fig = plt.figure(figsize=[20, 5])
        fig.suptitle(plot_title, fontsize=16)

        # 1 -- Confusion matrix
        plt.subplot(141)
        ax = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False,
                         annot_kws={"size": 14}, fmt='g')
        cmlabels = ['True Negatives', 'False Positives',
                    'False Negatives', 'True Positives']
        for i, t in enumerate(ax.texts):
            t.set_text(t.get_text() + "\n" + cmlabels[i])
        plt.title('Confusion Matrix', size=15)
        plt.xlabel('Predicted Values', size=13)
        plt.ylabel('True Values', size=13)

        # 2 -- Distributions of Predicted Probabilities of both classes
        plt.subplot(142)
        plt.hist(df[df.target == 1].probPos, density=True, bins=25,
                 alpha=.5, color='green', label=labels[0])
        plt.hist(df[df.target == 0].probPos, density=True, bins=25,
                 alpha=.5, color='red', label=labels[1])
        plt.axvline(thresh_opt, color='blue', linestyle='--', label='Boundary')
        plt.xlim([0, 1])
        plt.title('Distributions of Predictions', size=15)
        plt.xlabel('Positive Probability (predicted)', size=13)
        plt.ylabel('Samples (normalized scale)', size=13)
        plt.legend(loc="upper right")

        # 3 -- PRC:
        plt.subplot(143)
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.title('Recall-Precision Curve', size=15)
        plt.text(0.1, 0.3, f'AURPC = {round(aps, 2)}')
        plt.xlabel('Recall', size=13)
        plt.ylabel('Precision', size=13)
        plt.ylim([0.2, 1.05])
        plt.xlim([0.0, 1.0])
        # #plt.show(block=False)

        # 4 -- ROC curve with annotated decision point
        plt.subplot(144)
        plt.plot(fp_rates, tp_rates, color='green',
                 lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')

        # plot current decision point:
        plt.plot(fp / (fp + tn), tp / (tp + fn), 'bo', markersize=8, label='Decision Point')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', size=13)
        plt.ylabel('True Positive Rate', size=13)
        plt.title('ROC Curve', size=15)
        plt.legend(loc="lower right")
        plt.subplots_adjust(wspace=.3)
        #plt.show(block=False)

        filenameuse = os.path.join(mydir, plot_title+filename + '.png')
        plt.savefig(filenameuse)
        plt.close('all')

    # 5 -- F1 score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * (precision * recall) / (precision + recall)

    # 6 -- Accuracy:
    acc = (tp + tn) / (tp + tn + fp + fn)

    # Print
    printout = (
        f'Precision: {round(precision, 2)} | '
        f'Recall: {round(recall, 2)} | '
        f'F1 Score: {round(F1, 2)} | '
        f'Accuracy: {round(acc, 2)} | '
    )
    print(printout)

    return y, y_pred_class, acc, roc_auc, aps, recall,F1,precision,tn, tp,fp,fn

def get_classifier(X_train, y_train, classifier, param_grid, randomizer=False):
    '''
    Create and fit a Logistic Regression model.
    Parameters:
    -----------
    X: data
    y: labels
    classifier: classifier (e.g. LogisticRegression())
    param_grid: Parameter Grid

    Returns:
    --------
    classifier
    X_test
    y_test
    '''
    if randomizer == False:
        classifier = GridSearchCV(classifier, param_grid, cv=5, verbose=1, scoring='roc_auc')  # score=...
    else:
        classifier = RandomizedSearchCV(estimator=classifier, param_distributions=param_grid, n_iter=100, cv=3,
                                        verbose=1, random_state=42, n_jobs=-1,
                                        scoring='average_precision')  # 'roc_auc')

    classifier.fit(X_train, y_train)

    return classifier

def predictAllPatchesPerRun(partition, subset, labels, output, name, cv, runs, patches, n_patches, useTrn):
    if len(cv) > 1:
        raise ('Only one fold at a time for now!')

    # initialize
    rows = [0, 1]
    cols = [0, 1, 2]
    missingPatches = []

    # Loop over all validation patients:
    X = np.empty((len(partition[subset]), 216 * 3))

    # Loop through all csvs and get the relavant information
    map_Dist = np.zeros((6, 6, 6, len(partition[subset])))
    map_stdDist = np.zeros((6, 6, 6, len(partition[subset])))
    map_PredLabel = np.zeros((6, 6, 6, len(partition[subset])))
    map_stdPredLabel = np.zeros((6, 6, 6, len(partition[subset])))
    map_ACC = np.zeros((6, 6, 6))
    map_stdACC = np.zeros((6, 6, 6))

    # Load data for all runs and CV and average:
    mean_absdists = np.empty((n_patches, len(partition[subset]), len(cv)))
    std_absdists = np.empty((n_patches, len(partition[subset]), len(cv)))
    mean_dists = np.empty((n_patches, len(partition[subset]), len(cv)))
    std_dists = np.empty((n_patches, len(partition[subset]), len(cv)))
    mean_cv_dists = np.empty((n_patches, len(partition[subset])))
    std_cv_dists = np.empty((n_patches, len(partition[subset])))
    mean_cv_absdists = np.empty((n_patches, len(partition[subset])))
    std_cv_absdists = np.empty((n_patches, len(partition[subset])))
    bestacc = 0
    std_bestacc = 0
    bestpatch = 0

    for patch in range(0, n_patches):

        print('At step ' + str(patch) + ' of ' + str(len(patches)))

        # Get map location
        i2 = int(patch - np.floor(patch / 36) * 36 - np.floor((patch - np.floor(patch / 36) * 36) / 6) * 6)
        i1 = int(np.floor((patch - np.floor(patch / 36) * 36) / 6))
        i0 = int(np.floor(patch / 36))

        try:
            patches.index(patch)
        except:
            missingPatches.append(patch)
            print('Missing patch ' + str(patch))
            map_Dist[i0, i1, i2, :] = 0
            map_stdDist[i0, i1, i2, :] = 0
            map_PredLabel[i0, i1, i2, :] = 0
            map_stdPredLabel[i0, i1, i2, :] = 0
            map_ACC[i0, i1, i2] = 0
            map_stdACC[i0, i1, i2] = 0
            continue

        # Loop over a single CV fold
        counter = 0
        for c in range(0,len(cv)):
            dists = np.empty((len(runs), len(partition[subset])))
            absdists = np.empty((len(runs), len(partition[subset])))
            accs  = []

            # Loop over the SINGLE run
            for r in range(0,len(runs)):

                name_acc =  name + str(patch) + '_run' + str(runs[r]) + '_CV' + str(cv[c]) + '_results_overview.csv'
                if useTrn:
                    name_useDist = name + str(patch) + '_run' + str(runs[r]) + '_CV' + str(cv[c]) + '_results_dist_trn.csv'
                else:
                    name_useDist = name + str(patch) + '_run' + str(runs[r]) + '_CV' + str(cv[c]) + '_results_dist.csv'

                dist_df = pd.read_csv(os.path.join(output, 'run_distance', name_useDist))
                acc_df = pd.read_csv(os.path.join(output, 'run_overviews', name_acc))


                accs.append(acc_df['acc'].values[0])
                counter = counter + 1


                for patID in range(0, len(partition[subset])):

                    if dist_df['thresh'].values[0] == 0:
                        absdists[r, patID] = 0
                        dists[r, patID] = 0
                    else:
                        absdists[r, patID] = dist_df.loc[patID, 'dist']

                        tmp = (dist_df.loc[patID, 'y_pred'] - dist_df.loc[patID, 'thresh'])
                        thresh = dist_df.loc[patID, 'thresh']
                        if thresh == 0:
                            dist = 0
                        else:
                            if tmp < 0:
                                dist = tmp / thresh
                            else:
                                dist = tmp / (1 - thresh)
                        dists[r, patID] = dist
                        thresh = np.nan

            # Average over runs
            mean_acc = np.nanmean(accs)
            std_acc  = np.nanstd(accs)
            for patID in range(0, len(partition[subset])):
                mean_dists[patch, patID, c] = np.nanmean(dists[:, patID])
                std_dists[patch, patID, c] = np.nanstd(dists[:, patID])
                mean_absdists[patch, patID, c] = np.nanmean(absdists[:, patID])
                std_absdists[patch, patID, c] = np.nanstd(absdists[:, patID])

        if counter == 0:
            missingPatches.append(patch)
            print('Missing patch ' + str(patch))
            map_Dist[i0, i1, i2, :] = 0
            map_stdDist[i0, i1, i2, :] = 0
            map_PredLabel[i0, i1, i2, :] = 0
            map_stdPredLabel[i0, i1, i2, :] = 0

        # Average over cv
        map_stdACC[i0, i1, i2] = std_acc
        map_ACC[i0, i1, i2]    = mean_acc
        if mean_acc > bestacc:
            bestacc     = mean_acc
            std_bestacc = std_acc
            bestpatch   = patch

        for patID in range(0, len(partition[subset])):
            mean_cv_dists[patch, patID]    = np.mean(mean_dists[patch, patID, :])
            mean_cv_absdists[patch, patID] = np.mean(mean_absdists[patch, patID, :])
            std_cv_dists[patch, patID]     = std_dists[patch, patID, 0]
            std_cv_absdists[patch, patID]  = std_absdists[patch, patID, 0]

            # Fill map for this patient only
            map_Dist[i0, i1, i2, patID]         = mean_cv_absdists[patch, patID]
            map_stdDist[i0, i1, i2, patID] = std_cv_absdists[patch, patID]
            map_PredLabel[i0, i1, i2, patID] = mean_cv_dists[patch, patID]
            map_stdPredLabel[i0, i1, i2, patID] = std_cv_dists[patch, patID]

    print("Best acc: " + str(bestacc) + '+/-' + str(std_bestacc) + ' for patch ' + str(bestpatch))

    # Get missing patches:
    if len(missingPatches) > 1:
        print('Missing these patches: ', missingPatches)
    else:
        print('No missing patches')


    # Summarize
    trueLabels = []
    for patID in range(0, len(partition[subset])):

        print(patID)
        try:
            pat = partition[subset].values[patID,]
        except:
            pat = partition[subset][patID]
        trueLabel = labels[pat]
        trueLabels.append(trueLabel)

        # Get X:
        X[patID, :] = np.concatenate(
            (map_PredLabel[:, :, :, patID].reshape((np.prod(map_PredLabel[:, :, :, patID].shape),)),
             map_stdPredLabel[:, :, :, patID].reshape((np.prod(map_PredLabel[:, :, :, patID].shape),)),
             map_ACC[:, :, :].reshape((np.prod(map_PredLabel[:, :, :, patID].shape),)),
             ))


    return X, np.array(trueLabels)

def saveResults(str_name, rows_names, av_dfTDA_LR_cv,av_dfCombi_LR_cv,av_df3D_CNN_cv, av_df3D_LR_cv, av_dfEns_LR_cv):
    aps = [av_dfTDA_LR_cv.loc[str_name],
           av_dfCombi_LR_cv.loc[str_name],
           av_df3D_CNN_cv.loc[str_name],
           av_df3D_LR_cv.loc[str_name],
           av_dfEns_LR_cv.loc[str_name]]

    df_aps = pd.DataFrame(aps, index=rows_names)

    return df_aps
