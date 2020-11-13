###########################################################################################
# Libraries
###########################################################################################
import pandas as pd
import os
import tensorflow as tf
import argparse
import numpy as np
import pickle
from keras.models import Model
from sklearn.utils import class_weight
from tensorflow import keras
from keras.utils import np_utils

# Functions
from getData import getArray, getTDA
from models import TDA_3DCNN_combinedModel
from plotFcts import plot_trainingProcess, evalBinaryClassifierCNN


########################################################################################################################
# Global parameters
########################################################################################################################
useArgs = True
wd      = os.getcwd()
########################################################################################################################


if __name__ in "__main__":
    ###########################################################################################
    # Input section
    ###########################################################################################

    # Run info general selection
    doplots                = True
    partition_suffix       = '0CN_1AD_pat_1GO23_'

    # Model choices
    clf_3D          = 'gap'
    clf             = 'dense'         # Options: 'dense', 'clear'
    choice_modelTDA = 'PI_CNN_model'  # Options: 'PI_CNN_model', 'None'
    choice_model3D  = 'CNNflexi'      # Options: 'None', 'CNNflexi'


    # Data folders
    output          = ''
    data_folder_3D  = ''
    data_folder_TDA = ''
    wd              = os.getcwd()
    path            = wd


    # Arguments
    if useArgs:
        parser = argparse.ArgumentParser()
        parser.add_argument('--cv', nargs='+', type=int,default=1)
        parser.add_argument('--runs', nargs='+', type=int,default=0)
        parser.add_argument('--patch', nargs='+', type=int, default = 91)
        parser.add_argument('--useDimsTDA_ID', nargs='+', type=int, default=0)
        parser.add_argument('--parameterIDTDA', nargs='+', type=int, default=84)
        parser.add_argument('--parameterID3D', nargs='+', type=int, default = 0)
        parser.add_argument('--parameterIDCombi_Tail', nargs='+', type=int, default = 0)

        args                  = parser.parse_args()
        cv                    = args.cv[0]
        patch                 = args.patch[0]
        runIDsuse             = [args.runs[0]]
        parameterID3D         = args.parameterID3D[0]
        parameterIDCombi_Tail = args.parameterIDCombi_Tail[0]
        useDimsTDA_ID         = args.useDimsTDA_ID[0]
        parameterIDTDA        = [84]
    else:

        # Defaults:
        parameterIDTDA       = 84
        parameterID3D        = 0
        parameterIDCombi_Tail= 0
        cv                   = 1
        patch                = 91
        runIDsuse            = [0]
        useDimsTDA_ID        = 3



    # 3D image dimensions
    dim0p = 30
    dim1p = 36
    dim2p = 30
    patchdim = (dim0p, dim1p, dim2p)
    n_channels_3D = 1


    # TDA
    dim0tda = 50
    dim1tda = 50
    dim2tda = 0
    patchdimTDA = (dim0tda, dim1tda, dim2tda)
    n_channels_tda = 1
    if useDimsTDA_ID == 3:
        useDimsTDA     = [0, 1, 2]
        parameterIDTDA = [84, 136, 80]
    else:
        useDimsTDA = [useDimsTDA_ID]



    # Tail parameters
    n_classes   = 2
    n_epochs    = 2500
    batchSz_all = [20,50]
    lrs_all     = [0.0001,0.00001]
    l1_tail_all = [0.001,0.00001]
    batchSz_lst = []
    lrs_lst     = []
    l1_tail_lst = []
    for this_batchSz in batchSz_all:
        for this_lrs in lrs_all:
            for this_l1_tail in l1_tail_all:
                l1_tail_lst.append(this_l1_tail)
                lrs_lst.append(this_lrs)
                batchSz_lst.append(this_batchSz)
    bs      = batchSz_lst[parameterIDCombi_Tail]
    lr      = lrs_lst[parameterIDCombi_Tail]
    l1_tail = l1_tail_lst[parameterIDCombi_Tail]


    # 3D Parameters
    n_filter_1_3D = 32
    n_filter_2_3D = 64
    l2_3D_all     = [0,0.0001]
    l1_act3D_all  = [0,0.0001]
    BNloc_3D_all  = [0,1,2]
    useDO_3D_all  = [False,True]
    k_size_3D_all = [3,4]
    if clf_3D == 'dense':
        l1_den3D = [0,0.01,0.001,0.0001]
    else:
        l1_den3D = np.nan
    l1_act3D_lst  = []
    BNloc_3D_lst  = []
    useDO_3D_lst  = []
    k_size_3D_lst = []
    l2_3D_lst      = []
    for this_k_size_3D in k_size_3D_all:
        for this_useDO_3D in useDO_3D_all:
            for this_BNloc_3D in BNloc_3D_all:
                for this_l1_act3D in l1_act3D_all:
                    for this_l2_3D in l2_3D_all:
                        k_size_3D_lst.append(this_k_size_3D)
                        useDO_3D_lst.append(this_useDO_3D)
                        BNloc_3D_lst.append(this_BNloc_3D)
                        l1_act3D_lst.append(this_l1_act3D)
                        l2_3D_lst.append(this_l2_3D)
    l1_act3D  = l1_act3D_lst[parameterID3D]
    k_size_3D = k_size_3D_lst[parameterID3D]
    useDO_3D  = useDO_3D_lst[parameterID3D]
    BNloc_3D  = BNloc_3D_lst[parameterID3D]
    l2_3D     = l2_3D_lst[parameterID3D]



    # TDA Parameters
    n_layers_TDA_all  = [4]
    k_size_TDA_all    = [4,5]
    l1s_TDA_all       = [0,0.001,0.0001,0.00001]
    dos_TDA_all       = [0,0.5,0.25]
    mp_stride_TDA_all = [2]
    mp_ksize_TDA_all  = [2,4]
    lr_TDA_all        = [0.00001, 0.0001, 0.001]
    gap_TDA_all  = [False]

    l1_TDA_lst = []
    do_TDA_lst = []
    k_size_TDA_lst = []
    n_layers_TDA_lst = []
    mp_stride_TDA_lst = []
    gap_TDA_lst = []
    mp_ksize_TDA_lst = []
    lr_TDA_lst = []
    for this_n_layers_TDA in n_layers_TDA_all:
        for this_k_size_TDA in k_size_TDA_all:
            for this_l1_TDA in l1s_TDA_all:
                for this_dos_TDA in dos_TDA_all:
                    for this_mp_stride_TDA in mp_stride_TDA_all:
                        for this_gap_TDA in gap_TDA_all:
                            for this_lr_TDA in lr_TDA_all:
                                for this_mp_ksize_TDA in mp_ksize_TDA_all:
                                    l1_TDA_lst.append(this_l1_TDA)
                                    do_TDA_lst.append(this_dos_TDA)
                                    k_size_TDA_lst.append(this_k_size_TDA)
                                    n_layers_TDA_lst.append(this_n_layers_TDA)
                                    mp_stride_TDA_lst.append(this_mp_stride_TDA)
                                    gap_TDA_lst.append(this_gap_TDA)
                                    lr_TDA_lst.append(this_lr_TDA)
                                    mp_ksize_TDA_lst.append(this_mp_ksize_TDA)


    modeluse = choice_modelTDA + '_' + choice_model3D + '_' + clf
    print('Model use: '+modeluse)

    ###########################################################################################
    # Run model
    ###########################################################################################

    # Configure GPU
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))



    #####################################
    #       Get hyperparameters         #
    #####################################

    # Hyperparameters used for all model arm which HAVE TO BE shared
    input_Combi = { 'clf': clf,
                    'n_classes': n_classes,
                    'n_epochs': n_epochs,
                    'bs': bs,
                    'lr': lr,
                    'pat': int(0.05 * n_epochs),
                    'l1': l1_tail}



    #####################################
    #              Load Data            #
    #####################################

    # Get partition and labels containing information on training, validation, test samples and labels
    partition_suffix = partition_suffix + str(cv)
    partition_file   = path + 'partitions/partition_' + partition_suffix + '.npy'
    label_file       = path + 'partitions/labels_' + partition_suffix + '.npy'
    partition        = np.load(partition_file, allow_pickle='TRUE').item()  # IDs
    labels           = np.load(label_file, allow_pickle='TRUE').item()  # Labels


    # Load 3D image data
    if choice_model3D != 'None':

        print('Loading 3D MRI data')

        # Get the specified patches
        print('Working on patch ' + str(patch))
        patchid2 = patch - np.floor(patch / 36) * 36 - np.floor((patch - np.floor(patch / 36) * 36) / 6) * 6
        patchid1 = np.floor((patch - np.floor(patch / 36) * 36) / 6)
        patchid0 = np.floor(patch / 36)
        patchid  = (patchid0, patchid1, patchid2)

        # Get the actual array
        X_trn_3D, y_trn_3D, X_val_3D, y_val_3D = getArray(data_folder_3D, patch, partition, labels, patchdim,n_channels_3D)

        # Make categorical
        y_trn = np_utils.to_categorical(y_trn_3D, n_classes)
        y_val = np_utils.to_categorical(y_val_3D, n_classes)

        # Sample shapes
        sample_shape_3D = (input_Combi['bs'], dim0p, dim1p, dim2p, 1)

        # Hyperparameters for 3DCNN
        params_3D = {'n_channels': n_channels_3D,
                     'datadir': data_folder_3D,
                     'patchid': patch,
                     'patchdim': patchdim,
                     'BNloc': BNloc_3D,
                     'useDO': useDO_3D,
                     'n_filters_1': n_filter_1_3D,
                     'n_filters_2': n_filter_2_3D,
                     'kernel_size': k_size_3D,
                     'l2': l2_3D,
                     'l1_den': l1_den3D,
                     'l1_act': l1_act3D,
                     'sample_shape': sample_shape_3D,
                     'clf': clf_3D
                     }
    else:
        sample_shape_3D = (1,1,1)
        patchid2 = np.nan
        patchid1 = np.nan
        patchid0 = np.nan
        patchid = (patchid0, patchid1, patchid2)
        params_3D = {}


    # load TDA data, sample size and hyperparameters
    if choice_modelTDA == 'None':
        X_dicTDA = np.empty((1,1,1))
        y_dicTDA = 0
        sample_shape_TDA = 0
        params_TDA = {}
    else:
        print('Loading persistence images')


        # Get data
        X_dicTDA, y_dicTDA = getTDA(
            data_dir      = data_folder_TDA,
            dimensions    = useDimsTDA,
            partition_file= partition_file,
            labels_file   = label_file,
            reshape       = True,
            resolution    = [dim0tda, dim1tda])

        # Labels
        y_trnTDA = np.array(y_dicTDA['train'])
        y_valTDA = np.array(y_dicTDA['validation'])
        y_trnTDA = np_utils.to_categorical(y_trnTDA, n_classes)
        y_valTDA = np_utils.to_categorical(y_valTDA, n_classes)

        # Sample shape
        if choice_modelTDA == 'MLP':
            sample_shape_TDA = (np.asarray(X_dicTDA['train']).shape[1])
        else:
            if len(useDimsTDA)==3:
                sample_shape_TDA = (np.asarray(X_dicTDA['train']).shape[1],np.asarray(X_dicTDA['train']).shape[2],1)
            else:
                sample_shape_TDA = (np.asarray(X_dicTDA['train']).shape[1:])
        print('Sample shape TDA: ', sample_shape_TDA)

        # Hyperparameters
        if len(useDimsTDA) == 3:

            # Use all dimensions:
            params_TDA_0 = {'l1': l1_TDA_lst[parameterIDTDA[0]],
                            'do': do_TDA_lst[parameterIDTDA[0]],
                            'kernel_size': k_size_TDA_lst[parameterIDTDA[0]],
                            'n_layers': n_layers_TDA_lst[parameterIDTDA[0]],
                            'mp_stride': mp_stride_TDA_lst[parameterIDTDA[0]],
                            'gap': False,
                            'mp_ksize': mp_ksize_TDA_lst[parameterIDTDA[0]]}

            params_TDA_1 = {'l1': l1_TDA_lst[parameterIDTDA[1]],
                            'do': do_TDA_lst[parameterIDTDA[1]],
                            'kernel_size': k_size_TDA_lst[parameterIDTDA[1]],
                            'n_layers': n_layers_TDA_lst[parameterIDTDA[1]],
                            'mp_stride': mp_stride_TDA_lst[parameterIDTDA[1]],
                            'gap': False,
                            'mp_ksize': mp_ksize_TDA_lst[parameterIDTDA[1]]}

            params_TDA_2 = {'l1': l1_TDA_lst[parameterIDTDA[2]],
                            'do': do_TDA_lst[parameterIDTDA[2]],
                            'kernel_size': k_size_TDA_lst[parameterIDTDA[2]],
                            'n_layers': n_layers_TDA_lst[parameterIDTDA[2]],
                            'mp_stride': mp_stride_TDA_lst[parameterIDTDA[2]],
                            'gap': False,
                            'mp_ksize': mp_ksize_TDA_lst[parameterIDTDA[2]]}

            params_TDA = {'params_TDA_0': params_TDA_0,
                          'params_TDA_1': params_TDA_1,
                          'params_TDA_2': params_TDA_2, }
        else:
            params_TDA = {  'l1': l1_TDA_lst[parameterIDTDA[useDimsTDA_ID]],
                            'do': do_TDA_lst[parameterIDTDA[useDimsTDA_ID]],
                            'kernel_size': k_size_TDA_lst[parameterIDTDA[useDimsTDA_ID]],
                            'n_layers': n_layers_TDA_lst[parameterIDTDA[useDimsTDA_ID]],
                            'mp_stride': mp_stride_TDA_lst[parameterIDTDA[useDimsTDA_ID]],
                            'gap': False,
                            'mp_ksize': mp_ksize_TDA_lst[parameterIDTDA[useDimsTDA_ID]]}


    # Check compatability of TDA and patch CNN labels
    if choice_model3D == 'None':
        y_trn = y_trnTDA
        y_val = y_valTDA
    else:
        if choice_modelTDA != 'None':
            check_trn = y_trnTDA - y_trn
            check_val = y_valTDA - y_val
            if (sum(check_trn).sum()+sum(check_val).sum())>0:
                raise('Error: Labels of TDA and patch CNN do not agree!!!')


    # Loop over all runs - option to skip
    for counter,runIDuse in enumerate(runIDsuse):

        # Name
        name = modeluse + '_patch' + str(patch) + '_run' + str(runIDuse) + 'CV_' + str(cv) + '_pIDtda' + \
               str(parameterIDTDA) + '_pID3D' +str(parameterID3D) + '_pIDco'+ str(parameterIDCombi_Tail) + \
               '_tdadim'+ str(useDimsTDA)


        #####################################
        #           build the model         #
        #####################################

        # Model input
        input3D = {'sample_shape': sample_shape_3D,
                   'params': params_3D}
        inputTDA = {'sample_shape': sample_shape_TDA,
                    'params': params_TDA}
        if choice_modelTDA == 'None':
            input = X_trn_3D
            val_data = (X_val_3D, y_val)
            val_data_X = X_val_3D
        else:
            if choice_model3D == 'None':

                # One TDA dim:
                if len(useDimsTDA) == 3:
                    allX_TDA    = np.asarray(X_dicTDA['train'])
                    allXval_TDA = np.asarray(X_dicTDA['validation'])
                    input = [allX_TDA[:,:,:,0], allX_TDA[:,:,:,1], allX_TDA[:,:,:,2]]
                    val_data = ([allXval_TDA[:,:,:,0],allXval_TDA[:,:,:,1],allXval_TDA[:,:,:,2]], y_val)
                    val_data_X = [allXval_TDA[:,:,:,0],allXval_TDA[:,:,:,1],allXval_TDA[:,:,:,2]]
                else:
                    input = np.asarray(X_dicTDA['train'])
                    val_data = (np.asarray(X_dicTDA['validation']), y_val)
                    val_data_X = np.asarray(X_dicTDA['validation'])
            else:
                if len(useDimsTDA) == 3:
                    allX_TDA = np.asarray(X_dicTDA['train'])
                    allXval_TDA = np.asarray(X_dicTDA['validation'])
                    input = [allX_TDA[:,:,:,0], allX_TDA[:,:,:,1], allX_TDA[:,:,:,2], X_trn_3D]
                    val_data = ([allXval_TDA[:,:,:,0],allXval_TDA[:,:,:,1],allXval_TDA[:,:,:,2], X_val_3D], y_val)
                    val_data_X = [allXval_TDA[:,:,:,0],allXval_TDA[:,:,:,1],allXval_TDA[:,:,:,2], X_val_3D]
                else:
                    input = [np.asarray(X_dicTDA['train']), X_trn_3D]
                    val_data = ([np.asarray(X_dicTDA['validation']), X_val_3D], y_val)
                    val_data_X = [np.asarray(X_dicTDA['validation']), X_val_3D]

        # Get the model
        model = TDA_3DCNN_combinedModel(inputTDA, input3D, input_Combi, choice_modelTDA, choice_model3D)

        # Compile the model
        monitor_var = 'loss'
        monitor_val_var = 'val_loss'
        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.Adam(lr=input_Combi['lr']),
                      metrics=None)

        model.summary()

        #####################################
        #           train the model         #
        #####################################

        #  Early stopping and check points
        filepath      = os.path.join(os.path.join(output, 'run_hdf5'), name +'_nnet_run_test.hdf5')
        check         = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='auto')
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=input_Combi['pat'],
                                                         mode='auto')
        class_weights     = class_weight.compute_class_weight('balanced', np.unique(y_trn[:, 1]), y_trn[:, 1])
        class_weight_dict = dict(enumerate(class_weights))

        # Actual training
        history = model.fit(x=input, y=y_trn,
                            batch_size=input_Combi['bs'],
                            epochs=input_Combi['n_epochs'],
                            verbose=1,
                            callbacks=[check, earlyStopping],
                            validation_data=val_data,
                            class_weight=class_weight_dict,
                            shuffle = True
                            )



        #####################################
        #        evaluate the model         #
        #####################################

        # Load the best weights
        model.load_weights(filepath)

        # Get encoding
        encoder = Model(model.input, model.layers[-2].output)
        encoder.summary()

        # Predict
        x_trn_encoded = encoder.predict(input)
        x_val_encoded = encoder.predict(val_data_X)
        weights = model.layers[-2].get_weights()

        # Save encodings and weights of pre classificaiton layer
        x_trn_encoded_flat   = np.reshape(x_trn_encoded, (x_trn_encoded.shape[0], np.prod(x_trn_encoded.shape[1:])))
        x_val_encoded_flat   = np.reshape(x_val_encoded, (x_val_encoded.shape[0], np.prod(x_val_encoded.shape[1:])))
        filepath_enc_trn     = os.path.join(os.path.join(output, 'run_hdf5'),'encoding_trn_'+name+'.npy')
        filepath_enc_val     = os.path.join(os.path.join(output, 'run_hdf5'), 'encoding_val_' + name +'.npy')
        filepath_enc_weights = os.path.join(os.path.join(output, 'run_hdf5'), 'encoding_weights_' + name + '.npy')
        with open(filepath_enc_trn, 'wb') as f:
            np.save(f,x_trn_encoded_flat, allow_pickle=True)
        with open(filepath_enc_val, 'wb') as f:
            np.save(f, x_val_encoded_flat, allow_pickle=True)
        with open(filepath_enc_weights, 'wb') as f:
            np.save(f,weights, allow_pickle=True)


        # Plot training process (loss)
        if doplots:
            plot_trainingProcess(history, os.path.join(output, 'run_losses'), name)


        # Evaluate validation set and save
        tn, fp, fn, tp, acc, precision, recall, roc_auc, aps, dist, meanDist, stdDist, thresh_opt, y_pred = \
            evalBinaryClassifierCNN(model, val_data_X, y_val, os.path.join(output, 'run_eval'),
                                    labels=['Negatives','Positives'],
                                    plot_title='Evaluation validation of ' + modeluse + '_' + clf,
                                    doPlots=doplots,
                                    batchsize=input_Combi['bs'],
                                    filename='val_' + name)
        dic_dist = {'y_pred': y_pred, 'dist': dist, 'thresh': thresh_opt}
        df_dist = pd.DataFrame(dic_dist)


        # Evaluate training data and save
        tn_trn, fp_trn, fn_trn, tp_trn, acc_trn, precision_trn, recall_trn, roc_auc_trn, aps_trn, dist_trn, \
        meanDist_trn, stdDist_trn, thresh_opt_trn, y_pred_trn = \
            evalBinaryClassifierCNN(model, input, y_trn, os.path.join(output, 'run_eval'),
                                    labels=['Negatives','Positives'],
                                    plot_title='Evaluation validation of ' + modeluse + '_' + clf,
                                    doPlots=False,
                                    batchsize=input_Combi['bs'],
                                    filename='trn'+name)
        dic_dist_trn = {'y_pred': y_pred_trn, 'dist': dist_trn, 'thresh': thresh_opt_trn}
        df_dist_trn = pd.DataFrame(dic_dist_trn)


        # Save results to df
        data_df = {'runID': str(runIDuse),
                   'data_folder': data_folder_3D,
                   'data_folderTDA': data_folder_TDA,
                   'model': modeluse,
                   'clf': clf,
                   'n_epochs': n_epochs,
                   'tn': tn,
                   'fp': fp,
                   'fn': fn,
                   'tp': tp,
                   'acc': acc,
                   'precision': precision,
                   'recall': recall,
                   'auc': roc_auc,
                   'aps': aps,
                   'meanDist': meanDist,
                   'stdDist': stdDist
                   }
        df2 = pd.DataFrame(data=data_df, index=[runIDuse])
        if counter == 0:
            df = df2.copy()
        else:
            df = df.append(df2, ignore_index=True)

        df_dist_trn.to_csv(os.path.join(os.path.join(output, 'run_distance'),name +  '_results_dist_trn.csv'))
        df_dist.to_csv(os.path.join(os.path.join(output, 'run_distance'),name + '_results_dist.csv'))
        df.to_csv(os.path.join(os.path.join(output, 'run_overviews'), name + '_results_overview.csv'))


        # Save parameters
        if choice_model3D != 'None':
            file = os.path.join(os.path.join(output, 'run_parameters'), name +'_param3D.pkl')
            with open(file, 'wb') as f:
                pickle.dump(params_3D, f)

        if choice_modelTDA!= 'None':
            file = os.path.join(os.path.join(output, 'run_parameters'), name + '_paramTDA.pkl')
            with open(file, 'wb') as f:
                pickle.dump(params_TDA, f)


        # Clear keras
        tf.keras.backend.clear_session()

