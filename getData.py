###########################################################################################
# Imports
###########################################################################################

import os
import numpy as np
import json

###########################################################################################
# Functions
###########################################################################################

# TDA Data import
def getTDA(
        data_dir='',
        dimensions=[0, 1, 2],
        partition_file='',
        labels_file='',
        reshape=False,
        resolution=[50, 50],
):
    '''
    Visualize the performance of a binary Classifier.

    Displays a labelled Confusion Matrix, distributions of the predicted
    probabilities for both classes, the Recall-Precision curve, the ROC curve, and F1 score of a fitted
    Binary Logistic Classifier. Author: gregcondit.com/articles/logr-charts

    Parameters
    ----------
    data_dir:       str, path to the directory where the data is saved
    dimensions:     list, indicate which dimension persistence features should be used: choose from 0,1,2
    partition_file: dict, dictionary of the partion (train, validation or test) and the corresponding subject IDs
    labels_file:    dict, dictionary of the subject ID and corresponding label
    reshape:        bool, indicate if PI should be reshaped or not
    resolution:     optional, size of output persitance image

    Returns
    ----------
    X_dict: dictionary of the features

    y_dict: dictionary of the labels (0,1)
    '''


    part = np.load(partition_file, allow_pickle=True).item()
    labels = np.load(labels_file, allow_pickle=True).item()

    X_dict = {}
    y_dict = {}

    for key in part.keys():
        X_dict[key] = []
        y_dict[key] = []

        for composite in part[key]:
            comp_split = composite.split('-')
            subject = comp_split[0] + '-' + comp_split[1]
            session = comp_split[2]
            mode = comp_split[3]

            sess_dir = os.path.join(data_dir, subject, session)
            pi_list = [x for x in os.listdir(sess_dir) if (mode in x)]

            for d in [0, 1, 2]:  # exclude dimensions if desired:
                if d not in dimensions:
                    pi_list = [x for x in pi_list if f'dim{d}' not in x]

            next_pi = []
            for pi_im in pi_list:

                # get the persistence image of the subject & session:
                with open(os.path.join(sess_dir, pi_im), 'r') as pimg_data:
                    pi = json.load(pimg_data)
                if reshape:
                    next_pi.append(np.array(pi['0'][0]).reshape(resolution[0], resolution[1]))
                else:
                    next_pi.extend(pi['0'][0])
            if reshape:
                next_pi = np.transpose(np.array(next_pi), (1, 2, 0))

            X_dict[key].append(next_pi)
            y_dict[key].append(labels[composite])

    return X_dict, y_dict

# 3D CNN patch import
def getArray(data_folder_all, patch, partition, labels, dim, n_channels, usescale01 = False):


    folder = os.path.join(data_folder_all,str(patch))

    X_trn = np.empty((len(partition['train']), *dim, n_channels))
    y_trn = np.empty((len(partition['train'])), dtype=int)
    X_val = np.empty((len(partition['validation']), *dim, n_channels))
    y_val = np.empty((len(partition['validation'])), dtype=int)

    # Generate data
    for i, ID in enumerate(partition['train']):
        if i ==0:
            print('at step %d of %d' % (i, len(partition['train'])))

        # load
        thisX = np.load(os.path.join(folder, ID + '.npy'))

        # Scale
        if usescale01:
            minX = np.quantile(thisX,0.001)
            maxX = np.quantile(thisX, 0.999)
            thisX[thisX>maxX]= maxX
            thisX[thisX < minX] = minX
            sc_X = (thisX-minX)/(maxX-minX)
        else:
            sc_X = thisX

        # Store sample
        X_trn[i,] = np.expand_dims(sc_X, axis=3)

        # Store class
        y_trn[i]  = labels[ID]

    for i, ID in enumerate(partition['validation']):
        if i == 0:
            print('at step %d of %d' % (i, len(partition['validation'])))

        #scaler = MinMaxScaler()
        # load
        thisX = np.load(os.path.join(folder, ID + '.npy'))

        # Scale
        if usescale01:
            minX = np.quantile(thisX, 0.001)
            maxX = np.quantile(thisX, 0.999)
            thisX[thisX > maxX] = maxX
            thisX[thisX < minX] = minX
            sc_X = (thisX - minX) / (maxX - minX)
        else:
            sc_X = thisX

        # Store sample
        X_val[i,] = np.expand_dims(sc_X, axis=3)
        # Store class
        y_val[i]  = labels[ID]




    return X_trn, y_trn, X_val, y_val