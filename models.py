###########################################################################################
# Imports
###########################################################################################

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, AveragePooling2D, Input, \
    concatenate
from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D, Activation, GlobalAveragePooling2D, \
    GlobalAveragePooling3D, UpSampling3D, Lambda, Reshape, Conv1D, MaxPooling1D, GlobalAveragePooling1D

from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.activations import relu


###########################################################################################
# Model functions
###########################################################################################

# 3D Model
def build_CNNflexi(input):
    """Simple CNN with 2 Conv layers and either globalAveragePooling or fully connected layers for classification"""

    params_3D = input['params']
    sample_shape = input['sample_shape']
    inputA = Input(shape=sample_shape[1:])
    clf = params_3D['clf']

    # Model head
    if params_3D['BNloc'] == 0:
        x = Conv3D(
            params_3D['n_filters_1'],
            kernel_size=(params_3D['kernel_size'], params_3D['kernel_size'], params_3D['kernel_size']),
            activation=None,
            kernel_initializer='he_uniform',
            input_shape=params_3D['sample_shape'],
            padding='SAME',
            data_format='channels_last',
            kernel_regularizer=l2(params_3D['l2']),
            bias_regularizer=l2(params_3D['l2'])
        )(inputA)

        x = BatchNormalization(center=True, scale=True, trainable=False)(x)
        x = Activation(relu, activity_regularizer=l1(params_3D['l1_act']))(x)

    else:
        x = Conv3D(
            params_3D['n_filters_1'],
            kernel_size=(params_3D['kernel_size'], params_3D['kernel_size'], params_3D['kernel_size']),
            activation='relu',
            kernel_initializer='he_uniform',
            padding='SAME',
            kernel_regularizer=l2(params_3D['l2']),
            bias_regularizer=l2(params_3D['l2']),
            activity_regularizer=l1(params_3D['l1_act'])
        )(inputA)

    if params_3D['BNloc'] == 1:
        x = BatchNormalization(center=True, scale=True, trainable=False)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    if params_3D['BNloc'] == 2:
        x = BatchNormalization(center=True, scale=True, trainable=False)(x)

    if params_3D['useDO']:
        x = Dropout(0.25)(x)

    # Next layer
    if params_3D['BNloc'] == 0:
        x = Conv3D(
            params_3D['n_filters_2'],
            kernel_size=(params_3D['kernel_size'], params_3D['kernel_size'], params_3D['kernel_size']),
            activation=None,
            kernel_initializer='he_uniform',
            input_shape=params_3D['sample_shape'],
            padding='SAME',
            data_format='channels_last',
            kernel_regularizer=l2(params_3D['l2']),
            bias_regularizer=l2(params_3D['l2'])
        )(x)

        x = BatchNormalization(center=True, scale=True, trainable=False)(x)
        x = Activation(relu, activity_regularizer=l1(params_3D['l1_act']))(x)

    else:
        x = Conv3D(
            params_3D['n_filters_2'],
            kernel_size=(params_3D['kernel_size'], params_3D['kernel_size'], params_3D['kernel_size']),
            activation='relu',
            kernel_initializer='he_uniform',
            padding='SAME',
            kernel_regularizer=l2(params_3D['l2']),
            bias_regularizer=l2(params_3D['l2']),
            activity_regularizer=l1(params_3D['l1_act'])
        )(x)

    if params_3D['BNloc'] == 1:
        x = BatchNormalization(center=True, scale=True, trainable=False)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    if params_3D['BNloc'] == 2:
        x = BatchNormalization(center=True, scale=True, trainable=False)(x)

    if params_3D['useDO']:
        x = Dropout(0.25)(x)


    # Classifiers
    if clf == 'dense':
        x = Flatten()(x)
        x = Dense(256, activation='relu', kernel_initializer='he_uniform',
                  activity_regularizer=l1(params_3D['l1_den']))(x)
        x = Dense(128, activation='relu', kernel_initializer='he_uniform',
                  activity_regularizer=l1(params_3D['l1_den']))(x)
        x = Dense(64, activation='relu', kernel_initializer='he_uniform', activity_regularizer=l1(params_3D['l1_den']))(
            x)
    elif clf == 'gap':
        x = GlobalAveragePooling3D()(x)
    else:
        raise ('Invalid classifier choice!')

    model = Model(inputs=inputA, outputs=x)
    return model

# TDA model:
def conv2D_block(num_filters=32, sample_shape=None, inputA=None, drop_out=0, first_layer=False, conv_kernel_size=(5, 5),
                 conv_stride=(1, 1), GAP=True, max_pool_size=(2, 2), max_pool_stride=(1, 1), act_regularization=0):
    if first_layer:
        # inputA = Input(shape=sample_shape)
        conv = Conv2D(num_filters, kernel_size=conv_kernel_size, strides=conv_stride,
                      kernel_initializer='he_uniform',
                      # kernel_regularizer=l1(act_regularization),
                      # bias_regularizer=l1(act_regularization),
                      # activity_regularizer=None,
                      activation=None,
                      padding='same',
                      use_bias=False,
                      input_shape=sample_shape)(inputA)
    else:
        conv = Conv2D(num_filters, kernel_size=conv_kernel_size, strides=conv_stride,
                      kernel_initializer='he_uniform',
                      # kernel_regularizer=l1(act_regularization),
                      # bias_regularizer=l1(act_regularization),
                      # activity_regularizer=None,
                      activation=None,
                      padding='same',
                      use_bias=False)(inputA)

    bn = BatchNormalization(center=True, scale=True, trainable=False)(conv)
    act = Activation('relu', activity_regularizer=l1(act_regularization))(bn)
    if GAP:
        out = GlobalAveragePooling2D()(act)
    else:
        max_pool = MaxPooling2D(pool_size=max_pool_size, strides=max_pool_stride)(act)
        out = Dropout(drop_out)(max_pool)

    return out

# Combined model 3D + TDA
def TDA_3DCNN_combinedModel(inputTDA, input3D, inputTail, choice_modelTDA, choice_model3D):

    #########################################
    #     Top arms of the combined model    #
    #########################################

    # TDA
    if choice_modelTDA == 'PI_CNN_model':

        if len(inputTDA['params']) == 3:
            inputTDA0 = {'sample_shape': inputTDA['sample_shape'],
                         'params': inputTDA['params']['params_TDA_0']}
            inputTDA1 = {'sample_shape': inputTDA['sample_shape'],
                         'params': inputTDA['params']['params_TDA_1']}
            inputTDA2 = {'sample_shape': inputTDA['sample_shape'],
                         'params': inputTDA['params']['params_TDA_2']}
            modelTDA0 = PI_CNN_model(inputTDA0)
            modelTDA1 = PI_CNN_model(inputTDA1)
            modelTDA2 = PI_CNN_model(inputTDA2)
        else:
            modelTDA = PI_CNN_model(inputTDA)
    elif choice_modelTDA == 'None':
        modelTDA = []
    else:
        raise ('TDA model not supported!')

    # 3D patch model
    if choice_model3D == 'CNNflexi':
        model3D = build_CNNflexi(input3D)
    elif choice_model3D == 'None':
        model3D = []
    else:
        raise ('3D model not supported!')

    ###############################
    #           Merging           #
    ###############################

    # Combined input
    if choice_modelTDA == 'None':
        combinedInput = model3D.output
    else:
        if choice_model3D == 'None':
            if len(inputTDA['params']) == 3:
                combinedInput = concatenate([modelTDA0.output, modelTDA1.output, modelTDA2.output])
            else:
                combinedInput = modelTDA.output
        else:
            if len(inputTDA['params']) == 3:
                combinedInput = concatenate([modelTDA0.output, modelTDA1.output, modelTDA2.output, model3D.output])
            else:
                combinedInput = concatenate([modelTDA.output, model3D.output])

    # Tail
    if inputTail['clf'] == 'clear':
        x = Dense(inputTail['n_classes'], activation='sigmoid')(combinedInput)
    elif inputTail['clf'] == 'dense':
        x = Dense(64, activation='relu', kernel_initializer='he_uniform', activity_regularizer=l1(inputTail['l1']))(
            combinedInput)
        x = Dense(32, activation='relu', kernel_initializer='he_uniform', activity_regularizer=l1(inputTail['l1']))(x)
        x = Dense(inputTail['n_classes'], activation='sigmoid')(x)
    elif inputTail['clf'] == 'gap':
        x = GlobalAveragePooling3D()(combinedInput)
        x = Dense(inputTail['n_classes'], activation='sigmoid')(x)
    else:
        raise ('Invalid tail classifier choice!')


    # Combined output
    if choice_modelTDA == 'None':
        model = Model(inputs=model3D.inputs, outputs=x)
    else:
        if choice_model3D == 'None':
            if len(inputTDA['params']) == 3:
                model = Model(inputs=[modelTDA0.inputs, modelTDA1.inputs, modelTDA2.inputs], outputs=x)
            else:
                model = Model(inputs=modelTDA.inputs, outputs=x)
        else:
            if len(inputTDA['params']) == 3:
                model = Model(inputs=[modelTDA0.inputs, modelTDA1.inputs, modelTDA2.inputs, model3D.inputs], outputs=x)
            else:
                model = Model(inputs=[modelTDA.inputs, model3D.inputs], outputs=x)

    return model


########################################################################################################################
# Subfucntions
########################################################################################################################
def PI_CNN_model(input):
    params = input['params']
    dense_tail = False
    gap_tail = params['gap']

    num_conv_layers = params['n_layers']
    first_layer = True
    num_filters = 32
    drop_out = params['do']
    act_regularization = params['l1']
    input_dim = input['sample_shape']
    conv_kernel_size = (params['kernel_size'], params['kernel_size'])
    conv_stride = (1, 1)
    max_pool_stride = (params['mp_stride'], params['mp_stride'])
    max_pool_size = (params['mp_ksize'], params['mp_ksize'])

    model = Sequential()
    setGap = False
    inputB = Input(shape=input_dim)

    for i in range(0, num_conv_layers):

        if i == num_conv_layers - 1:
            setGap = True

        if i > 0:
            first_layer = False
            out = conv2D_block(inputA=out, num_filters=num_filters, drop_out=drop_out, sample_shape=input_dim,
                               GAP=setGap,
                               act_regularization=act_regularization, conv_kernel_size=conv_kernel_size,
                               conv_stride=conv_stride, max_pool_size=max_pool_size, max_pool_stride=max_pool_stride)
        else:
            out = conv2D_block(inputA=inputB, num_filters=num_filters, drop_out=drop_out, sample_shape=input_dim,
                               GAP=setGap,
                               act_regularization=act_regularization, conv_kernel_size=conv_kernel_size,
                               conv_stride=conv_stride, max_pool_size=max_pool_size, max_pool_stride=max_pool_stride)

    if dense_tail:
        out = Dense(32, activation='relu', kernel_initializer='he_uniform',
                    activity_regularizer=l1(act_regularization))(out)

    if gap_tail:
        out = GlobalAveragePooling2D()(out)

    model = Model(inputs=inputB, outputs=out)

    return model



