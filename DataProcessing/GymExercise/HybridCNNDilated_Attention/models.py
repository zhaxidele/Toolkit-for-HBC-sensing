
#%%
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Conv1D, Conv2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Flatten 
from tensorflow.keras.layers import Add, Concatenate, Lambda, Input, Permute
from tensorflow.keras.regularizers import L2
from tensorflow.keras.constraints import max_norm

from tensorflow.keras import backend as K

from attention_models import attention_block


def Post_Fusion(n_classes, in_chans=7, in_samples=80, n_windows=4, F1=16, D=2, kernelSize=64, dropout=0.1, di_kernelSize=4, di_filters=32, di_dropout=0.3, di_activation='elu'):

    input_1= Input(shape=(1, in_samples, in_chans))
    print("input_1: ", K.shape(input_1))

    input_1_cap_slicing_layer = tf.expand_dims(Lambda(lambda x: x[:, :, :, -1], name='slice_cap')(input_1), axis=-1)
    input_1_imu_slicing_layer = Lambda(lambda x: x[:, :, :, 0:6], name='slice_imu')(input_1)

    print("input_1_cap_slicing_layer: ", K.shape(input_1_cap_slicing_layer))
    print("input_1_imu_slicing_layer: ", K.shape(input_1_imu_slicing_layer))

    input_2_cap = Permute((2, 3, 1))(input_1_cap_slicing_layer)
    input_2_imu = Permute((2, 3, 1))(input_1_imu_slicing_layer)

    #input_2 = Permute((2, 3, 1))(input_1)

    dense_weightDecay = 0.5
    conv_weightDecay = 0.009
    conv_maxNorm = 0.8

    numFilters = F1
    F2 = numFilters * D

    block1_cap = Conv_block_(input_layer=input_2_cap, F1=F1, D=D, kernLength=kernelSize,
                         weightDecay=conv_weightDecay, maxNorm=conv_maxNorm, in_chans=1, dropout=dropout)
    block1_imu = Conv_block_(input_layer=input_2_imu, F1=F1, D=D, kernLength=kernelSize,
                         weightDecay=conv_weightDecay, maxNorm=conv_maxNorm, in_chans=6, dropout=dropout)

    block1 = Concatenate(axis=-1)([block1_imu, block1_cap])
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)

    # Sliding window
    sw_concat = []  # to store concatenated or averaged sliding window outputs
    for i in range(n_windows):
        st = i
        end = block1.shape[1] - n_windows + i + 1
        block2 = block1[:, st:end, :]

        # Attention_model
        block2 = attention_block(block2, attention)

        # Temporal convolutional network (TCN)
        block3 = DI_block_(input_layer=block2, input_dimension=F2,
                            kernel_size=di_kernelSize, filters=di_filters,
                            weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                            dropout=di_dropout, activation=di_activation)
        # Get feature maps of the last sequence
        block3 = Lambda(lambda x: x[:, -1, :])(block3)

        # Outputs of sliding window: Average_after_dense
        sw_concat.append(Dense(n_classes, kernel_regularizer=L2(dense_weightDecay))(block3))

    sw_concat = tf.keras.layers.Average()(sw_concat[:])
    out = Activation('softmax', name='softmax')(sw_concat)
    return Model(inputs=input_1, outputs=out)


def Conv_block_(input_layer, F1=4, kernLength=64, D=2, in_chans=22, weightDecay = 0.009, maxNorm = 0.6, dropout=0.1):

    F2= F1*D
    block1 = Conv2D(F1, (kernLength, 1), padding = 'same', data_format='channels_last', 
                    kernel_regularizer=L2(weightDecay),
                    # In a Conv2D layer with data_format="channels_last", the weight tensor has shape 
                    # (rows, cols, input_depth, output_depth), set axis to [0, 1, 2] to constrain 
                    # the weights of each filter tensor of size (rows, cols, input_depth).
                    kernel_constraint = max_norm(maxNorm, axis=[0,1,2]),
                    use_bias = True)(input_layer)
    block1 = LayerNormalization()(block1)
    block1 = Activation('elu')(block1)

    block2 = DepthwiseConv2D((1, in_chans),  
                             depth_multiplier = D,
                             data_format='channels_last',
                             depthwise_regularizer=L2(weightDecay),
                             depthwise_constraint  = max_norm(maxNorm, axis=[0,1,2]),
                             use_bias = True)(block1)
    block2 = LayerNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = Dropout(dropout)(block2)
    
    block3 = Conv2D(F2, (10, 1),
                            data_format='channels_last',
                            kernel_regularizer=L2(weightDecay),
                            kernel_constraint = max_norm(maxNorm, axis=[0,1,2]),
                            use_bias = True, padding = 'same')(block2)
    block3 = LayerNormalization()(block3)
    block3 = Activation('elu')(block3)
    block3 = Dropout(dropout)(block3)
    return block3

def DI_block_(input_layer,input_dimension,kernel_size,filters, dropout, weightDecay = 0.009, maxNorm = 0.6, activation='relu'):
    
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint = max_norm(maxNorm, axis=[0,1]),
                    padding = 'causal',kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=1,activation='linear',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint = max_norm(maxNorm, axis=[0,1]),
                    padding = 'causal',kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if(input_dimension != filters):
        conv = Conv1D(filters,kernel_size=1,
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint = max_norm(maxNorm, axis=[0,1]),
                    padding='same')(input_layer)
        added = Add()([block,conv])
    else:
        added = Add()([block,input_layer])
    out = Activation(activation)(added)

    block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=2**(i+1),activation='linear',
                kernel_regularizer=L2(weightDecay),
                kernel_constraint = max_norm(maxNorm, axis=[0,1]),
               padding = 'causal',kernel_initializer='he_uniform')(out)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=2**(i+1),activation='linear',
                kernel_regularizer=L2(weightDecay),
                kernel_constraint = max_norm(maxNorm, axis=[0,1]),
                padding = 'causal',kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    added = Add()([block, out])
    out = Activation(activation)(added)
        
    return out
