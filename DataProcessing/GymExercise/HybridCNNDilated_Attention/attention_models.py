
#%%
import math
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense
from tensorflow.keras.layers import multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.keras.layers import Dropout, MultiHeadAttention, LayerNormalization, Reshape
from tensorflow.keras import backend as K


#%% Create and apply the attention model
def attention_block(in_layer, attention_model, ratio=8, residual = False, apply_to_input=True): 
    in_sh = in_layer.shape # dimensions of the input tensor
    in_len = len(in_sh) 
    expanded_axis = 2 # defualt = 2
    
    if(in_len > 3):
        in_layer = Reshape((in_sh[1],-1))(in_layer)
    out_layer = mha_block(in_layer)
        
    if (in_len == 3 and len(out_layer.shape) == 4):
        out_layer = tf.squeeze(out_layer, expanded_axis)
    elif (in_len == 4 and len(out_layer.shape) == 3):
        out_layer = Reshape((in_sh[1], in_sh[2], in_sh[3]))(out_layer)
    return out_layer


#%% Multi-head self Attention (MHA) block
def mha_block(input_feature, key_dim=8, num_heads=4, dropout = 0.1):
    """Multi Head self Attention (MHA) block.
    Here we include two types of MHA blocks: 
            The original multi-head self-attention as described in https://arxiv.org/abs/1706.03762
            The multi-head local self attention as described in https://arxiv.org/abs/2112.13492v1
    """    
    # Layer normalization
    x = LayerNormalization(epsilon=1e-6)(input_feature)
    x = MultiHeadAttention(key_dim = key_dim, num_heads = num_heads, dropout = dropout)(x, x)
    x = Dropout(0.3)(x)
    # Skip connection
    mha_feature = Add()([input_feature, x])
    
    return mha_feature

