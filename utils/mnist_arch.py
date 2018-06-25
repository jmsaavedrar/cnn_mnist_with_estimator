#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:38:03 2018

@author: Jose M. Saavedra
"""

import tensorflow as tf
from . import layers

# A net for mnist classification
# features: containing feature vectors to be trained
# input_shape: [height, width]
# n_classes int
# is_training: boolean [it should be True for training and False for testing]
def net_fn(features, input_shape, n_classes, is_training = True):    
    with tf.variable_scope("model_scope"):            
        x_tensor = tf.reshape(features, [-1, input_shape[0], input_shape[1],  1 ] )                
        conv_1 = layers.conv_layer(x_tensor, shape = [3, 3, 1, 32], name='conv_1'); 
        conv_1 = layers.max_pool_layer(conv_1, 3, 2) # 14 x 14
        print(" conv_1: {} ".format(conv_1.get_shape().as_list()))
        #conv_2
        conv_2 = layers.conv_layer(conv_1, shape = [3, 3, 32, 64], name = 'conv_2')
        conv_2 = layers.max_pool_layer(conv_2, 3, 2) # 7 x 7
        print(" conv_2: {} ".format(conv_2.get_shape().as_list()))
        #conv_3
        conv_3 = layers.conv_layer(conv_2, shape = [3, 3, 64, 64], name = 'conv_3')
        conv_3 = layers.max_pool_layer(conv_3, 3, 2) # 3 x 3
        print(" conv_3: {} ".format(conv_3.get_shape().as_list()))    
        #fully connected
        fc6 = layers.fc_layer(conv_3, 250, name = 'fc6')
        print(" fc6: {} ".format(fc6.get_shape().as_list()))
        #fully connected
        fc7 = layers.fc_layer(fc6, n_classes, name = 'fc7', use_relu = False)
        print(" fc7: {} ".format(fc7.get_shape().as_list()))    
        
    return {"output": fc7, "deep_feature": fc6}