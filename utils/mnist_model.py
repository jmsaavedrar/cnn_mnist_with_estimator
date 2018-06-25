#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 11:29:43 2018

@author: Jose M. Saavedra
"""

import tensorflow as tf
from . import mnist_arch as arch
import os
        
def initializedModel(model_dir) :
    return os.path.exists(model_dir + '/init.init')    
        
#create a file indicating that init was done
def saveInitializationIndicator(model_dir) :
    with open(model_dir + '/init.init') as f :
        f.write('1')    

#defining a model that feeds the Estimator
def model_fn (features, labels, mode, params):    
    if mode == tf.estimator.ModeKeys.TRAIN:
        is_training = True
    else :
        is_training = False
    #creating the net to be used by Estimator
    net = arch.net_fn(features, params['image_shape'], params['number_of_classes'], is_training)        
    #    
    train_net = net["output"]
    #---------------------------------------
    idx_predicted_class = tf.argmax(train_net, 1)            
    #--------------------------------------    
    # If prediction mode, 
    predictions = { "idx_predicted_class": idx_predicted_class,
                    "predicted_probabilities": tf.nn.softmax(train_net, name="pred_probs"),
                    "deep_feature" : net["deep_feature"]
                   }
    if mode == tf.estimator.ModeKeys.PREDICT:
        estim_specs = tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else : # TRAIN or EVAL
        #initialization is carried out if params[ckpt] is defined. This is used for fine-tuning
        if mode == tf.estimator.ModeKeys.TRAIN and not initializedModel(params['model_dir']):             
            variables = tf.trainable_variables()
            if 'ckpt' in params:
                if params['ckpt'] is not None:
                    print('---Loading checkpoint : ' + params['ckpt'])
                    tf.train.init_from_checkpoint(ckpt_dir_or_file = params['ckpt'],
                                                  assignment_map = { v.name.replace('model_scope/', '').split(':')[0]  :  v for v in variables })
                    #save the indicator file
                    saveInitializationIndicator(params['model_dir'])
        #-----------------------------------------------------------------------
        idx_true_class = tf.argmax(labels, 1)            
        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=idx_true_class, predictions=idx_predicted_class)    
        # Define loss - e.g. cross_entropy -> mean(cross_entropy x batch)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = train_net, labels = labels)
        loss = tf.reduce_mean(cross_entropy)   
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # in order to allow updating in batch_normalization
        with tf.control_dependencies(update_ops) :
            optimizer = tf.train.AdamOptimizer(learning_rate= params['learning_rate'])
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())        
        #EstimatorSpec 
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=idx_predicted_class,
            loss=loss,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})    
    
    return  estim_specs