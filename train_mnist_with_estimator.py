#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:30:08 2018
@author: jose.saavedra

A convolutional neural network for sk_net
This uses Dataset and Estimator componentes from tensorflow

"""
import utils.data as data
import numpy as np
import os
import argparse
from utils.configuration import ConfigurationFile
import utils.mnist_model as mnistnet
import tensorflow as tf


#define the input function
def input_fn(filename, image_shape, mean_img, is_training, configuration):     
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(lambda x: data.parser_tfrecord_sk(x, image_shape[0], mean_img, configuration.getNumnberOfClasses()))
    dataset = dataset.batch(conf.getBatchSize())   
    if is_training:
        dataset = dataset.shuffle(configuration.getNumberOfBatches())
        dataset = dataset.repeat(configuration.getNumberOfEpochs())            
    # for testing shuffle and repeat are not required    
    return dataset
    
#-----------main----------------------
if __name__ == '__main__':            
    parser = argparse.ArgumentParser(description = "training / testing xk models")
    parser.add_argument("-mode", type=str, choices=['test', 'train'], help=" test | train ", required = True)
    parser.add_argument("-device", type=str, choices=['cpu', 'gpu'], help=" cpu | gpu ", required = True)
    parser.add_argument("-arch", type=str, help=" name of section in the configuration file", required = True)    
    parser.add_argument("-ckpt", type=str, help=" <optional>, it defines the checkpoint for training<fine tuning> or testing", required = False)    
    pargs = parser.parse_args() 
    conf = ConfigurationFile("sk_configuration.config", pargs.arch)      
    run_mode = pargs.mode
    device_name = "/" + pargs.device + ":0"
    #verifying if output path exists
    if not os.path.exists(os.path.dirname(conf.getSnapshotPrefix())) :
        os.makedirs(os.path.dirname(conf.getSnapshotPrefix()))        
    #metadata
    filename_mean = os.path.join(conf.getDataDir(), "mean.dat")
    metadata_file = os.path.join(conf.getDataDir(), "metadata.dat")
    #reading metadata    
    metadata_array = np.fromfile(metadata_file, dtype=np.int)
    image_shape = metadata_array[0:2]    
    number_of_classes = metadata_array[2]
    print(metadata_array)
    
    #load mean
    mean_img =np.fromfile(filename_mean, dtype=np.float64)
    mean_img = np.reshape(mean_img, image_shape.tolist())    
    #defining files for training and test
    filename_train = os.path.join(conf.getDataDir(), "train.tfrecords")
    filename_test = os.path.join(conf.getDataDir(), "test.tfrecords")
    
    #-using device gpu or cpu
    with tf.device(device_name):        
        estimator_config = tf.estimator.RunConfig(model_dir = conf.getSnapshotPrefix(),
                                                  save_checkpoints_steps=conf.getSnapshotTime(),
                                                  keep_checkpoint_max=10)
        
        classifier = tf.estimator.Estimator(model_fn = mnistnet.model_fn, 
                                            config = estimator_config,
                                            params = {'learning_rate' : conf.getLearningRate(),
                                                      'number_of_classes' : conf.getNumnberOfClasses(),
                                                      'image_shape' : image_shape,
                                                      'model_dir': conf.getSnapshotPrefix(),
                                                      'ckpt' : pargs.ckpt,
                                                      'arch' : pargs.arch
                                                      }
                                            )        
        #
        tf.logging.set_verbosity(tf.logging.INFO) # Just to have some logs to display for demonstration
        #training
        if run_mode == 'train':
            train_spec = tf.estimator.TrainSpec(input_fn = lambda: input_fn(filename_train, 
                                                                            image_shape, 
                                                                            mean_img, 
                                                                            is_training = True, 
                                                                            configuration =  conf),
                                                max_steps = conf.getNumberOfIterations())
            #max_steps is not usefule when inherited checkpoint is used   
            eval_spec = tf.estimator.EvalSpec(input_fn = lambda: input_fn(filename_test, 
                                                                          image_shape, 
                                                                          mean_img, 
                                                                          is_training = False, 
                                                                          configuration =  conf),
                                              start_delay_secs = conf.getTestTime(),
                                              throttle_secs = conf.getTestTime()*2)        
            #
            tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
        
        #testing    
        if run_mode == 'test' :
            result = classifier.evaluate(input_fn=lambda: input_fn(filename_test, image_shape, mean_img, is_training = False, configuration =  conf),
                                checkpoint_path = pargs.ckpt)
            print(result)
        
    print("ok")
        
    
    