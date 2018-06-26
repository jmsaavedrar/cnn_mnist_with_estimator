#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:30:08 2018
@author: jose.saavedra

A convolutional neural network for mnist

demo for mnist

"""

import utils.data as data
import numpy as np
import os
import argparse
from utils.configuration import ConfigurationFile
import utils.mnist_model as mnistnet
import tensorflow as tf


#-----------main

#define the input function
def input_fn(filename, image_shape, mean_img):
    image = data.readImage(filename)
    image = data.processMnistImage(image, (image_shape[1], image_shape[0]))
    image = image - mean_img
    image = image.astype(np.float32)
    image = np.reshape(image, [1, image_shape[0], image_shape[1]])
    return image



if __name__ == '__main__':            
    parser = argparse.ArgumentParser(description = "training / testing xk models")
    #parser.add_argument("-mode", type=str, choices=['test', 'train'], help=" test | train ", required = True)
    parser.add_argument("-image", type=str, help=" filename of image to be processed", required = True)
    parser.add_argument("-device", type=str, choices=['cpu', 'gpu'], help=" cpu | gpu ", required = True)
    parser.add_argument("-arch", type=str, help=" name of the architecture to train & test", required = True)
    parser.add_argument("-ckpt", type=str, help="  checkpoint", required = False)
    pargs = parser.parse_args()
    conf = ConfigurationFile("configuration.config", pargs.arch)
    #conf.show()            

    device_name = "/" + pargs.device + ":0"
    # verifying output path exists
    if not os.path.exists(os.path.dirname(conf.getSnapshotPrefix())) :
        os.makedirs(os.path.dirname(conf.getSnapshotPrefix()))

    print ("loading data [train and test] \n")
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
    #
    filename_train = os.path.join(conf.getDataDir(), "train.tfrecords")
    filename_test = os.path.join(conf.getDataDir(), "test.tfrecords")

    with tf.device(device_name):
        classifier = tf.estimator.Estimator(model_fn = mnistnet.model_fn,
                                            model_dir = os.path.dirname(pargs.ckpt),
                                            params = {'learning_rate' : 0,
                                                      'number_of_classes' : conf.getNumnberOfClasses(),
                                                      'image_shape' : image_shape
                                                      })
        #
        tf.logging.set_verbosity(tf.logging.INFO) # Just to have some logs to display for demonstration
        #training
        filename = pargs.image        
        input_image = input_fn(filename, image_shape, mean_img)
        print(input_image.shape)
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x=input_image,
                num_epochs=1,
                shuffle=False
                )

        predicted_result = list(classifier.predict(input_fn = predict_input_fn))
        for prediction in predicted_result:
            print("idx_predicted_class: {}".format(prediction["idx_predicted_class"]))
            deep_features = prediction["deep_feature"]
            #uncomment to deep-features be showed
            #print("deep_feature: {}".format(deep_features.shape))
            #print("deep-features")
            print("-------------")
            print(deep_features)

    print("ok")
