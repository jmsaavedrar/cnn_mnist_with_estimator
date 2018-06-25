#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 2018
@author: jsaavedr

Description: A list of function to create tfrecords
"""

import os
import struct
import sys
import numpy as np
import random
import cv2
import tensorflow as tf

# %% int64 should be used for integer numeric values
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# %% byte should be used for string  | char data
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# %% float should be used for floating point data
def _float_feature(value):    
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

#load mnist from the binay files
def loadMNIST(pathname = ".",  dataset = "train", shuffle = True):
    if dataset == "train":
        fname_images = os.path.join(pathname, 'train-images.idx3-ubyte')
        fname_labels = os.path.join(pathname, 'train-labels.idx1-ubyte')
    elif dataset == "test" :
        fname_images = os.path.join(pathname, 't10k-images.idx3-ubyte')
        fname_labels = os.path.join(pathname, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("use loadMNIST with train | test")
        
    with open(fname_labels, 'rb') as f_lbl:
        magic, num  = struct.unpack(">II", f_lbl.read(8))
        labels = np.fromfile(f_lbl, dtype = np.uint8)
    
    with open(fname_images, 'rb') as f_img:
        magic, num, rows, cols = struct.unpack(">IIII", f_img.read(16))
        images = np.fromfile(f_img, dtype = np.uint8).reshape(num, rows, cols)
    
    if shuffle:
        inds = list(range(len(labels)))
        np.random.shuffle(inds)
        images = images[inds]        
        labels = labels[inds]    
    return images, labels
#%%
def readImage(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #print("***** {}".format(image.shape))    
    if not os.path.exists(filename):
        raise ValueError(filename + " does not exist!")
    return image

# %% read data from text files
def readDataFromTextFile(str_path, dataset = "train" , shuf = True):
    datafile = str_path
    datafile = os.path.join(str_path, dataset + ".txt")
    print(datafile)
    assert os.path.exists(datafile)        
    # reading data from files  
    with open(datafile) as file :        
        lines = [line.rstrip() for line in file]     
        if shuf:
            random.shuffle(lines)
        lines_ = [tuple(line.rstrip().split('\t'))  for line in lines ] 
        filenames, labels = zip(*lines_)
        labels = [int(label) for label in labels]
    return filenames, labels

#%%
def processSkImage(image, imsize):
    #resize uses format (w,h)
    image_out = cv2.resize(image, imsize)
    image_out[image_out < 200 ] = 1 # black is foreground
    image_out[image_out >= 200 ] = 0 # white is background
    image_out = cv2.morphologyEx(image_out, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
    #image_out = cv2.morphologyEx(image_out, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
    return image_out*255

def processMnistImage(image, imsize):
    #resize uses format (w,h)
    image_out = cv2.resize(image, imsize)
    #image_out[image_out < 200 ] = 1 # black is foreground
    #image_out[image_out >= 200 ] = 0 # white is background
    #image_out = cv2.morphologyEx(image_out, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
    #image_out = cv2.morphologyEx(image_out, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
    return image_out

#%%
#creating tfrecords
def createTFRecord(images, labels, image_shape, processFun, tfr_filename):
    h = image_shape[0]
    w = image_shape[1]    
    writer = tf.python_io.TFRecordWriter(tfr_filename)    
    assert len(images) == len(labels)
    mean_image = np.zeros([h,w], dtype=np.float32)
    for i in range(len(images)):        
        print("---{}".format(i))         
        image = processFun(images[i,:,:], (w, h))
        #print("{}label: {}".format(label[i]))
        #create a feature
        feature = {'train/label': _int64_feature(labels[i]), 
                   'train/image': _bytes_feature(tf.compat.as_bytes(image.tostring()))}
        #crate an example protocol buffer
        example = tf.train.Example(features = tf.train.Features(feature=feature))        
        #serialize to string an write on the file
        writer.write(example.SerializeToString())
        mean_image = mean_image + images[i, :, :] / len(images)

    mean_image = mean_image         
    #serialize mean_image
    writer.close()
    sys.stdout.flush()
    return mean_image

#%%
#creating tfrecords from a list of files
def createTFRecordFromList(filenames, labels, target_shape, processFun, tfr_filename):
    h = target_shape[0]
    w = target_shape[1]
    writer = tf.python_io.TFRecordWriter(tfr_filename)    
    assert len(filenames) == len(labels)
    mean_image = np.zeros([h,w], dtype=np.float32)
    number_of_classes = max(labels)
    for i in range(len(filenames)):        
        if i % 500 == 0:
            print("---{}".format(i))           
        image = readImage(filenames[i])
        image = processFun(image, (w, h))        
        #create a feature        
        feature = {'train/label': _int64_feature(labels[i]), 
                   'train/image': _bytes_feature(tf.compat.as_bytes(image.tostring()))}
        #crate an example protocol buffer
        example = tf.train.Example(features = tf.train.Features(feature=feature))        
        #serialize to string an write on the file
        writer.write(example.SerializeToString())
        mean_image = mean_image + image / len(filenames)            
    #serialize mean_image
    writer.close()
    sys.stdout.flush()
    return mean_image, number_of_classes
#%%
#create TFRecords for MNIST dataset
"""
    @str_path: path where data can be found
    @id_type: 0 [only train]
             1 [only test]
             2 [both] 
     @im_size: int
     @from_binary_source, when True data is loaded from the binary sources; 
                 when False, data is loaded from text files, requires images to be stored in the filesystem     
"""    
def createMnistTFRecord(str_path, id_type, im_size, from_binary_source = True):    
    image_shape = np.array([im_size, im_size])
    number_of_classes = 0 # deprecated, this variable is kept por compatibility    
    if ( id_type + 1 ) & 1 : # train
        tfr_filename = os.path.join(str_path, "train.tfrecords")
        if from_binary_source :
            images, labels = loadMNIST(str_path, dataset="train", shuffle = True) 
            mean_image = createTFRecord(images, labels, image_shape, processMnistImage, tfr_filename)
        else :
            filenames, labels = readDataFromTextFile(str_path, dataset="train", shuf = True)
            mean_image, _ = createTFRecordFromList(filenames, labels, image_shape, processMnistImage, tfr_filename)
        #saving mean image        
        print("train_record saved at {}.".format(tfr_filename))        
        mean_file = os.path.join(str_path, "mean.dat")
        print("mean_file {}".format(mean_image.shape))
        mean_image.tofile(mean_file)
        print("mean_file saved at {}.".format(mean_file))                          
    if ( id_type + 1 ) & 2 : # test
        tfr_filename = os.path.join(str_path, "test.tfrecords")    
        if from_binary_source :
            images, labels = loadMNIST(str_path, dataset="test", shuffle = True) 
            createTFRecord(images, labels, image_shape, processMnistImage, tfr_filename)
        else :
            filenames, labels = readDataFromTextFile(str_path, dataset="test", shuf = True)
            createTFRecordFromList(filenames, labels, image_shape, processMnistImage, tfr_filename)
        print("test_record saved at {}.".format(tfr_filename))
    else :
        raise ValueError("id_type is incorrect for createMnistTFRecord")
    #saving metadata file    
    metadata_array = np.append(image_shape, [number_of_classes])                        
    metadata_file = os.path.join(str_path, "metadata.dat")
    metadata_array.tofile(metadata_file)
    print("metadata_file saved at {}.".format(metadata_file))      
#%% 
""" id_type = 0: only train 
              1: only test    
              2: both                              
"""
def createSkTFRecord(str_path, id_type, im_size):    
    #saving metadata
    image_shape = np.array([im_size, im_size])
    number_of_classes_train = -1
    number_of_classes_test = -1
    #------------- creating train data
    if ( id_type + 1 ) & 1 : # train   ( 0 + 1 ) & 1  == 1 
        filenames, labels = readDataFromTextFile(str_path, dataset="train", shuf = True)    
        tfr_filename = os.path.join(str_path, "train.tfrecords")
        training_mean, number_of_classes_train = createTFRecordFromList(filenames, labels, image_shape, processSkImage, tfr_filename)
        print("train_record saved at {}.".format(tfr_filename))
        #saving training mean
        mean_file = os.path.join(str_path, "mean.dat")
        print("mean_file {}".format(training_mean.shape))
        training_mean.tofile(mean_file)
        print("mean_file saved at {}.".format(mean_file))  
    #-------------- creating test data    
    if ( id_type + 1 ) & 2 : # test ( 1 + 1 ) & 2  == 2
        filenames, labels = readDataFromTextFile(str_path, dataset="test", shuf = True)  
        tfr_filename = os.path.join(str_path, "test.tfrecords")
        _ , number_of_classes_test = createTFRecordFromList(filenames, labels, image_shape, processSkImage, tfr_filename)
        print("test_record saved at {}.".format(tfr_filename))    
        
    number_of_classes = -1
    if  number_of_classes_train == -1 or number_of_classes_test == -1:
        number_of_classes = max([number_of_classes_train, number_of_classes_test]) + 1
    elif number_of_classes_train == number_of_classes_test:
        number_of_classes = number_of_classes_test + 1
    else:
        raise ValueError("number of classes train vs test are incompatible!")    
    metadata_array = np.append(image_shape, [number_of_classes])                    
    #saving metadata file    
    metadata_file = os.path.join(str_path, "metadata.dat")
    metadata_array.tofile(metadata_file)
    print("metadata_file saved at {}.".format(metadata_file))      
# %% parser sk
#---------parser_tfrecord for mnist
def parser_tfrecord_sk(serialized_example, im_size, mean_img, number_of_classes):    
    features = tf.parse_example([serialized_example],
                                features={
                                        'train/image': tf.FixedLenFeature([], tf.string),
                                        'train/label': tf.FixedLenFeature([], tf.int64),
                                        })
    image = tf.decode_raw(features['train/image'], tf.uint8)    
    image = tf.reshape(image, [im_size, im_size])
    image = tf.cast(image, tf.float32) - tf.cast(tf.constant(mean_img), tf.float32)
    #image = image * 1.0 / 255.0    
    #one-hot 
    label = tf.one_hot(tf.cast(features['train/label'], tf.int32), number_of_classes)
    label = tf.reshape(label, [number_of_classes])
    label = tf.cast(label, tf.float32)
    return image, label

# %% parser sk
#---------parser_tfrecord for mnist
def parser_tfrecord_mnist(serialized_example, im_size, mean_img, number_of_classes):
    features = tf.parse_example([serialized_example],
                                features={
                                        'train/image': tf.FixedLenFeature([], tf.string),
                                        'train/label': tf.FixedLenFeature([], tf.int64),
                                        })
    image = tf.decode_raw(features['train/image'], tf.uint8)
    image = tf.reshape(image, [im_size, im_size])
    image = tf.cast(image, tf.float32) - tf.cast(tf.constant(mean_img), tf.float32)    
    
    label = tf.one_hot(tf.cast(features['train/label'], tf.int32), number_of_classes)
    label = tf.reshape(label, [number_of_classes])
    label = tf.cast(label, tf.float32)
    return image, label