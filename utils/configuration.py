#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:00:28 2018

@author: jsaavedr
"""
from configparser import SafeConfigParser

class ConfigurationFile:
    """
     An instance of ConfigurationFile contains required parameters to train a 
     convolutional neural network
    """    
    def __init__(self, str_config, arch):
        config = SafeConfigParser()
        config.read(str_config)
        self.sections = config.sections()
        print (self.sections)
        if arch in self.sections:
            self.number_of_classes = config.getint(arch,"NUM_CLASSES")
            self.number_of_iterations= config.getint(arch,"NUM_ITERATIONS")
            self.dataset_size = config.getint(arch, "DATASET_SIZE")
            self.test_size = config.getint(arch, "TEST_SIZE")
            self.batch_size = config.getint(arch, "BATCH_SIZE")
            self.estimated_number_of_batches =  int ( float(self.dataset_size) / float(self.batch_size) )
            self.estimated_number_of_batches_test = int ( float(self.test_size) / float(self.batch_size) )
            self.snapshot_time = config.getint(arch, "SNAPSHOT_TIME")
            self.test_time = config.getint(arch, "TEST_TIME")
            self.lr = config.getfloat(arch, "LEARNING_RATE")
            self.snapshot_prefix = config.get(arch, "SNAPSHOT_PREFIX")
            self.number_of_epochs = int ( float(self.number_of_iterations) / float(self.estimated_number_of_batches) )
            self.data_dir = config.get(arch,"DATA_DIR")
        else:
            raise ValueError(" {} is not a valid section".format(arch))
    
    def getNumnberOfClasses(self) :
        return self.number_of_classes
    
    def getNumberOfIterations(self):
        return self.number_of_iterations
    
    def getNumberOfEpochs(self):
        return self.number_of_epochs
    
    def getDatasetSize(self):
        return self.dataset_size
    
    def getBatchSize(self):
        return self.batch_size
    
    def getNumberOfBatches(self):
        return self.estimated_number_of_batches
    
    def getNumberOfBatchesForTest(self):
        return self.estimated_number_of_batches_test
    
    def getSnapshotTime(self):
        return self.snapshot_time
    
    def getTestTime(self):
        return self.test_time
    
    def getSnapshotPrefix(self):
        return self.snapshot_prefix
    
    def getDataDir(self):
        return self.data_dir
    
    def getLearningRate(self):
        return self.lr
    
    def isAValidSection(self, str_section):
        return str_section in self.sections
    
    def show(self):
        print("NUM_ITERATIONS: {}".format(self.getNumberOfIterations()))
        print("DATASET_SIZE: {}".format(self.getDatasetSize()))
        print("LEARNING_RATE: {}".format(self.getLearningRate()))
        print("NUMBER_OF_BATCHES: {}".format(self.getNumberOfBatches()))
        print("DATA_DIR: {}".format(self.getDataDir()))