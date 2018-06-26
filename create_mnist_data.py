"""
Created on Mon Apr 16 2018
@author: jsaavedr

Description: prepare your data for training and testing
"""

import utils.data as data
import argparse



if __name__ == '__main__':        
    #pathname = "/home/vision/smb-datasets/MNIST/source"
    parser = argparse.ArgumentParser(description = "Create a dataset for training an testing")
    """ pathname should include train.txt and test.txt, files that should declare the data that will be processed"""
    parser.add_argument("-pathname", type = str, help = "<string> path where the data is stored" , required = True)
    parser.add_argument("-type", type = int, help = "<int> 0: only train, 1: only test, 2: both", required = True )
    parser.add_argument("-imsize", type = int, help = "<int> size of the image", required = True)
    parser.add_argument("-from_binary", help = "read from binary source", required = False, action = "store_true", default = False)
    pargs = parser.parse_args()            
    
    data.createMnistTFRecord(pargs.pathname, pargs.type, pargs.imsize, pargs.from_binary)
    print("tfrecords created for sk " + pargs.pathname)
    
    
    