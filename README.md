# Dataset and Estimator Python 3.6
1. Preparing our data <p>
Prepare the data in a appropriate format. In this step we will use tfRecords as proposed by Tensorflow. In this example  you can use: <p>
**create_mnist_data.py**  *-pathname* [path to the data] <p>
                      *-type* [0: train, 1: test, 2: both] <p>
                      *-imsize* [size of target images] <p>
                      *-from_binary* [use it if the data is loaded from the binary source] <p>                        
The source code used for this functionality can be found in utils/data.py <p>
2. Training 
**train_mnist_with_estimator.py**  
                      *-mode* ['test' | 'train'] <p>
                      *-device* ['cpu' | 'gpu'] <p>
                      *-arch* [name of the architecture to be used, it should be found in the configuration file] <p>
                      *-ckpt* [optional, a checkpoint file used for fine-tuning] <p>                        
**Configurarion file [configuration.config]** <p>
A file containing the following info: <p>
[MNIST] <p>
NUM_ITERATIONS = 10000 <p>
NUM_CLASSES = 10 <p>
DATASET_SIZE = 60000 <p>
TEST_SIZE = 10000 <p>
BATCH_SIZE = 100 <p>
SNAPSHOT_TIME = 1000 <p>
TEST_TIME = 60 <p>
LEARNING_RATE = 0.0001 <p>
SNAPSHOT_PREFIX = ./trained_mnist <p>
DATA_DIR = /home/vision/smb-datasets/MNIST/ [need to be replaced]
  
2. Prediction <p>
**demo_mnist.py**  -image /home/vision/smb-datasets/MNIST/Test/digit_mnist_09989_5.png -device cpu -arch MNIST -ckpt model/model.ckpt-10000 <p>
You can get deep-features if line 89 is uncommented
# Dependencies
  * Tensorflow  >1.8 
  * Opencv 3.4.1
