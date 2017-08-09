import sys 
import theano
from pylearn2.train import Train
from pylearn2.datasets.mnist import MNIST
from pylearn2.training_algorithms.bgd import BGD
from pylearn2.models.mlp import MLP, Sigmoid, Softmax
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.termination_criteria import And, MonitorBased, EpochCounter

# Importing the format in which the model will be trained
theano.config.floatX = sys.argv[1]

# Loading the MNIST dataset
train_set = MNIST(which_set = 'train',
				  start = 0,
				  stop = 50000)

valid_set = MNIST(which_set = 'train',
                  start = 50000,
                  stop = 60000)

test_set = MNIST(which_set = 'test')
                          
ds = {"train" : train_set,
	  "valid" : valid_set,
	  "test" : test_set}

# Building the first hidden layer
h0 = Sigmoid(layer_name = 'h0',
             dim = int(sys.argv[2]),
             sparse_init = 15)

# Building the output layer
y = Softmax(layer_name = 'y',
            n_classes = 10,
            irange = 0.)

# Building the model
model = MLP(layers = [h0, y],
			nvis = 784)

# Defining algorithm properties
algo = BGD(batch_size = 10000,
           line_search_mode = 'exhaustive',
           conjugate = 1,
           updates_per_batch = 10,
           monitoring_dataset = ds,
           termination_criterion = And(criteria = [MonitorBased(channel_name = "valid_y_misclass"),
												   EpochCounter(max_epochs = 10000)]))

# Pull it all together
train = Train(dataset = train_set,
			  model = model,
			  algorithm = algo,
			  extensions = [MonitorBasedSaveBest(channel_name = 'valid_y_misclass',
												 save_path = "mlp_best.pkl")])

# Training the model
train.main_loop()
