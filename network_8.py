import sys
import theano
from pylearn2.train import Train
from pylearn2.models.maxout import Maxout
from pylearn2.datasets.mnist import MNIST
from pylearn2.models.mlp import MLP, Softmax
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.termination_criteria import MonitorBased
from pylearn2.training_algorithms.sgd import SGD, ExponentialDecay
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.training_algorithms.learning_rule import Momentum, MomentumAdjustor

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
	
ds = {'train' : train_set,
	  'valid' : valid_set,
	  'test' : test_set}
	
# Building the first hidden layer
h0 = Maxout(layer_name = 'h0',
			num_units = 240,
			num_pieces = 5,
			irange = .005,
			max_col_norm = 1.9365)
	
# Building the second hidden layer
h1 = Maxout(layer_name = 'h1',
			num_units = 240,
			num_pieces = 5,
			irange = .005,
			max_col_norm = 1.9365)
	
# Building the output layer
y = Softmax(max_col_norm = 1.9365,
			layer_name = 'y',
			n_classes = 10,
			irange = .005)
	
# Building the model
model = MLP(layers = [h0, h1, y],
			nvis = 784)
	
# Defining algorithm properties
algo = SGD(batch_size = 100,
		   learning_rate = .1,
		   learning_rule = Momentum(init_momentum = .5),
		   monitoring_dataset = ds,
		   cost = Dropout(input_include_probs = { 'h0' : .8 },
						  input_scales = { 'h0': 1. }),
		   termination_criterion = MonitorBased(channel_name = "valid_y_misclass",
												prop_decrease = 0.,
												N = 100),
		   update_callbacks = ExponentialDecay(decay_factor = 1.000004,
											   min_lr = .000001))
											   
# Pull it all together
train = Train(dataset = train_set,
			  model = model,
			  algorithm = algo,
			  extensions = [MonitorBasedSaveBest(channel_name = 'valid_y_misclass',
												 save_path = "maxout_best.pkl"),
							MomentumAdjustor(start = 1,
											 saturate = 250,
											 final_momentum = .7)],
			  save_path = "maxout"+theano.config.floatX+".pkl",
			  save_freq = 1)
			  	
# Training the model
train.main_loop()
