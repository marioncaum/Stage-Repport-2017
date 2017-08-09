import sys
import theano
from pylearn2.train import Train
from pylearn2.space import Conv2DSpace
from pylearn2.datasets.mnist import MNIST
from pylearn2.costs.mlp import WeightDecay
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.cost import SumOfCosts, MethodCost
from pylearn2.models.mlp import MLP, ConvRectifiedLinear, Softmax
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.termination_criteria import And, MonitorBased, EpochCounter
from pylearn2.training_algorithms.learning_rule import MomentumAdjustor, Momentum

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
h2 = ConvRectifiedLinear(layer_name = 'h2',
						 output_channels = 64,
						 irange = .05,
						 kernel_shape = [5, 5],
						 pool_shape = [4, 4],
						 pool_stride = [2, 2],
						 max_kernel_norm = 1.9365)

# Building the second hidden layer
h3 = ConvRectifiedLinear(layer_name = 'h3',
						 output_channels = 64,
						 irange = .05,
						 kernel_shape = [5, 5],
						 pool_shape = [4, 4],
						 pool_stride = [2, 2],
						 max_kernel_norm = 1.9365)

# Building the output layer
y = Softmax (max_col_norm = 1.9365,
             layer_name = 'y',
             n_classes = 10,
             istdev = .05)

# Building the model
model = MLP(batch_size = 100,
			input_space = Conv2DSpace(shape = [28, 28],
									  num_channels = 1),
			layers = [h2, h3, y])

# Defining algorithm properties
algo = SGD(batch_size = 100,
		   learning_rate = .01,
		   learning_rule = Momentum(init_momentum = .5),
		   monitoring_dataset = ds,
		   cost = SumOfCosts(costs = [MethodCost(method = 'cost_from_X'),
									  WeightDecay(coeffs = [.00005, .00005, .00005])]),
           termination_criterion = And(criteria = [MonitorBased(channel_name = "valid_y_misclass",
																prop_decrease = 0.50,
																N = 10),
												   EpochCounter(max_epochs = 10000)]))

# Pull it all together
train = Train(dataset = train_set,
			  model = model,
			  algorithm = algo,
			  extensions = [MonitorBasedSaveBest(channel_name = 'valid_y_misclass',
												 save_path = "convolutional_network_best.pkl"),
							MomentumAdjustor(start = 1,
											 saturate = 10,
											 final_momentum = .99)])

# Training the model
train.main_loop()
