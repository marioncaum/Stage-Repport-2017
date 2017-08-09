import sys
import theano
from pylearn2.train import Train
from pylearn2.datasets.mnist import MNIST
from pylearn2.costs.cost import SumOfCosts
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.mlp import Default, L1WeightDecay, WeightDecay
from pylearn2.models.mlp import MLP, Tanh, RectifiedLinear, Softmax
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.termination_criteria import And, MonitorBased, EpochCounter
from pylearn2.training_algorithms.learning_rule import MomentumAdjustor, Momentum

# Importing the format in which the model will be trained
theano.config.floatX = sys.argv[1]
theano.config.device = cuda

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
mlp3_h0 = Tanh(layer_name = 'h0',
			   dim = int(sys.argv[2]),
			   sparse_init = 15)

# Building the second hidden layer            
mlp3_h1 = RectifiedLinear(layer_name = 'h1',
						  dim = int(sys.argv[3]),
						  sparse_init = 2)

# Building the second Multilayer Perceptrons
mlp3 = MLP(layer_name = 'mlp3',
		   layers = [mlp3_h0, mlp3_h1])
 
# Building the third hidden layer
mlp2_h0 = Tanh(layer_name = 'h0',
               dim = int(sys.argv[4]),
               sparse_init = 15)

# Building the fourth hidden layer        
mlp2_h1 = RectifiedLinear(layer_name = 'h1',
                          dim = int(sys.argv[5]),
                          sparse_init = 2)

# Building the output layer
mlp2_y = Softmax(layer_name = 'y',
                 n_classes = 10,
                 irange = 0.)

# Building the third Multilayer Perceptrons
mlp2 = MLP(layer_name = 'mlp2',
		   layers = [mlp2_h0, mlp2_h1, mlp2_y])
		   
# Building the first Multilayer Perceptrons
mlp1 = MLP(layer_name = 'mlp1',
		   layers = [mlp3, mlp2])
    
# Building the model
model = MLP(layers = [mlp1],
			nvis = 784)

# Defining algorithm properties
algo = SGD(batch_size = 100,
           learning_rate = .01,
           monitoring_dataset = ds,
           cost = SumOfCosts(costs = [Default(), 
									  L1WeightDecay(coeffs = [[[.0005, .0005], [.00005, .00005, .00005]]]),
									  WeightDecay(coeffs = [[[.0005, .0005], [.00005, .00005, .00005]]])]),
           learning_rule = Momentum(init_momentum = .5),
		   termination_criterion = And(criteria = [MonitorBased(channel_name = "valid_mlp1_mlp2_y_misclass",
																prop_decrease = 0.,
																N = 10),
												   EpochCounter(max_epochs = 10000)]))

# Pull it all together
train = Train(dataset = train_set,
			  model = model,
			  algorithm = algo,
			  extensions = [MonitorBasedSaveBest(channel_name = 'valid_mlp1_mlp2_y_misclass',
												 save_path = "mlp_3_best.pkl"),
							MomentumAdjustor(start = 1,
											 saturate = 10,
											 final_momentum = .99)])
	
#Training the model										 
train.main_loop()
