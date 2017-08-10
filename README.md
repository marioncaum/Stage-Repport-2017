# Stage-Repport-2017


## Requirements

* Theano 0.9
* Pylearn2 0.1 
* a CUDA capable GPU

## How to run networks_1_2_3_4.py ?
This program run well on CPU but is much faster when executed on GPU. To run this program on GPU, create a file named .theanorc and write the two following lines in this file :<br />
[global]<br />
device = cuda
### Comand Line
python networks_1_2_3_4.py [format][number of neurons of the first layer][number of neurons of the second layer][number of neurons of the third layer][number of neurons of the fourth layer]
### Format
There are 2 differents format : single precision floating point (float32) and double precision floating point (float64)
### Number of neurons
The number of neurons should be a strictly positive integers.
### Examples
python networks_1_2_3_4.py float32 5 20 30 40 <br />
python networks_1_2_3_4.py float64 5 20 30 40 <br />
python networks_1_2_3_4.py float32 10 40 60 80 <br />
python networks_1_2_3_4.py float64 10 40 60 80 <br />
python networks_1_2_3_4.py float32 50 200 300 400 <br />
python networks_1_2_3_4.py float64 50 200 300 400 <br />
python networks_1_2_3_4.py float32 500 2000 3000 4000 <br />
python networks_1_2_3_4.py float64 500 2000 3000 4000
### Results of training
The results of the different networks are saved in the file named mlp_3_best.pkl. To see the accuracy you can use the program named print_monitor.py that is located in the pylearn2/scripts/ repertory of pylearn2 repertory.

## How to run network_5.py ?
This program run well on CPU but is much faster when executed on GPU. To run this program on GPU, create a file named .theanorc and write the two following lines in this file :<br />
[global]<br />
device = cuda
### Comand Line
python network_5.py [format][number of neurons of the first layer]
### Format
There are 2 differents format : single precision floating point (float32) and double precision floating point (float64)
### Number of neurons
The number of neurons should be a strictly positive integers.
### Examples
python network_5.py float32 500 <br />
python network_5.py float64 500
### Results of training
The results of the different networks are saved in the file named mlp_best.pkl. To see the accuracy you can use the program named print_monitor.py that is located in the pylearn2/scripts/ repertory of pylearn2 repertory.

## How to run network_6.py ?
This program run well on CPU but is much faster when executed on GPU. To run this program on GPU, create a file named .theanorc and write the two following lines in this file :<br />
[global]<br />
device = cuda
### Comand Line
python network_6.py [format][number of neurons of the first layer][number of neurons of the second layer]
### Format
There are 2 differents format : single precision floating point (float32) and double precision floating point (float64)
### Number of neurons
The number of neurons should be a strictly positive integers.
### Examples
python network_6.py float32 500 1000 <br />
python network_6.py float64 500 1000
### Results of training
The results of the different networks are saved in the file named mlp_2_best.pkl. To see the accuracy you can use the program named print_monitor.py that is located in the pylearn2/scripts/ repertory of pylearn2 repertory.

## How to run network_7.py ?
This program run well on CPU but is much faster when executed on GPU. To run this program on GPU, create a file named .theanorc and write the two following lines in this file :<br />
[global]<br />
device = cuda
### Comand Line
python network_7.py [format][number of neurons of the first layer][number of neurons of the second layer]
### Format
There are 2 differents format : single precision floating point (float32) and double precision floating point (float64)
### Number of neurons
The number of neurons should be a strictly positive integers.
### Examples
python network_7.py float32 500 1000 <br />
python network_7.py float64 500 1000
### Results of training
The results of the different networks are saved in the file named mlp_3_best.pkl. To see the accuracy you can use the program named print_monitor.py that is located in the pylearn2/scripts/ repertory of pylearn2 repertory.

## How to run network_8.py ?
This program run well on CPU but is much faster when executed on GPU. To run this program on GPU, create a file named .theanorc and write the two following lines in this file :<br />
[global]<br />
device = cuda
### Comand Line
python network_8.py [format][number of neurons of the first layer][number of neurons of the second layer]
### Format
There are 2 differents format : single precision floating point (float32) and double precision floating point (float64)
### Number of neurons
The number of neurons should be a strictly positive integers.
### Examples
python network_8.py float32 500 1000 <br />
python network_8.py float64 500 1000
### Results of training
The results of the different networks are saved in the file named maxout_best.pkl. To see the accuracy you can use the program named print_monitor.py that is located in the pylearn2/scripts/ repertory of pylearn2 repertory.

## How to run network_9.py ?
This program have to be executed on GPU. To run this program on GPU, create a file named .theanorc and write the two following lines in this file :<br />
[global]<br />
device = gpu
### Comand Line
python network_9.py [format]
### Format
There are 2 differents format : single precision floating point (float32) and double precision floating point (float64). If you run this program with double precision floating point it will run on CPU, else it will run on GPU.
### Examples
python network_9.py float32 <br />
python network_9.py float64
### Results of training
The results of the different networks are saved in the file named convolutional_network_best.pkl. To see the accuracy you can use the program named print_monitor.py that is located in the pylearn2/scripts/ repertory of pylearn2 repertory.

