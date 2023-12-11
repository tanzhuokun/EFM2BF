

## model hyper parameters


data_type = 'brain'

input_dim = 22325


hidden_dim = 3000
'''
coexpression.npz  4224


'''

# the hidden dimension for consensus channel
common_dim = 128


# number of iterations
num_epoch = 20

# layer aggregation function: max mean concat none
layer_agg = 'mean'

# number of layers
num_layer = 3

# learning rate
learning_rate = 0.005

# dropout rate
dropout_rate = 0.1

# balance factor
beta = 1.0

# nonlinear activation function for multi-gate module: none relu leakyrelu tanh
act_mg = 'relu'


# train patience for early stop
patience = 25
