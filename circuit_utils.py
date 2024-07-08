import torch
import numpy as np
import itertools

# In this circuit, "0" is represented by -1, "1" is represented by "1"
# Thus, X AND Y = X * Y

def split_layers(rollout, nodes, layers):
    layer_np = np.zeros((layers, 2, nodes), dtype=np.int32)
    for i in range(layers):
        layer_np[i,0,:] = rollout[i*2*nodes:(2*i+1)*nodes]
        layer_np[i,1,:] = rollout[(i*2+1)*nodes:(2*i+2)*nodes]
    return layer_np

def layer_prop(input_vect, layer):
    expanded_input = np.concatenate([input_vect, input_vect*-1, np.array(input_vect.shape[0] * [[-1,1]])], axis=1)
    input_group_1 = expanded_input[:,layer[0]]
    input_group_2 = expanded_input[:,layer[1]]
    return input_group_1 * input_group_2

def circuit_prop(input_vect, rollout, nodes, layers):
    layer_np = split_layers(rollout, nodes, layers)
    layer_result = input_vect
    for i in range(layers):
        layer_result = layer_prop(input_vect, layer_np[i])
    return layer_result

def input_generator(n_inputs, nodes):
    assert n_inputs <= nodes
    gen_input = np.array(list(itertools.product([1, -1], repeat=n_inputs)))
    padded_input = np.ones((gen_input.shape[0], nodes)) * -1
    padded_input[:,:n_inputs] = gen_input
    return padded_input

def bin_to_dec(num):
    # shape of num: N * n_bits
    num = (num+1)/2 # -1 --> 0, 1 --> 1
    n_bits = num.shape[1]
    two_p = np.array([pow(2, x) for x in range(n_bits)])
    return (two_p * num).sum(axis=1)


if __name__ == "__main__":
    nodes = 10
    layers = 3
    n_inputs = 4
    rollout = [19, 19, 16, 2, 18, 6, 4, 1, 14, 18, 21, 9, 13, 9, 13, 14, 6, 18, 20, 13, 17, 0, 18, 16, 0, 15, 3, 21, 11, 11, 4, 0, 11, 6, 0, 11, 10, 19, 12, 7, 0, 8, 13, 1, 8, 11, 10, 20, 11, 19, 7, 16, 8, 1, 5, 10, 10, 11, 13, 12]
    # input_vect = np.array([[1] * 10]*N)
    input_vect = input_generator(n_inputs, nodes)
    print(circuit_prop(input_vect, rollout, nodes, layers))
   