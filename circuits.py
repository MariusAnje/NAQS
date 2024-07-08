from circuit_utils import circuit_prop, input_generator, bin_to_dec
import numpy as np

class Circuit():

    def __init__(self, n_inputs, n_split, nodes, layers, random=False) -> None:
        self.n_inputs = n_inputs
        self.n_split = n_split
        self.nodes = nodes
        self.layers = layers
        self.input_vect = input_generator(n_inputs, nodes)
    
    def mse_reward(self, output_num):
        return - np.sqrt(((output_num - self.ground_truth)**2).sum()) / len(output_num)
    
    def prop(self, rollout):
        return circuit_prop(self.input_vect, rollout, self.nodes, self.layers)
    
    def rollout_to_reward(self, rollout):
        output = self.prop(rollout)
        output = self.output_process(output)
        output_num = bin_to_dec(output)
        mse_reward = self.mse_reward(output_num)
        return [mse_reward]


class Multiplier(Circuit):
    def __init__(self, N, n_split, nodes, layers, random=False) -> None:
        self.N = N
        n_inputs = N * 2
        super().__init__(n_inputs, n_split, nodes, layers, random)
        self.gen_ground_truth()
        
    
    def gen_ground_truth(self):
        op1 = self.input_vect[:, :self.N]
        op2 = self.input_vect[:, self.N:]
        op1 = bin_to_dec(op1)
        op2 = bin_to_dec(op2)
        self.ground_truth = op1 * op2
    
    def output_process(self, output):
        return output[:, :self.N*2]