"""
File containing the Neural net stuff for bell training.
Let's just keep it all separate, because I know what I'm doing these days...
"""

import numpy as np
import os

class ForceNet():
    """
    Class for the neural net used for determining forces from the inputs.
    Need to be consistent with inputs really, have position and angular velocity as 0 and 1, but then might become a little messy. We'll see...
    """

    def __init__(self, n_nodes=10, n_inputs=2):
        #Generate arraysfor the weights and things. Then broadcast to these arrays so the dimensions don't get mixed up...
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.nparas = n_inputs*n_nodes + n_nodes*2 + 1

        self.weights_in = np.zeros((n_inputs, n_nodes))
        self.biases_in = np.zeros((n_nodes))
        self.weights_out = np.zeros((n_nodes))
        self.biases_out = np.zeros((1))
        self.parameter_set = np.zeros((self.nparas))

    def generate_random_seed(self, sigma=0.1):
        """
        Does a seed with Gaussian weightings of sigma
        """
        self.parameter_set = np.random.normal(scale=sigma, size = self.nparas)
        self.weights_in[:,:] = np.reshape(self.parameter_set[0:self.n_inputs*self.n_nodes], shape = np.shape(self.weights_in))
        self.biases_in[:] = self.parameter_set[self.n_inputs*self.n_nodes:self.n_inputs*self.n_nodes+self.n_nodes]
        self.weights_out[:] = self.parameter_set[self.n_inputs*self.n_nodes+self.n_nodes:self.n_inputs*self.n_nodes+self.n_nodes*2]
        self.biases_out[:] = self.parameter_set[self.n_inputs*self.n_nodes+self.n_nodes*2:self.n_inputs*self.n_nodes+self.n_nodes*2+1]

    def update_network(self, parameter_set):
        """
        For a given parameter set, updates the network biases etc.
        """
        self.parameter_set = parameter_set.copy()
        self.weights_in[:,:] = np.reshape(parameter_set[0:self.n_inputs*self.n_nodes], shape = np.shape(self.weights_in))
        self.biases_in[:] = parameter_set[self.n_inputs*self.n_nodes:self.n_inputs*self.n_nodes+self.n_nodes]
        self.weights_out[:] = parameter_set[self.n_inputs*self.n_nodes+self.n_nodes:self.n_inputs*self.n_nodes+self.n_nodes*2]
        self.biases_out[:] = parameter_set[self.n_inputs*self.n_nodes+self.n_nodes*2:self.n_inputs*self.n_nodes+self.n_nodes*2+1]

    def sigmoid(self, x):
        """
        Returns the sigmoid function of the input x
        """
        return 1.0/(1.0 + np.exp(-x))

    def tanh(self, x):
        """
        Returns the tanh function of the input x
        """
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def force(self, inputs):
        """
        For given input arrays, runs the neural net to find the expected output (number between 0 and 1)
        """
        inputs = np.array(inputs)
        node_activations = np.zeros(self.n_nodes)

        node_activations = np.sum(self.weights_in[:,:]*inputs[:, np.newaxis], axis=0) + self.biases_in[:]
        node_activations = np.clip(node_activations, a_min = -1e3, a_max = 1e3)  #Stop over and underflow in the exponentials
        node_activations = self.tanh(node_activations)
        output = np.sum(node_activations*self.weights_out[:]) + self.biases_out
        output = self.sigmoid(output)
        return output

    def save_current_state(self, mode, minimiser):
        """
        Appends to the filename the current ability of the net in question. SO can restart a training run with impunity etc.
        """
        fname = f'./nets/{mode}.txt'

        step = 0
        if os.path.exists(fname):
            with open(fname) as f:
                for line in f.readlines():
                    step += 1

        save_line = [step, minimiser, self.n_inputs, self.n_nodes] + self.parameter_set.tolist()
        with open(fname, "a") as f:
            f.write(" ".join(f"{x:.16f}" for x in save_line) + "\n")
        return

    def load_best_state(self, mode, override_nnodes=False, latest=False):

        fname = f'./nets/{mode}.txt'
        if not latest:
        #Determine the correct number of parameters for this best state
            if os.path.exists(fname):
                print('Using bespoke best state')
                best_score = 1e6; best_id = 0
                best_parameters = []
                cut = 10  #Only take the best from the last 250
                with open(fname, "r") as f:
                    data = f.readlines()
                    if len(data) > cut:
                        data = data[-cut:]

                    for li, line in enumerate(data[:]):
                        if float(line.split(' ')[1]) < best_score:
                            best_score = float(line.split(' ')[1])
                            best_ninputs = int(float(line.split(' ')[2]))
                            best_nnodes = int(float(line.split(' ')[3]))
                            best_id = li

                for val in data[best_id].split(' ')[4:]:
                    best_parameters.append(float(val))
                print('Best score', best_score)
            elif os.path.exists('./nets/default.txt'):
                print('Using default state')
                best_score = 1e6; best_id = 0
                best_parameters = []
                with open('./nets/default.txt', "r") as f:
                    data = f.readlines()
                    for li, line in enumerate(data[:]):
                        if float(line.split(' ')[1]) < best_score:
                            best_score = float(line.split(' ')[1])
                            best_ninputs = int(float(line.split(' ')[2]))
                            best_nnodes = int(float(line.split(' ')[3]))
                            best_id = li

                for val in data[best_id].split(' ')[4:]:
                    best_parameters.append(float(val))
            else:
                raise Exception('Log file not found...')
        else:
            if os.path.exists(fname):
                print('Using most recent state')
                best_score = 1e6; best_id = 0
                best_parameters = []
                with open(fname, "r") as f:
                    data = f.readlines()
                line = data[-1]
                best_score = float(line.split(' ')[1])
                best_ninputs = int(float(line.split(' ')[2]))
                best_nnodes = int(float(line.split(' ')[3]))

                for val in data[-1].split(' ')[4:]:
                    best_parameters.append(float(val))
            else:
                raise Exception('Log file not found...')

        target_nparas = len(best_parameters)

        if best_nnodes != self.n_nodes:
            print("Best parameters are not for the correct amount of nodes")
        if best_ninputs != self.n_inputs:
            print("Best parameters are not for the correct amount of inputs")

        if self.n_nodes == best_nnodes and self.n_inputs == best_ninputs:
            print('Intended network size matches the best one.')
            self.parameter_set[:] = np.array(best_parameters)
            self.weights_in[:,:] = np.reshape(self.parameter_set[0:self.n_inputs*self.n_nodes], shape = np.shape(self.weights_in))
            self.biases_in[:] = self.parameter_set[self.n_inputs*self.n_nodes:self.n_inputs*self.n_nodes+self.n_nodes]
            self.weights_out[:] = self.parameter_set[self.n_inputs*self.n_nodes+self.n_nodes:self.n_inputs*self.n_nodes+self.n_nodes*2]
            self.biases_out[:] = self.parameter_set[self.n_inputs*self.n_nodes+self.n_nodes*2:self.n_inputs*self.n_nodes+self.n_nodes*2+1]

        elif best_nnodes <= self.n_nodes and best_ninputs <= self.n_inputs and override_nnodes:
            print('Extending best solution to the larger model')
            parameter_set_in = np.array(best_parameters)

            self.weights_in[:best_ninputs,:best_nnodes] = np.reshape(parameter_set_in[0:best_ninputs*best_nnodes], shape = (best_ninputs, best_nnodes))
            self.biases_in[:best_nnodes] = parameter_set_in[best_ninputs*best_nnodes:best_ninputs*best_nnodes+best_nnodes]
            self.weights_out[:best_nnodes] = parameter_set_in[best_ninputs*best_nnodes+best_nnodes:best_ninputs*best_nnodes+best_nnodes*2]
            self.biases_out[:] = parameter_set_in[best_ninputs*best_nnodes+best_nnodes*2:best_ninputs*best_nnodes+best_nnodes*2+1]

            self.parameter_set[0:self.n_inputs*self.n_nodes] = np.reshape(self.weights_in, shape = (self.n_inputs*self.n_nodes))
            self.parameter_set[self.n_inputs*self.n_nodes:self.n_inputs*self.n_nodes+self.n_nodes] = self.biases_in[:]
            self.parameter_set[self.n_inputs*self.n_nodes+self.n_nodes:self.n_inputs*self.n_nodes+self.n_nodes+self.n_nodes] = self.weights_out[:]
            self.parameter_set[self.n_inputs*self.n_nodes+self.n_nodes*2:self.n_inputs*self.n_nodes+self.n_nodes*2+1] = self.biases_out[:]

        else:
            raise Exception("Cannot use best parameters. Sort it out.")

        return

    def load_latest_state(self, fname='net_state.txt'):
        """
        Loads the last state found in the log, not necessarily the best
        """
        if os.path.exists(fname):
            best_score = 1e6; best_id = 0
            best_parameters = []
            with open(fname, "r") as f:
                data = f.readlines()
                cut = len(data)
                line = data[-1]

            for val in data[-1].split(' ')[4:]:
                best_parameters.append(float(val))
            print('Latest score', best_score)
        else:
            raise Exception('Log file not found...')

        self.parameter_set[:] = np.array(best_parameters)
        self.weights_in[:,:] = np.reshape(self.parameter_set[0:self.n_inputs*self.n_nodes], shape = np.shape(self.weights_in))
        self.biases_in[:] = self.parameter_set[self.n_inputs*self.n_nodes:self.n_inputs*self.n_nodes+self.n_nodes]
        self.weights_out[:] = self.parameter_set[self.n_inputs*self.n_nodes+self.n_nodes:self.n_inputs*self.n_nodes+self.n_nodes*2]
        self.biases_out[:] = self.parameter_set[self.n_inputs*self.n_nodes+self.n_nodes*2:self.n_inputs*self.n_nodes+self.n_nodes*2+1]
        return

    def extend_net_dontuse(self, n_inputs_target, n_nodes_target):
        """
        Adds more neurons to an existing net, up to n_nodes_target of them.
        """
        nparas_new = n_inputs_target*n_nodes_target + n_nodes_target*2 + 1
        new_parameter_set = np.zeros(nparas_new)

        new_weights = np.zeros((n_inputs_target, n_nodes_target))
        new_weights[:,:self.n_nodes] = self.weights_in

        #Create a new LARGER parameter set. The cuts are tricky here.
        new_parameter_set[0:n_inputs_target*n_nodes_target] = np.reshape(new_weights, shape = (n_inputs_target*n_nodes_target))
        new_parameter_set[n_inputs_target*n_nodes_target:n_inputs_target*n_nodes_target+self.n_nodes] = self.biases_in[:]
        new_parameter_set[n_inputs_target*n_nodes_target+n_nodes_target:n_inputs_target*n_nodes_target+n_nodes_target+self.n_nodes] = self.weights_out[:]
        new_parameter_set[n_inputs_target*n_nodes_target+n_nodes_target*2:n_inputs_target*n_nodes_target+n_nodes_target*2+1] = self.biases_out[:]

        #Update Net metadata
        self.parameter_set = new_parameter_set
        self.n_nodes = n_nodes_target
        self.n_inputs = n_inputs_target
        self.nparas = self.n_inputs*self.n_nodes + self.n_nodes*2 + 1

        self.weights_in = np.zeros((self.n_inputs, n_nodes_target))
        self.biases_in = np.zeros((n_nodes_target))
        self.weights_out = np.zeros((n_nodes_target))
        self.biases_out = np.zeros((1))

        self.weights_in[:,:] = np.reshape(self.parameter_set[0:self.n_inputs*self.n_nodes], shape = (self.n_inputs, self.n_nodes))
        self.biases_in[:] = self.parameter_set[self.n_inputs*self.n_nodes:self.n_inputs*self.n_nodes+self.n_nodes]
        self.weights_out[:] = self.parameter_set[self.n_inputs*self.n_nodes+self.n_nodes:self.n_inputs*self.n_nodes+self.n_nodes*2]
        self.biases_out[:] = self.parameter_set[self.n_inputs*self.n_nodes+self.n_nodes*2:self.n_inputs*self.n_nodes+self.n_nodes*2+1]

        return
