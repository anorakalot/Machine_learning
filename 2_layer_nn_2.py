from numpy import exp, array, random, dot

class NeuronLayer(object):
#gives a set of weights for each layer
    def __init__(self,number_of_neurons,number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron,number_of_neurons)) -1


#a neural network class with all the necessary functions
class NeuralNetwork(object):
    #init the class layer objects to be the paramter objects
    def __init__(self,layer1,layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    #the sigmoiod function in order to calculate the neuron output
    def __sigmoid(self,x):
        return 1 / (1+exp(-x))

    #the derivative of the sigmoid function in order to help with weight adjustment calculations
    def __sigmoid_derivative(self,x):
        return x * (1-x)



    #the training functions used for training the neural_network (meaning adjusting the weights for both layers)
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):

            #get output from the think function
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            #calculates error based off difference from predetermined training_set_output
            layer2_error = training_set_outputs - output_from_layer_2

            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)


            #dont know why its layer2 delta dot product with layer 2 weights for first layer error
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)

            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            #finish the adjustment calculations by also multiplying by the inputs for each layer
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            #adjust the weights based on the calculated adjustment
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment




    #passes each layer through the sigmoid function in order to get an output
    #notice that output_from_layer_2 utilizes output_from_layer_1 instead of inputs
    def think(self,inputs):
        output_from_layer_1 = self.__sigmoid(dot(inputs,self.layer1.synaptic_weights))
        output_from_layer_2 = self.__sigmoid(dot(output_from_layer_1,self.layer2.synaptic_weights))
        return output_from_layer_1,output_from_layer_2

    #prints the current layer weights
    def print_weights(self):
        print "\n"
        print " Layer 1: 4 neurons each with 3 inputs"
        print self.layer1.synaptic_weights
        print "\n"
        print " Layer 2: (1, neuron, with 4 inputs):"
        print "\n"
        print self.layer2.synaptic_weights
        print "\n"




if __name__ == "__main__":

    random.seed(1)

    layer1 = NeuronLayer(4,3)

    layer2 = NeuronLayer(1,4)

    neural_network = NeuralNetwork(layer1,layer2)

    print "Stage 1) Starting snaptic weights"

    neural_network.print_weights()

    training_set_inputs = array([[0,0,1],[0,1,1],[1,0,1],[0,1,0],[1,0,0],[1,1,1],[0,0,0]])

    training_set_outputs = array([[0,1,1,1,1,0,0]]).T

    neural_network.train(training_set_inputs, training_set_outputs,60000)

    print "Stage 2) new weights after training: "
    neural_network.print_weights()

    print "Stage 3) Considering new input [1,1,0] after weight training"
    hidden_output_1, output = neural_network.think(array([1,0,0]))
    print output
