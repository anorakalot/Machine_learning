from numpy import exp,array,random,dot

class NeuralNetwork(object):
    def __init__(self):

        random.seed()

        self.random_weights = 2 * random.random((3,1)) -1


#sigmoid function for output normalizes it between 0 and 1
    def __sigmoid(self,x):
        return 1/(1+exp(-x))

#find the sigmoid derivative for the error function
#how confident the output is (thing about sigmoid functions curves)
    def __sigmoid_derivative(self,x):
        return x * (1-x)



    def train(self,training_set_inputs,training_set_outputs,number_of_training_iterations):
        for step in range(number_of_training_iterations):
            #find the output
            output = self.think(training_set_inputs)

            #find the difference between output
            error = training_set_outputs - output

            #find the change needed by scaling by input, error and how confident the output was in its decision(smafller derivative
            #adjusted less) larger sigmoid derivative adjusted more
            adjustment = dot(training_set_inputs.T,error * self.__sigmoid_derivative(output))

            #adjust the weight by the adjustment
            self.random_weights += adjustment



#find the ouput using simoid function
    def think(self,inputs):
        return self.__sigmoid(dot(inputs,self.random_weights))






if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print ("Random starting weights")

    print (neural_network.random_weights)

    training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1],[1,0,0],[0,0,1]])
    training_set_outputs = array([[0,1,1,0,1,0]]).T

    neural_network.train(training_set_inputs,training_set_outputs,10000)

    print ("New weights after training: ")

    print (neural_network.random_weights)

    print ("Testing with new input! [1,0,0]")

    print (neural_network.think(array([1,0,0])))
