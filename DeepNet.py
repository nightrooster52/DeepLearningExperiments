"""
rewrite neuralnet 01.py in an object oriented format
"""

import numpy as np 

def main():
    X = np.array([[0,0,1], 
                [0,1,1],
                [1,1,0],
                [1,1,1]])
    
    y = np.array([[0],
                 [0],
                 [0],
                 [1]])
    
    np.random.seed(1)
    
    network = DeepNet(X[0], [4], y[0])
    
    #network = generateDeepNet(X[0], [4], y[0])


    for i in range(60000):
        network.fullPass(X, y)
        
        if(i % 10000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output. 
            print("Epoch "+ str(i))
            print("Error : " + str(np.mean(np.abs(network.pass_error))))
#            print("outputs : " +  str(outputs) )
    print("output after training")
    print(network.activations[-1])

 
class DeepNet:
    
    def __init__(self, observation, layer_sizes, conclusion ):
        
        self.network = self.generateDeepNet(observation, layer_sizes, conclusion) 
        self.activations = []
        self.pass_error = []
        self.bias_nodes = 1 #per layer
        
        
        
    def fullPass(self, observations, conclusions):
        self.forwardPass(observations)
        self.backwardPass(conclusions)

    
    def generateDeepNet(self, observation, layer_sizes, conclusion):
        """
        Takes an observation ( array of attributes ) , layer_sizes ( array ) , and conclusion ( array of attributes ) 
        generates the appropriate network ( list of matrixes )
        """
        
        network = []
        bias_nodes = 1
        
        #+1 is for bias weights
        #connect input array to first layer of weights
        network.append(2*np.random.random((len(observation)+bias_nodes, layer_sizes[0]+bias_nodes))-1)
        #connect hidden layers
        for i in range(len(layer_sizes)-1):
            network.append(2*np.random.random((layer_sizes[i]+bias_nodes,layer_sizes[i+1]+bias_nodes))-1)
            #set bias node to value 1

        #connect last hidden layer to conclusion array
        network.append(2*np.random.random((layer_sizes[-1]+bias_nodes,len(conclusion)))-1)

        #add bias nodes
        for i in range(len(network)):
            print(network[i])
        
        print(network)
        
        return network
    
    def forwardPass(self, observations):
        
        network = self.network
        #make list of activations
        activations = []
        
        #first activation is the observation
        #hstack adds bias node to each observation
        activations.append(np.hstack ((observations, [[1]] * len (observations) )))
        
        #pass the activation through the network
        for i in range(len(network)):
            activations.append(nonlin(np.dot(activations[i], network[i])))

        self.activations = activations
        return(activations[-1])

        
    def backwardPass(self, conclusions):
        # Back propagation of errors using the chain rule.
        pass_error = []
        error = []
        deltas = []
        network = self.network
        activations = self.activations
        
        
        #step backwards through the network propagating deltas
        for i in range(len(network)):

            #start at -1 and go to negative len of network
            back_index = -1-i
                
                
            if (i==0):
                #final error comes from conclusions
                pass_error = conclusions - activations[-1]
                error = pass_error
            else:
                #sublayer error is dot product of previous changes and layer weights
                error = deltas[i-1].dot(network[back_index+1].T)
            
            
            #this layer weight change is previous activation error x derivative of this layer
            deltas.append(error*nonlin(activations[back_index], deriv=True))
            
            #update network layer weights
            network[back_index] += activations[back_index-1].T.dot(deltas[i])

        self.pass_error = pass_error
        self.network = network

    
def nonlin(x, deriv=False):  
    if(deriv==True):
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))   

main()

