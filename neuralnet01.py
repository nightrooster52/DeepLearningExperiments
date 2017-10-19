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
    
    
    network = generateDeepNet(X[0], [4], y[0])


    for i in range(60000):
        network, activations, error = step(X, network, y)
        
        if(i % 10000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output. 
            print("Epoch "+ str(i))
            print("Error : " + str(np.mean(np.abs(error))))
#            print("outputs : " +  str(outputs) )
    print("output after training")
    print(activations[-1])

def nonlin(x, deriv=False):  
    if(deriv==True):
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))

def generateDeepNet(observation, layer_sizes, conclusion):
    """
    Takes an observation ( array of attributes ) , layer_sizes ( array ) , and conclusion ( array of attributes ) 
    generates the appropriate network ( list of matrixes )
    """
    
    network = []
    
    #connect input array to first layer of weights
    network.append(2*np.random.random((len(observation), layer_sizes[0]))-1)
    
    #connect hidden layers
    for i in range(len(layer_sizes)-1):
        network.append(2*np.random.random((layer_sizes[i],layer_sizes[i+1]))-1)
        
    #connect last hidden layer to conclusion array
    network.append(2*np.random.random((layer_sizes[-1],len(conclusion)))-1)
    
    return network

def step(observations, network, conclusions):
    
    #make list of activations
    activations = []
    
    #first activation is the observation
    activations.append(observations)
    
    #pass the activation through the network
    for i in range(len(network)):
        activations.append(nonlin(np.dot(activations[i], network[i])))
    # Back propagation of errors using the chain rule.
    pass_error = []
    error = []
    deltas = []
    
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
    

    
    return network, activations, pass_error
    
    
main()
