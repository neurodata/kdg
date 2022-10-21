import numpy as np

def unit_step(x):
    '''
    returns the unit step function output to the given inputs
    '''
    x = np.maximum(x, 0)
    x[x > 0] = 1
    return x

def get_polytope_memberships(X, nn):
    '''
    returns the polytope memberships of given data samples w.r.t to a trained neural network
    '''
    polytope_memberships = []
    last_activations = X

    # Iterate through neural network manually, getting node activations at each step
    for layer_id in range(len(nn.layers)):
        weights, bias = nn.layers[layer_id].get_weights()

        # Calculate new activations based on input to this layer
        preactivation = np.matmul(last_activations, weights) + bias

        # get list of activated nodes in this layer
        if layer_id == len(nn.layers) - 1:
            binary_preactivation = (preactivation > 0.5).astype('int')
        else:
            binary_preactivation = (preactivation > 0).astype('int')
        
        if layer_id < len(nn.layers) - 1:
            polytope_memberships.append(binary_preactivation)
        
        # remove all nodes that were not activated
        last_activations = preactivation * binary_preactivation

    # Concatenate all activations for given observation
    polytope_obs = np.concatenate(polytope_memberships, axis = 1)
    polytope_memberships = [np.tensordot(polytope_obs, 2 ** np.arange(0, np.shape(polytope_obs)[1]), axes = 1)]
    
    return polytope_memberships

def get_activation_pattern(polytope_id, network_shape):
    '''
    returns the activation pattern encoded by the given polytope membership
    '''
    binary_string = np.binary_repr(polytope_id, width=sum(network_shape)-network_shape[-1])[::-1] 
    return np.array(list(binary_string)).astype('int')
