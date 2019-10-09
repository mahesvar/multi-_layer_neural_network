import numpy as np

class backpropagationnet:
    # class members
    layercount = 0
    layershape = None
    weights = []
    
    # class method
    def __init__(self,layersize):
        self.layercount = len(layersize)
        self.layershape = layersize
        
        self._inputlayer = []
        self._outputlayer = []
        
        # create weight array
        for (l1,l2) in zip(layersize[:-1],layersize[1:]):
            self.weights.append(np.random.normal(scale = 0.1 , size = (l2,l1+1)))
        
    def run(self,Input):
        self.lncases = Input.shape[0]
        
        # clear the previouus intermediate valuelistss
        self._inputlayer = []
        self._outputlayer = []
        # run it
        for index in range(self.layercount-1):
            # determine layer input
            if index == 0:
                layerinput = self.weights[0].dot(np.vstack([Input.T, np.ones([1,self.lncases])]))
            else:
                layerinput = self.weights[index].dot(np.vstack([self._outputlayer[-1], np.ones([1,self.lncases])]))
            self._inputlayer.append(layerinput)
            self._outputlayer.append(self.sigmoid(layerinput))
        return self._outputlayer[-1].T
    
    # trainepoch method
    def trainepoch(self,Input,target,trainingRate = 0.2):
        # this method trains the network for one epoch
        delta =[]
        self.lncases = Input.shape[0]
        
        # first run the network
        self.run(Input)
        
        # calculate delta
        # backward propagation 
        for index in reversed(range(self.layercount)):
           if index == self.layercount-1:
               # compare to the target values
               output_delta = self._outputlayer[index] - target.T
               error = np.sum(output_delta**2)
               delta.append(output_delta * self.sigmoid(self._inputlayer[index], True))
               
           else:
               #compare to the following layer's delta
               delta_pullback = self.weights[index+1].T.dot(delta[-1])
               delta.append(delta_pullback[-1,:] * self.sigmoid(self._inputlayer[index], True))
               
    
    # transfer function
    def sigmoid(self, x, Derivative =  False):
        if not Derivative:
            return 1/ (1+np.exp(-x))
        else:
            out = self.sigmoid(x)
            return out*(1-out)            
            
if __name__ == "__main__":
    bpn = backpropagationnet((2,2,2))
    print(bpn.layershape)
    print(bpn.weights)
    
    lvinput = np.array([[0,0], [1,1], [-1,0.5]])
    lvoutput = bpn.run(lvinput)
    
    print("input : {0}\n output:{1}".format(lvinput, lvoutput))
