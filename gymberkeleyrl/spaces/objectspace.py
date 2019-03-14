from gym.spaces import Space
import numpy as np


class ObjectSpace(Space):
    '''
    A space defined by a set of objects. 
    Useful when an action or state is a choice of strings.
    '''
    
    def __init__(self, objects):
        super(ObjectSpace, self).__init__()
        if objects:
            self.objects = tuple(objects)
        else:
            self.objects = tuple()
            
        self.np_random = np.random.RandomState()

    def seed(self, seed):
        self.np_random.seed(seed)
        
    def sample(self):
        idx = self.np_random.choice(len(self.objects))
        return self.objects[idx]
    
    def contains(self, x):
        return x in self.objects
    
    def __len__(self):
        return len(self.objects)
    
    def __repr__(self):
        return "ObjectSpace(%d)" % self.objects

    def __eq__(self, other):
        return self.objects == other.objects
    
    