class ObjectSpace(Space):
    '''
    A gym space defined by a set of objects. 
    Useful when an action is a choice of strings.
    '''
    
    def __init__(self, objects):
        super(ObjectSpace, self).__init__()
        if objects:
            self.objects = tuple(objects)
        else:
            self.objects = tuple()
        
    def sample(self):
        idx = self.np_random.choice(len(self.objects))
        return self.objects[idx]
    
    def contains(self, x):
        return x in self.objects
    
    def __len__(self):
        return len(self.objects)
    
    