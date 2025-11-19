def none_of(*xs): 
    '''
    return true if all args are None, return false if any are not None
    '''
    return all(x is None for x in xs)

def all_set(*xs): 
    '''
    return true if all args are not None (all set)
    '''
    return all(x is not None for x in xs)