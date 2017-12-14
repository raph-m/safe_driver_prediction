import numpy as np

def log_loss(y, p):
    print("hello")
    p_2 = np.minimum(p, np.ones(len(p))-np.power(10.0, -8))
    p_2 = np.maximum(p_2, np.zeros(len(p))+np.power(10.0, -8))
    return -np.mean(y*np.log(p_2)+(1-y)*np.log(1-p_2))
