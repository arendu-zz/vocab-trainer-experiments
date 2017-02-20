import numpy as np
from six.moves import cPickle
import copy
def load_obj(path): 
    f = open(path, 'rb')
    return cPickle.load(f)

def save_obj(obj, path):
    f = open(path, 'wb')
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def rargmax(vec):
    assert len(vec.shape) == 1
    return np.random.choice(np.nonzero(vec == np.amax(vec))[0], 1)[0]

def cosine_sim(v1, v2):
    assert len(v1) == len(v2)
    assert isinstance(v1, np.ndarray)
    assert isinstance(v2, np.ndarray)
    dot = 0.0
    v1_sq = 0.0
    v2_sq = 0.0
    for i in xrange(len(v1)):
        dot += v1[i] * v2[i]
        v1_sq += v1[i] ** 2.0
        v2_sq += v2[i] ** 2.0
    denom = (sqrt(v1_sq) * sqrt(v2_sq))
    if denom > 0.0:
        return dot / denom
    else:
        return float('inf')

def gradient_checking(theta, eps, likelihood_func, f_id, e_id):
    f_approx = np.zeros(np.shape(theta))
    for i, t in enumerate(theta):
        theta_plus = copy.deepcopy(theta)
        theta_minus = copy.deepcopy(theta)
        theta_plus[i] = theta[i] + eps
        theta_minus[i] = theta[i] - eps
        f_approx[i] = (likelihood_func(f_id, e_id, theta_plus) - likelihood_func(f_id, e_id, theta_minus)) / (2 * eps)
        #print i, len(theta)
    return f_approx

