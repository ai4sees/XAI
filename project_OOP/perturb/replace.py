import numpy as np


def _reshape(x, mask):
    assert x.shape == mask.shape, (
        f"{x.shape} does not match with mask shape, {mask.shape}"
    )
    if x.shape == 1:
        x = x.reshape(-1, 1)
        mask = mask.reshape(-1, 1)
    return x, mask


def zeros(x):
    return np.zeros_like(x)

def local_mean(x, mask):
    x, mask = _reshape(x, mask)
    _, features = x.shape
    r = np.zeros_like(x)
    for i in range(features):
        for s in np.unique(mask):
            idx = (mask[:, i] == s)
            r[idx, i] = np.average(r[idx, i])
            
    return r


def global_mean(x):
    r = local_mean(x, np.ones_like(x))
    return r


def inverse_max(x):

    max_ = x.max(axis = 0)
    r = max_ - x
    return r

def inverse_mean(x):
    mean_ = global_mean(x)
    r = mean_ - x
    return r

def random(x):
    r = np.random.rand(x.shape)
    return r