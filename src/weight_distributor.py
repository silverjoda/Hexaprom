import numpy as np

class Wdist:
    def __init__(self):
        self.w_dict = {}
        self.ctr = 0

    def _prod(self, a):
        acc = 1
        for x in a:
            acc *= x
        return acc

    def addW(self, shape, name):
        n_ctr = self.ctr + self._prod(shape)
        self.w_dict[name] = [self.ctr, n_ctr, shape]
        self.ctr = n_ctr

    def get_w(self, name, w):
        f,t,s = self.w_dict[name]
        return w[f:t].reshape(s)

    def fill_w(self, name, w, val):
        f,t,s = self.w_dict[name]
        w[f:t] = val

    def get_N(self):
        return self.ctr