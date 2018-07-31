import numpy as np
import cma

def fun(x):
    return  x[0] ** 2 + 3 * x[1] ** 2

es = cma.CMAEvolutionStrategy([4, 3], 0.5)
es.optimize(fun)
es.result_pretty()
