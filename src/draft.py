import numpy as np
import cma

def f2(w):
    # fl
    out = np.tanh(np.matmul(np.array([1,1]), w[:-3].reshape((2,3))) + w[-3:])
    loss = np.mean(np.square(out - np.array([0,.2,-0.4])))
    return loss

# Generate weights
w = np.random.randn(2 * 3 + 3)

N_weights = len(w)
print("Nweights: {}".format(N_weights))
W_MULT = 1
ACT_MULT = 1

w = np.random.randn(N_weights)
print("Loss before opt : {}".format(f2(w)))
es = cma.CMAEvolutionStrategy(w, 0.5)

try:
    es.optimize(f2, iterations=1000)

except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()


print("Loss after opt : {}".format(f2(es.result.xbest)))