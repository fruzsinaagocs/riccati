# Unit tests for ricatti.py
import numpy as np
import ricatti

def test_cheb():
    a = 3.0
    f = lambda x: np.sin(a*x + 1.0)
    df = lambda x: a*np.cos(a*x + 1.0)
    for i in [16, 32]:
        D, x = ricatti.cheb(i)
        maxerr = max(np.abs(np.matmul(D, f(x)) - df(x))) 
        print("At n = {} points, maximum error from Chebyshev differentiation is {}".format(i, maxerr))
        assert  maxerr < 1e-8

