'''
Test scilog
'''
import unittest
from scilog import record,load
import numpy as np
from scilog.scilog import ConvergencePlotter
class TestScilog(unittest.TestCase):
    def test(self):
        def np_rand(n):
            X=np.random.rand(n,n)
            return X
        experiments=[int(10**(i/2)) for i in range(9)]
        path=record(func=np_rand,inputs=experiments,memory_profile=True)
        results=load(path)
        print(results)
class Meaner():
    def __init__(self):
        self.results = np.zeros(0)
    def f(self,j):
        self.results = np.concatenate((self.results,np.random.rand(10000)))
        print(self.results)
        print(np.mean(self.results))
        return np.mean(self.results),np.std(self.results)

if __name__ == "__main__":
    unittest.main()
    record(func = Meaner().f,inputs = range(5),analyze = ConvergencePlotter(qois = 2,cumulative = True))
