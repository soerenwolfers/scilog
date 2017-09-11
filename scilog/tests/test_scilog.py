'''
Test scilog
'''
import unittest
from scilog import conduct,load
import numpy as np
class TestScilog(unittest.TestCase):
    def test(self):
        def np_rand(n):
            X=np.random.rand(n,n)
            return X
        experiments=[int(10**(i/2)) for i in range(9)]
        path=conduct(func=np_rand,experiments=experiments,memory_profile=True)
        info,results,directory=load(path)
        print(info)
        #print(results)
        print(directory)

if __name__ == "__main__":
    unittest.main()
