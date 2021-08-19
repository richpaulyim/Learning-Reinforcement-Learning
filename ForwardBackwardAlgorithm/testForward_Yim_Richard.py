import sys
sys.path.append('../src/')

import unittest
from ddt import ddt, data, unpack
import forward_Yim_Richard as forward_Yim_Richard #change to file name


@ddt
class TestForward(unittest.TestCase):
    
    def assertNumericDictAlmostEqual(self, calculatedDictionary, expectedDictionary, places=7):
        self.assertEqual(calculatedDictionary.keys(), expectedDictionary.keys())
        for key in calculatedDictionary.keys():
            self.assertAlmostEqual(calculatedDictionary[key], expectedDictionary[key], places=places)

##################################################
#		Complete the code below
##################################################  
                
    # data parameter ordering (xT_1Distribution, eT, transitionTable, sensorTable, expectedResult)
    @data(({0:0.3, 1:0.7}, 1, {0:{0:0.6, 1:0.4}, 1:{0:0.3, 1:0.7}}, 
              {0:{0:0.6, 1:0.3, 2:0.1}, 1:{0:0, 1:0.5, 2:0.5}}, 
              {0: 0.27725, 1: 0.722748}), # expectedResult_1 (original `forward.py` example)
                # dataset one (above)
          ({0:0.7, 1:0.8}, 0, {0:{0:0.2, 1:0.8}, 1:{0:0.4, 1:0.6}}, 
              {0:{0:0.3, 1:0.4, 2:0.3}, 1:{0:6, 1:0.2, 2:0.2}}, 
              {0: 0.02163, 1: 0.97836}), # expectedResult_2 
                # dataset two (above)
          ({0:0.4, 1:0.5, 2:0.8}, 2,
             {0:{0:0.3, 1:0.2, 2:0.5}, 1:{0:0.0, 1:0.2, 2:0.8}, 2:{0:0.0, 1:0.3, 2:0.7}},
             {0:{0:0.6, 1:0.3, 2:0.1, 3:0.0}, 1:{0:1, 1:0.5, 2:0.3, 3:0.1}, 2:{0:1, 1:0.4, 2:0.3, 3:0.2}},
             {0: 0.02469135, 1: 0.25925, 2: 0.71604})) # expectedResult_3 
                # dataset three (above)

    @unpack # unpack data tuple
    def test_forward(self, xT_1Distribution, eT, transitionTable, sensorTable, expectedResult):
        
        # compute calculated results 
        calculatedResult = forward_Yim_Richard.forward(xT_1Distribution, eT, transitionTable, sensorTable)
        
        # unit test assertion
        self.assertNumericDictAlmostEqual(calculatedResult, expectedResult, places=4)

##################################################
#		Complete the code above
##################################################  
	
    def tearDown(self):
       pass
 
if __name__ == '__main__':
    unittest.main(verbosity=2)
