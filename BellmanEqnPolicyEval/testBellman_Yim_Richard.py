import sys
sys.path.append('../src/')

import unittest
from ddt import ddt, data, unpack
import policyEvaluation_Yim_Richard as pEYR # my file name


@ddt
class TestBellman(unittest.TestCase):
    
    def assertNumericScalarAlmostEqual(self, calculatedScalar, expectedScalar, places=7):
        self.assertAlmostEqual(calculatedScalar, expectedScalar, places=places)

##################################################
#		Complete the code below
##################################################  
                
    # data parameter ordering (s, policy, V, transitionTable, getSPrimeRDistribution, gamma, expectedResult)
    @data(((2), lambda Q: pEYR.e_greedyProbability(Q, 0.4),                     # state and policy
            {(0): 10.22, (1):-38.2, (2):-49.21, (3):199.21, (4):42, (5):-9.6},  # value table
            {(0):                                                               # transition table 
                {(1): {(1): 1}, (-1): {(0): 1}},
             (1):                              
                {(1): {(2): 1}, (-1): {(0): 1}},
             (2):                              
                {(1): {(3): 1}, (-1): {(1): 1}},
             (3):                              
                {(1): {(4): 1}, (-1): {(2): 1}},
             (4):                              
                {(1): {(5): 1}, (-1): {(3): 1}},
             (5):                              
                {(1): {(5): 1}, (-1): {(4): 1}}}, 
             lambda x,y: {((3), 100): 1},                                       # SPrimeR function
                0.6,                                                            # gamma
                219.526),                                                       # expectedResult_1
                # 1 dimensional walk over 6 tiles
                # dataset one (above) ============================================================
              ((0,0), lambda Q: pEYR.e_greedyProbability(Q, 0.6),               # state and policy
            {(0,0):-22.10, (0,1):2.38, (1,0):21.49, (1,1):12.91},               # Value table
            {(0,0):                                                             # transition table 
                {(1,0): {(1,0): 1}, (0,1): {(0,1): 1}, (-1,0): {(0,0): 1}, (0,-1): {(0,0): 1}},
             (0,1):                              
                {(1,0): {(1,1): 1}, (0,1): {(0,1): 1}, (-1,0): {(0,1): 1}, (0,-1): {(0,0): 1}},
             (1,0):                              
                {(1,0): {(1,0): 1}, (0,1): {(1,1): 1}, (-1,0): {(0,0): 1}, (0,-1): {(1,0): 1}},
             (1,1):                              
                {(1,0): {(1,1): 1}, (0,1): {(1,1): 1}, (-1,0): {(0,1): 1}, (0,-1): {(1,0): 1}}},
             lambda x,y: {((1, 0), -1000): 1},                                  # SPrimeR function
                0.55,                                                           # gamma
                -988.1805))                                                     # expectedResult_2 
                # 2 dimensional walk over 2x2 cells
                # dataset two (above) ============================================================
    @unpack # unpack data tuple
    def test_Bellman(self, s, policy, V, transitionTable, getSPrimeRDistribution, gamma, expectedResult):
        
        # compute calculated results 
        calculatedResult = pEYR.Bellman(s, policy, V, transitionTable, 
                getSPrimeRDistribution, gamma)
        
        # unit test assertion
        self.assertNumericScalarAlmostEqual(calculatedResult, expectedResult, places=4)

##################################################
#		Complete the code above
##################################################  
	
    def tearDown(self):
       pass
 
if __name__ == '__main__':
    unittest.main(verbosity=2)
