import sys
sys.path.append('../src/')

import unittest
from ddt import ddt, data, unpack
import qValueIteration_Yim_Richard as qVIYR # my file name


@ddt
class TestBellman(unittest.TestCase):
    
    # ==========================================================================================
    # output assertion functions
    def assertNumericScalarAlmostEqual(self, calculatedScalar, expectedScalar, places=7):
        self.assertAlmostEqual(calculatedScalar, expectedScalar, places=places)

    def assertNumericDictionaryItemsAlmostEqual(self, calculatedDictionary, expectedScalar, places=7):
        
        [self.assertAlmostEqual(calculatedDictionary[k], expectedScalar, places=places) for k in calculatedDictionary]

    def assertNumericTableItemsAlmostEqual(self, calculatedTable, expectedTable, places=7):
        for state in calculatedTable:
            for action in calculatedTable[state]:
                self.assertAlmostEqual(calculatedTable[state][action], expectedTable[state][action], places=places)


    # ==========================================================================================
    # updateQFull
    # datasets and expected
    @data((1, 2, {0: {1: 10, -1: 8}, 1: {1: 4, -1: -8}, 2: {1: 1, -1: 1}}, 
            lambda x,y: {(1,10): 0.8, (1,2): 0.2},
            0.6, 10.8), # dataset 1
            (0, 0, {0: {-1: 10, 0:20, 1: 8}, 1: {-1: 4, 0:30, 1: -8}, 2: {-1: 1, 0:3, 1: 1}}, 
            lambda x,y: {(0,10): 0.3, (0,55): 0.7},
            0.9, 59.49999) # dataset 2
          )
    # test function
    @unpack # unpack data tuple
    def test_updateQFull(self, s, a, Q, getSPrimeRDistribution, gamma, expectedResult):
        
        # compute calculated results 
        calculatedResult = qVIYR.updateQFull(s,a,Q,getSPrimeRDistribution, gamma)
        
        # unit test assertion
        self.assertNumericScalarAlmostEqual(calculatedResult, expectedResult, places=4)

    # ==========================================================================================
    # qValueIteration
    # datasets and expected
    @data(({(0, 0): {(0, 1): 0, (0, -1): 0, (1, 0): 0, (-1, 0): 0}, (0, 1): {(0, 1): 0, (0, -1): 0, (1, 0): 0, (-1, 0): 0},
           (0, 2): {(0, 1): 0, (0, -1): 0, (1, 0): 0, (-1, 0): 0}, (1, 0): {(0, 1): 0, (0, -1): 0, (1, 0): 0, (-1, 0):
              0}, (1, 2): {(0, 1): 0, (0, -1): 0, (1, 0): 0, (-1, 0): 0}, (2, 0): {(0, 1): 0, (0, -1): 0, (1, 0): 0,
                  (-1, 0): 0}, (2, 1): {(0, 1): 0, (0, -1): 0, (1, 0): 0, (-1, 0): 0}, (2, 2): {(0, 1): 0, (0, -1): 0,
                      (1, 0): 0, (-1, 0): 0}, (3, 0): {(0, 1): 0, (0, -1): 0, (1, 0): 0, (-1, 0): 0}, (3, 1): {(0, 1):
                          0, (0, -1): 0, (1, 0): 0, (-1, 0): 0}, (3, 2): {(0, 1): 0, (0, -1): 0, (1, 0): 0, (-1, 0): 0}},
                      lambda x,y,z: qVIYR.Q1[x][y], # this Q table at the top of (qValueIteration_Yim_Richard.py)
           [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)],
           [(0, 1), (0, -1), (1, 0), (-1, 0)], 0.0000001, 
           {(0, 0): {(0, 1): 10, (0, -1): 20, (1, 0): 30, (-1, 0): 40}, (0, 1): {(0, 1): 50, (0, -1): 60, (1, 0): 80, (-1, 0): 0},
                (0, 2): {(0, 1): -9, (0, -1): 8, (1, 0): 9, (-1, 0): 2}, (1, 0): {(0, 1): 0, (0, -1): 100, (1, 0): 80, (-1, 0): 0},
                (1, 2): {(0, 1): 0, (0, -1): 0, (1, 0): 0, (-1, 0): 0}, (2, 0): {(0, 1): 0, (0, -1): 0, (1, 0): 0, (-1, 0): 0}, (2,
                    1): {(0, 1): 0, (0, -1): 0, (1, 0): 0, (-1, 0): 0}, (2, 2): {(0, 1): 0, (0, -1): 0, (1, 0): 0, (-1, 0): 99}, (3,
                        0): {(0, 1): 0, (0, -1): 0, (1, 0): 0, (-1, 0): 0}, (3, 1): {(0, 1): 0, (0, -1): 21, (1, 0): 0, (-1, 0): 0},
                    (3, 2): {(0, 1): 0, (0, -1): 0, (1, 0): 10, (-1, 0): 30}}), # dataset 1
            ({(0,0): {(0,1): 100, (0,-1):-10}, (0,1):{(0,-1):20,(0,1):0}, (0,-1):{(0,1):30, (0,-1):0}},
            lambda x,y,z: qVIYR.Q2[x][y], # this Q table at the top of (qValueIteration_Yim_Richard.py)
            [(0,-1),(0,0),(0,1)],
            [(0,-1),(0,1)], 0.0000001,
            {(0, 0): {(0, 1): 30, (0, -1): -100}, (0, 1): {(0, -1): -29, (0, 1): 29}, (0, -1): {(0, 1): 30, (0, -1): 20}})
            )
    # test function
    @unpack # unpack data tuple
    def test_qValueIteration(self, Q, updateQ, stateSpace, actionSpace, convergenceTolerance, expectedResult):
        
        # compute calculated results 
        calculatedResult = qVIYR.qValueIteration(Q, updateQ, stateSpace, actionSpace, convergenceTolerance)
        
        # unit test assertion
        self.assertNumericTableItemsAlmostEqual(calculatedResult, expectedResult, places=4)

    # ==========================================================================================
    # getPolicyFull
    # datasets and expected
    @data(({-1:-10,1:10}, 0.1, 1),# dataset 1
          ({-1:-10, 0:-10, 1:-10}, 0.00003, 0.3333333) # dataset 2
          ) 

    # test function
    @unpack # unpack data tuple
    def test_getPolicyFull(self, Q, roundingTolerance, expectedResult):
        
        # compute calculated results 
        calculatedResult = qVIYR.getPolicyFull(Q, roundingTolerance)
        
        # unit test assertion
        self.assertNumericDictionaryItemsAlmostEqual(calculatedResult, expectedResult, places=4)

    # ==========================================================================================
    # tearDown
    def tearDown(self):
       pass

if __name__ == '__main__':
    unittest.main(verbosity=2)
