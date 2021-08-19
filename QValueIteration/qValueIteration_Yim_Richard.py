import numpy as np
import drawHeatMap as hm
import rewardTable as rt
import transitionTable as tt
Q1={(0, 0): {(0, 1): 10, (0, -1): 20, (1, 0): 30, (-1, 0): 40}, (0, 1): {(0, 1): 50, (0, -1): 60, (1, 0): 80, (-1, 0): 0}, 
        (0, 2): {(0, 1): -9, (0, -1): 8, (1, 0): 9, (-1, 0): 2}, (1, 0): {(0, 1): 0, (0, -1): 100, (1, 0): 80, 
            (-1, 0): 0}, (1, 2): {(0, 1): 0, (0, -1): 0, (1, 0): 0, (-1, 0): 0}, (2, 0): {(0, 1): 0, (0, -1): 0, 
                (1, 0): 0, (-1, 0): 0}, (2, 1): {(0, 1): 0, (0, -1): 0, (1, 0): 0, (-1, 0): 0}, (2, 2): {(0, 1): 0, 
                    (0, -1): 0, (1, 0): 0, (-1, 0): 99}, (3, 0): {(0, 1): 0, (0, -1): 0, (1, 0): 0, (-1, 0): 0}, 
                (3, 1): {(0, 1): 0, (0, -1): 21, (1, 0): 0, (-1, 0): 0}, (3, 2): {(0, 1): 0, (0, -1): 0, 
                    (1, 0): 10, (-1, 0): 30}}
Q2={(0,0): {(0,1): 30, (0,-1):-100}, (0,1):{(0,-1):-29, (0,1):29}, (0,-1):{(0,1):30, (0,-1):20}}

def expect(xDistribution, function):
    expectation=sum([function(x)*px for x, px in xDistribution.items()])
    return expectation

def getSPrimeRDistributionFull(s, action, transitionTable, rewardTable):
    reward=lambda sPrime: rewardTable[s][action][sPrime]
    p=lambda sPrime: transitionTable[s][action][sPrime]
    sPrimeRDistribution={(sPrime, reward(sPrime)): p(sPrime) for sPrime in transitionTable[s][action].keys()}
    return sPrimeRDistribution
    
def updateQFull(s, a, Q, getSPrimeRDistribution, gamma):

##################################################
#		Your code here
##################################################  

    # s, to be used as state key (scalar)
    # a, to be used as action key (scalar)
    # Q, Q-table, tuples of (s,a) {s: {a: Q(s,a)}}
    # getSPrimeRDistribution (function[s,a]) return:
    # - {(s',r): p(s',r)}
    # gamma, scale factor (scalar)
    # RETURN SCALAR 

    # dictionary distribution 
    psprsa = getSPrimeRDistribution(s,a)
    spr = list(psprsa.keys())
    # compute Qas
    Qas = sum([psprsa[sp] * (sp[1] + gamma * max(Q[sp[0]][ap] for ap in list(Q[sp[0]].keys()))) for sp in spr])

    # return Qas
    return Qas

def qValueIteration(Q, updateQ, stateSpace, actionSpace, convergenceTolerance):

##################################################
#		Your code here
##################################################  
    
    # Q, table to be updated, {s: {a: Q(s,a)}}
    # updateQ, function(s, a, Q) = Qnew(s,a), scalar
    # stateSpace, list of all possible states
    # actionSpace, a list of all possible actions
    # convergenceTolerance, theta, scalar
    # RETURN DICTIONARY


    while True:

        delta = 0
        for s in stateSpace:
            # update Q-table for given state across each action
            for a in actionSpace:
                # store old q
                q = Q[s][a]
                # update Q with respect to s and a
                Q[s][a] = updateQ(s, a, Q)
                # compute error-delta
                delta = max([delta, abs(q-Q[s][a])])

        # verifying stopping condition criterion
        if delta < convergenceTolerance:
            break

    # assign Qnew and return the table
    QNew = Q
    
    # return QNew
    return QNew

def getPolicyFull(Q, roundingTolerance):

##################################################
#		Your code here
##################################################  

    # Q, dictionary of Q(a|s), actions and Q conditioned on state, {a: Q(s,a)}
    # rounding tolerance, a scalar
    # RETURN DICTIONARY

    # find optimal action
    optimalIndex = np.argmax([Q[a] for a in list(Q.keys())])
    optimalAction = [list(Q.keys())[optimalIndex]]

    # find other policies within tolerance rang (tied-action boolean indices)
    optimalIndices = np.abs(np.asarray([Q[a] for a in list(Q.keys())])-Q[optimalAction[0]]) < roundingTolerance
    optimalActions = []
    # create list of optimal actions 
    for j, oi in enumerate(optimalIndices):
        # check if within rounding tolerance
        if oi:
            optimalActions.append(list(Q.keys())[j])
        
    # construct policy dictionary, uniform policy distribution 
    policy = {oA: 1/len(optimalActions) for oA in optimalActions}

    # return policy
    return policy


##################################################  
def viewDictionaryStructure(d, levels, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(levels[indent]) + ": "+ str(key))
        if isinstance(value, dict):
            viewDictionaryStructure(value, levels, indent+1)
        else:
            print('\t' * (indent+1) + str(levels[indent+1])+ ": " + str(value))


def main():
    
    minX, maxX, minY, maxY=(0, 3, 0, 2)
    
    actionSpace=[(0,1), (0,-1), (1,0), (-1,0)]
    stateSpace=[(i,j) for i in range(maxX+1) for j in range(maxY+1) if (i, j) != (1, 1)]
    Q={s:{a: 0 for a in actionSpace} for s in stateSpace}
    
    normalCost=-0.04
    trapDict={(3,1):-1}
    bonusDict={(3,0):1}
    blockList=[(1,1)]
    
    p=0.8
    transitionProbability={'forward':p, 'left':(1-p)/2, 'right':(1-p)/2, 'back':0}
    transitionProbability={move: p for move, p in transitionProbability.items() if transitionProbability[move]!=0}
    
    transitionTable=tt.createTransitionTable(minX, minY, maxX, maxY, trapDict, bonusDict, blockList, actionSpace, transitionProbability)
    rewardTable=rt.createRewardTable(transitionTable, normalCost, trapDict, bonusDict)

    
    """
    levelsReward  = ["state", "action", "next state", "reward"]
    levelsTransition  = ["state", "action", "next state", "probability"]
    
    viewDictionaryStructure(transitionTable, levelsTransition)
    viewDictionaryStructure(rewardTable, levelsReward)
    """
        
    getSPrimeRDistribution=lambda s, action: getSPrimeRDistributionFull(s, action, transitionTable, rewardTable)
    gamma = 0.8       
    updateQ=lambda s, a, Q: updateQFull(s, a, Q, getSPrimeRDistribution, gamma)
    
    convergenceTolerance = 10e-7
    QNew=qValueIteration(Q, updateQ, stateSpace, actionSpace, convergenceTolerance)
    
    roundingTolerance= 10e-7
    getPolicy=lambda Q: getPolicyFull(Q, roundingTolerance)
    policy={s:getPolicy(QNew[s]) for s in stateSpace}
    
    V={s: max(QNew[s].values()) for s in stateSpace}
    
    VDrawing=V.copy()
    VDrawing[(1, 1)]=0
    VDrawing={k: v for k, v in sorted(VDrawing.items(), key=lambda item: item[0])}
    policyDrawing=policy.copy()
    policyDrawing[(1, 1)]={(1, 0): 1.0}
    policyDrawing={k: v for k, v in sorted(policyDrawing.items(), key=lambda item: item[0])}

    hm.drawFinalMap(VDrawing, policyDrawing, trapDict, bonusDict, blockList, normalCost)

    
    
    
if __name__=='__main__': 
    main()
