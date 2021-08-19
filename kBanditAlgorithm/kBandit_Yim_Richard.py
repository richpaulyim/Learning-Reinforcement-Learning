# sources for homework: numpy and matplotlib documentation
import numpy as np
import matplotlib.pyplot as plt

def getSamplar():
    mu=np.random.normal(0,10)
    sd=abs(np.random.normal(5,2))
    getSample=lambda: np.random.normal(mu,sd)
    return getSample

def e_greedy(Q, e):

##################################################
#		Your code here
##################################################  

    # generate probability
    policy = np.random.choice([1,0], size=1, p=[1-e,e])

    if policy:
        # find index of maximum reward
        action = np.argmax([Q[a] for a in Q])
    else:
        # random action
        action = np.random.choice(list(Q.keys()))

    return action 

def upperConfidenceBound(Q, N, c):
   
##################################################
#		Your code here
##################################################  
    
    # specify the action space
    actionspace = list(Q.keys())
    # time step
    t = np.sum([N[a] for a in N])
    # check if any of the actions have yet to be chosen
    unexplored = np.where(np.asarray([N[a] for a in N])==0)[0]
    if len(unexplored)!=0:
        # random choice for infinite upper confidence bound
        action = np.random.choice([actionspace[a] for a in unexplored])
    else: 
        # upper confidence bound rule
        action = np.argmax([Q[a] + c*np.sqrt(np.log(t)/N[a]) for a in actionspace])
 
    return action

def updateQN(action, reward, Q, N):

##################################################
#		Your code here
##################################################  

    # initialization
    QNew = Q.copy(); NNew = N.copy()
    # update
    NNew[action] = NNew[action]+1
    QNew[action] = Q[action]+(reward-Q[action])/NNew[action]
    del Q; del N
 
    return QNew, NNew

def decideMultipleSteps(Q, N, policy, bandit, maxSteps):

##################################################
#		Your code here
##################################################  

    # initialize action reward
    actionReward = []
    # begin looping
    for i in range(maxSteps):
        action = policy(Q,N)
        r = bandit(action)
        Q, N = updateQN(action, r, Q, N)
        actionReward.append((action, r))
 
    return {'Q':Q, 'N':N, 'actionReward':actionReward}

def plotMeanReward(actionReward,label):
    maxSteps=len(actionReward)
    reward=[reward for (action,reward) in actionReward]
    meanReward=[sum(reward[:(i+1)])/(i+1) for i in range(maxSteps)]
    plt.plot(range(maxSteps), meanReward, linewidth=0.9, label=label)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')

def main():
    np.random.seed(2020)
    K=10
    maxSteps=1000
    Q={k:0 for k in range(K)}
    N={k:0 for k in range(K)}
    testBed={k:getSamplar() for k in range(K)}
    bandit=lambda action: testBed[action]()
    
    policies={}
    policies["e-greedy-0.5"]=lambda Q, N: e_greedy(Q, 0.5)
    policies["e-greedy-0.1"]=lambda Q, N: e_greedy(Q, 0.1)
    policies["UCB-2"]=lambda Q, N: upperConfidenceBound(Q, N, 2)
    policies["UCB-20"]=lambda Q, N: upperConfidenceBound(Q, N, 20)
    
    allResults = {name: decideMultipleSteps(Q, N, policy, bandit, maxSteps) for (name, policy) in policies.items()}
    
    for name, result in allResults.items():
         plotMeanReward(allResults[name]['actionReward'], label=name)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    
if __name__=='__main__':
    main()
