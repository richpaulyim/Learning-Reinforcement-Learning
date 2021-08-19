# =============================================================================================
# MODULE 1
# =============================================================================================

def expect(xDistribution, function):
    fxProduct=[px*function(x) for x, px in xDistribution.items()]
    expectation=sum(fxProduct)
    return expectation

def getUnnormalizedPosterior(prior, likelihood):
    
##################################################
#		Your code here
################################################## 
# prior is a dictionary
# likelihood is a dictionary
# return is the unnormalized posterior
    
    # one-liner, just cycle through keys
    # keys should be the same between dictionaries
    return {s: likelihood[s]*prior[s] for s in likelihood}

def normalize(unnormalizedDistribution):

##################################################
#		Your code here
################################################## 
# unnormalizedDistribution is a dictionary
# return is the normalized empirical frequency distribution

    # normalization (relative frequency) constant 
    constant = getSumOfProbability(unnormalizedDistribution)

    # return normalized distribution
    return {x: unnormalizedDistribution[x]/constant for x in unnormalizedDistribution}

def getSumOfProbability(unnormalizedDistribution):
    
##################################################
#		Your code here
################################################## 
# unnormalizedDistribution is a dictionary
# return is the sum of the absolute frequency
    
    # one-liner, just cycle through keys
    # just get unnormalized distribution and take sum
    return sum([unnormalizedDistribution[x] for x in unnormalizedDistribution])

def getPosterior(prior, likelihood):

##################################################
#		Your code here
################################################## 
# prior is a dictionary
# likelihood is a dictionary
# return is the normalized posterior distribution

    # one-liner just call normalized and get unnormalized posterior
    return normalize(getUnnormalizedPosterior(prior=prior,likelihood=likelihood))

def getMarginalOfData(prior, likelihood):
    
##################################################
#		Your code here
################################################## 
# prior is a dictionary
# likelihood is a dictionary
# return is the marginal probability of prior distribution (evidence)

    # one-liner call get sum of probability
    return getSumOfProbability(getUnnormalizedPosterior(prior=prior,likelihood=likelihood))

# =============================================================================================
# MODULE 2
# =============================================================================================

def getEU(action, sDistribution, rewardTable):
    
##################################################
#		Your code here
################################################## 
# action is a string indicating which action is being evaluated
# sDistribution is the probability distribution of current state
# rewardTable is a dictionary of taking action a in state s

    # one-liner call get expected utility of the action using expect function provided
    return expect(sDistribution, lambda x: rewardTable[x][action])

def getMaxEUFull(evidence, prior, likelihoodTable, rewardTable, actionSpace):
    
##################################################
#		Your code here
################################################## 
# evidence, string indicating the evidence collection {it can be None}
# prior, dictionary of {s: p(s)}
# likelihoodTable, dictionary of {e: {s: p(e|s)}}
# rewardTable, dictionary of {s: {a: R(a,s)}}
# actionSpace, list of actions ['a1','a2']
# return is a scalar value representing max EU after receiving ej

    # check that evidence is None or string
    if evidence==None:
        # since None, we are using the prior distribution
        sDist = prior
    else:
        # since new evidence, we are using the posterior P(s|e)
        sDist = getPosterior(prior=prior, likelihood=likelihoodTable[evidence])

    # one-liner, compute EU
    return max([getEU(action=a, sDistribution=sDist, rewardTable=rewardTable) for a in actionSpace])

# =============================================================================================
# MODULE 3
# =============================================================================================

def getValueOfInformationOfATest(prior, evidenceSpace, getMarginalOfEvidence, getMaxEU):
    
##################################################
#		Your code here
################################################## 
# prior, dictionary {s: p(s)} 
# evidenceSpace, list of possible from test [use elements from here as inputs for functions below]
# getMarginalOfEvidence [input is an element of evidence]
# getMaxEU, function for either EU(a|e) or EU(a) [input is an element of evidence]

    ea = getMaxEU(None)
    sumpea = sum([getMarginalOfEvidence(e)*getMaxEU(e) for e in evidenceSpace])

    return sumpea - ea



def main():
    
    prior={'Well 1 contains oil': 0.2, 'Well 2 contains oil': 0.4, 'Well 3 contains oil': 0.2, 'Well 4 contains oil': 0.2}
    
    actionSpace=['Buy Well 1', 'Buy Well 2', 'Buy Well 3', 'Buy Well 4']
    rewardTable={'Well 1 contains oil': {'Buy Well 1': 100, 'Buy Well 2': 0, 'Buy Well 3': 0, 'Buy Well 4': 0},
                 'Well 2 contains oil': {'Buy Well 1': 0, 'Buy Well 2': 100, 'Buy Well 3': 0, 'Buy Well 4': 0},
                 'Well 3 contains oil': {'Buy Well 1': 0, 'Buy Well 2': 0, 'Buy Well 3': 100, 'Buy Well 4': 0},
                 'Well 4 contains oil': {'Buy Well 1': 0, 'Buy Well 2': 0, 'Buy Well 3': 0, 'Buy Well 4': 100}}   
    
    testSpace=['Test Well 1', 'Test Well 2', 'Test Well 3', 'Test Well 4']
    evidenceSpace=['Microbe', 'No microbe']
    likelihoodTable={'Test Well 1':{'Microbe': {'Well 1 contains oil': 0.8, 'Well 2 contains oil': 0.1, 'Well 3 contains oil': 0.1, 'Well 4 contains oil': 0.1},
                                    'No microbe': {'Well 1 contains oil': 0.2, 'Well 2 contains oil': 0.9, 'Well 3 contains oil': 0.9, 'Well 4 contains oil': 0.9}},
                     'Test Well 2':{'Microbe': {'Well 1 contains oil': 0.1, 'Well 2 contains oil': 0.8, 'Well 3 contains oil': 0.1, 'Well 4 contains oil': 0.1},
                                    'No microbe': {'Well 1 contains oil': 0.9, 'Well 2 contains oil': 0.2, 'Well 3 contains oil': 0.9, 'Well 4 contains oil': 0.9}},
                     'Test Well 3':{'Microbe': {'Well 1 contains oil': 0.1, 'Well 2 contains oil': 0.1, 'Well 3 contains oil': 0.8, 'Well 4 contains oil': 0.1},
                                    'No microbe': {'Well 1 contains oil': 0.9, 'Well 2 contains oil': 0.9, 'Well 3 contains oil': 0.2, 'Well 4 contains oil': 0.9}},
                     'Test Well 4':{'Microbe': {'Well 1 contains oil': 0.1, 'Well 2 contains oil': 0.1, 'Well 3 contains oil': 0.1, 'Well 4 contains oil': 0.8},
                                    'No microbe': {'Well 1 contains oil': 0.9, 'Well 2 contains oil': 0.9, 'Well 3 contains oil': 0.9, 'Well 4 contains oil': 0.2}}}

    getMarginalOfEvidenceGivenTest=lambda test, evidence: getMarginalOfData(prior, likelihoodTable[test][evidence]) 
    getMaxEU=lambda test, evidence:getMaxEUFull(evidence, prior, likelihoodTable[test], rewardTable, actionSpace)
    
    getValueOfInformation=lambda test: getValueOfInformationOfATest(prior, evidenceSpace,
                                                                    lambda evidence: getMarginalOfEvidenceGivenTest(test, evidence), 
                                                                    lambda evidence: getMaxEU(test, evidence))
    
    testExample1='Test Well 1'
    print(getValueOfInformation(testExample1))
    
    testExample2='Test Well 2'
    print(getValueOfInformation(testExample2))

if __name__=="__main__":
    main()      
