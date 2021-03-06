def expect(xDistribution, function):
    fxProduct=[px*function(x) for x, px in xDistribution.items()]
    expectation=sum(fxProduct)
    return expectation


def forward(xT_1Distribution, eT, transitionTable, sensorTable):
    
##################################################
#		Your code here
################################################## 

    # implementation of the formula
    unnormPx = {x:sensorTable[x][eT]*sum([transitionTable[xt][x]*xT_1Distribution[xt] for xt in xT_1Distribution]) for x in sensorTable}

    # normalization constant
    normConst = sum([unnormPx[x] for x in unnormPx])

    # return value is a dictionary representing belief distribution
    return {x: unnormPx[x]/normConst for x in unnormPx}

def main():
    
    pX0={0:0.3, 1:0.7}
    e=1
    transitionTable={0:{0:0.6, 1:0.4}, 1:{0:0.3, 1:0.7}}
    sensorTable={0:{0:0.6, 1:0.3, 2:0.1}, 1:{0:0, 1:0.5, 2:0.5}}
    
    xTDistribution=forward(pX0, e, transitionTable, sensorTable)
    print(xTDistribution)

if __name__=="__main__":
    main()
