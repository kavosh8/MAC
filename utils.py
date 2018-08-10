import numpy, sys

def rewardToReturn(rewards,gamma):
    T=len(rewards)
    returns=T*[0]
    returns[T-1]=rewards[T-1] 
    for t in range(T-2,-1,-1):
        returns[t]=rewards[t]+gamma*returns[t+1]
    return returns