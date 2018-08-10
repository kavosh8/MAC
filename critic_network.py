import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import sys
import tensorflow as tf
import numpy

class critic:        
    params=0
    network=0
    def __init__(self,params):
        self.params=params
        self.network=self.build()

    def build(self):
        '''creates the neural network representing
           action value function. The parameters of the network
           are determined using self.params dictionary
           the activation function is Relu and the
           optimizer is Adam
        '''

        model = Sequential()
        
        model.add(Dense(units=self.params['critic_|h|'],
                        activation='relu',
                        input_dim=self.params['state_|dimension|'])
                 )
        
        for _ in range(self.params['critic_num_h']-1):
            model.add(Dense(units=self.params['critic_|h|'], activation='relu'))

        model.add(Dense(units=self.params['|A|'], activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.params['critic_lr'])
                     )
        return model

    def train(self,states,actions,returns):
        '''
        trains the critic using state-action pairs
        and observed targets using the monte carlo approach! (not TD for now)
        computes q(s,a) for all states and actions, and then changes the
        taken actions' targets. This ensures there is no gradient signal
        due to other actions!
        '''  
        for e in range(self.params['critic_num_epochs']):
            state_q=self.network.predict(numpy.array(states))
            for index in range(state_q.shape[0]):
                a=actions[index]
                r=returns[index]
                state_q[index][a]=r
            self.network.fit(x=numpy.array(states),y=state_q,
                            batch_size=self.params['batch_size'],epochs=1,verbose=0)