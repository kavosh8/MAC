import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import sys
import tensorflow as tf
import numpy, random

class critic:        
    def __init__(self,params):
        self.params=params
        self.network=self.build()
        self.target_network=self.build()
        self.target_network.set_weights(self.network.get_weights())

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

    def train_model_free_monte_carlo(self,states,actions,returns):
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
                            batch_size=self.params['critic_batch_size'],epochs=1,verbose=0)

    def train_model_free_TD(self,memory,actor,meta_params,alg_params,e):
        '''
        Train the Q network using TD-style update
        get a batch from buffer of experience, do update
        r+gamma sum pi Q
        and then update the target network
        '''
        if alg_params['critic_batch_size']>len(memory):
            return
        #self.target_network.set_weights(self.network.get_weights())
        for _ in range(alg_params['critic_num_epochs']):
            minibatch = random.sample(memory, alg_params['critic_batch_size'])
            next_states_li=[m[3] for m in minibatch]
            time_to_terminal=[m[4] for m in minibatch]
            Pi=actor.network.predict(numpy.array(next_states_li))
            Q=self.target_network.predict(numpy.array(next_states_li))
            next_states_values=numpy.sum(Pi*Q,axis=1)
            for index,t in enumerate(time_to_terminal):
                if t==1:
                    next_states_values[index]=0
            rewards=[m[2] for m in minibatch]
            actions=[m[1] for m in minibatch]
            states=[m[0] for m in minibatch]
            state_pred=self.network.predict(numpy.array(states))
            for index in range(alg_params['critic_batch_size']):
                a=actions[index]
                r=rewards[index]+meta_params['gamma']*next_states_values[index]
                state_pred[index][a]=r
            self.network.fit(x=numpy.array(states),y=state_pred,
                            batch_size=self.params['critic_batch_size'],epochs=1,verbose=0)
        if e%alg_params['critic_target_net_freq']==0:
            self.target_network.set_weights(self.network.get_weights())
