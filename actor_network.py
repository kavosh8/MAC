import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import sys
import tensorflow as tf
import numpy

def objective(y_true,y_pred):
    "policy gradient objective"
    return  -tf.reduce_mean(tf.multiply(y_true, y_pred))

class actor:
    def __init__(self,params):
        self.params=params
        self.network=self.build()
        
    def build(self):
        '''
           creates the neural network representing
           the policy a.k.a the actor. The parameters of the network
           are determined using self.params dictionary
           the activation function is Relu (excpet for last one, a softmax) and the
           optimizer is Adam
        '''
        model = Sequential()
        
        model.add(Dense(units=self.params['actor_|h|'],
                        activation='relu',
                        input_dim=self.params['state_|dimension|'])
                 )
        
        for _ in range(self.params['actor_num_h']-1):
            model.add(Dense(units=self.params['actor_|h|'], activation='relu'))

        model.add(Dense(units=self.params['|A|'], activation='softmax'))
        model.compile(loss=objective,
                      optimizer=Adam(lr=self.params['actor_lr'])
                     )
        return model

    def select_action(self,state):
        '''
            selects an action given a state. First computes \pi(.|s)
            using the neural net, and then draws an action a~\pi(.|s)
        '''
        pr=self.network.predict(numpy.array(state).reshape(1,self.params['state_|dimension|']))[0]
        a=numpy.random.choice([x for x in range(self.params['|A|'])],p=pr)
        return a

    def train(self,states,critic):
        '''
           computes the policy gradient, namely \sum_{a} \grad(\pi(a|s)) Q(s,a)
           first uses the critic to get Q(s,a) for all a
           then performs the policy gradient update. There is only one batch,
           whose size is equal to the length of the most recent episode.
        '''
        state_q=critic.network.predict(numpy.array(states))
        self.network.fit(x=numpy.array(states),y=state_q,
                        batch_size=len(states),epochs=1,verbose=0)




