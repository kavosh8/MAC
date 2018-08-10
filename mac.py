import numpy
import critic_network
import actor_network
import gym
import sys
import utils
import tensorflow as tf

class mac:
	'''
	a class representing the Mean Actor-Critic algorithm.
	It contains and actor and a critic + a train function.
	'''
	params={}
	actor=0
	critic=0
	def __init__(self,params):
		self.params=params
		self.actor=actor_network.actor(self.params)
		self.critic=critic_network.critic(self.params)

	def train(self,meta_params):
		li_episode_length=[]
		for episode in range(1,meta_params['max_learning_episodes']):
			states,actions,returns,rewards=self.interactOneEpisode(meta_params)
			li_episode_length.append(len(returns))
			self.critic.train(states,actions,returns)
			self.actor.train(states,self.critic)

			#log performance
			if episode%10==0:
				print(episode,numpy.mean(li_episode_length[-10:]))
			#log performance
		return li_episode_length

	def interactOneEpisode(self,meta_params):
		'''
			given the mac agent and an environment, executes their
			interaction for an episode, and then returns important information
			used for training.
		'''
		s0,rewards,states,actions,t=meta_params['env'].reset(),[],[],[],0
		s=s0
		while True:
			a=self.actor.select_action(s)
			s_p,r,done,_= meta_params['env'].step(a)
			if t>meta_params['max_time_steps']:
				done=True
			states.append(s),actions.append(a),rewards.append(r)
			s,t=(s_p,t+1)
			if done==True:
				break
		returns=utils.rewardToReturn(rewards,meta_params['gamma'])
		return states,actions,returns,rewards	
