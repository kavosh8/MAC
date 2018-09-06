import numpy
import critic_network,actor_network
import gym,sys,utils, os
from collections import deque

class mac:
	'''
	a class representing the Mean Actor-Critic algorithm.
	It contains and actor and a critic + a train function.
	'''
	def __init__(self,params):
		'''
		This initializes MAC agent by creating an actor
		a critic.
		'''
		self.params=params
		self.memory = deque(maxlen=self.params['max_buffer_size'])
		self.actor=actor_network.actor(self.params)
		self.critic=critic_network.critic(self.params)

	def add_2_memory(self,states,actions,rewards):
		T=len(states)
		for index,(s,a,r) in enumerate(zip(states,actions,rewards)):
			if index<T-1:
				self.memory.append( (s,a,r,states[index+1],T-index-1) )

	def train(self,meta_params):
		'''
		This function trains a MAC agent for max_learning_episodes episodes.
		It proceeds with interaction for one episode, followed by critic
		and actor updates.
		'''
		print("training has begun...")
		li_episode_length=[]
		li_returns=[]
		for episode in range(1,meta_params['max_learning_episodes']):
			states,actions,returns,rewards=self.interactOneEpisode(meta_params,episode)
			self.add_2_memory(states,actions,rewards)
			#log performance
			li_episode_length.append(len(states))
			if episode%10==0:
				print(episode,"return in last 10 episodes",numpy.mean(li_returns[-10:]))
			li_returns.append(returns[0])
			sys.stdout.flush()
			#log performance

			#train the Q network
			if self.params['critic_train_type']=='model_free_critic_monte_carlo':
				self.critic.train_model_free_monte_carlo(states,actions,returns)
			elif self.params['critic_train_type']=='model_free_critic_TD':
				self.critic.train_model_free_TD(self.memory,self.actor,meta_params,self.params,episode)
			self.actor.train(states,self.critic)
			#train the Q network
			
		print("training is finished successfully!")
		return

	def interactOneEpisode(self,meta_params,episode):
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
			
			if episode%50==0:
				meta_params['env'].render()
			
			states.append(s),actions.append(a),rewards.append(r)
			s,t=(s_p,t+1)
			if done==True:
				states.append(s_p)#we actually store the terminal state!
				a=self.actor.select_action(s_p)
				actions.append(a),rewards.append(0)
				break
		returns=utils.rewardToReturn(rewards,meta_params['gamma'])
		return states,actions,returns,rewards
