import mac,numpy,gym,sys,random
import tensorflow as tf
import utils

#get and set hyper-parameters
meta_params,alg_params={},{}
try:
	meta_params['env_name']=sys.argv[1]
	meta_params['seed_number']=int(sys.argv[2])
except:
	print("default environment is Lunar Lander ...")
	meta_params['env_name']='LunarLander-v2'
	meta_params['seed_number']=0

meta_params['env']=gym.make(meta_params['env_name'])

if meta_params['env_name']=='CartPole-v0':
	meta_params['max_learning_episodes']=200
	meta_params['gamma']=0.9999
	meta_params['plot']=False
	alg_params={}
	alg_params['|A|']=meta_params['env'].action_space.n
	alg_params['state_|dimension|']=len(meta_params['env'].reset())
	alg_params['critic_num_h']=1
	alg_params['critic_|h|']=64
	alg_params['critic_lr']=0.01
	alg_params['actor_num_h']=1
	alg_params['actor_|h|']=64
	alg_params['actor_lr']=0.005
	alg_params['critic_batch_size']=32
	alg_params['critic_num_epochs']=10
	alg_params['critic_target_net_freq']=1
	alg_params['max_buffer_size']=2000
	alg_params['critic_train_type']='model_free_critic_TD'#or model_free_critic_monte_carlo

if meta_params['env_name']=='LunarLander-v2':
	meta_params['max_learning_episodes']=3000
	meta_params['gamma']=0.9999
	meta_params['plot']=False
	alg_params={}
	alg_params['|A|']=meta_params['env'].action_space.n
	alg_params['state_|dimension|']=len(meta_params['env'].reset())
	alg_params['critic_num_h']=1
	alg_params['critic_|h|']=64
	alg_params['critic_lr']=0.005
	alg_params['actor_num_h']=1
	alg_params['actor_|h|']=64
	alg_params['actor_lr']=0.0005
	alg_params['critic_batch_size']=32
	alg_params['critic_num_epochs']=10
	alg_params['critic_target_net_freq']=1
	alg_params['max_buffer_size']=5000
	alg_params['critic_train_type']='model_free_critic_TD'#or model_free_critic_monte_carlo

#ensure results are reproducible
numpy.random.seed(meta_params['seed_number'])
random.seed(meta_params['seed_number'])
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(meta_params['seed_number'])
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
meta_params['env'].seed(meta_params['seed_number'])
#ensure results are reproducible

#create a MAC agent and run
agent=mac.mac(alg_params)
agent.train(meta_params)
#create a MAC agent and run

