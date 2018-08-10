import mac,numpy,gym,sys, random
import tensorflow as tf


meta_params={}
meta_params['seed_number']=0
meta_params['env']=gym.make('CartPole-v0')
meta_params['max_learning_episodes']=1000
meta_params['max_time_steps']=200
meta_params['gamma']=0.99



alg_params={}
alg_params['|A|']=meta_params['env'].action_space.n
alg_params['state_|dimension|']=len(meta_params['env'].observation_space.sample())
alg_params['critic_num_h']=1
alg_params['critic_|h|']=64
alg_params['critic_lr']=0.01
alg_params['critic_num_epochs']=10
alg_params['actor_num_h']=1
alg_params['actor_|h|']=64
alg_params['actor_lr']=0.001
alg_params['batch_size']=32

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

agent=mac.mac(alg_params)
agent.train(meta_params)
sys.exit(1)
