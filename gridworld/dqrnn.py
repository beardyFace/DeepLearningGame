#heavily based off https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-6-partial-observability-and-deep-recurrent-q-68463e9aeefc

import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
import tensorflow.contrib.slim as slim
import cv2
import pygame

class Qnetwork():
    def __init__(self,h_size,rnn_cell,myScope):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        self.scalarInput =  tf.placeholder(shape=[None,21168],dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,3])
        self.conv1 = slim.convolution2d( \
            inputs=self.imageIn,num_outputs=32,\
            kernel_size=[8,8],stride=[4,4],padding='VALID', \
            biases_initializer=None,scope=myScope+'_conv1')
        self.conv2 = slim.convolution2d( \
            inputs=self.conv1,num_outputs=64,\
            kernel_size=[4,4],stride=[2,2],padding='VALID', \
            biases_initializer=None,scope=myScope+'_conv2')
        self.conv3 = slim.convolution2d( \
            inputs=self.conv2,num_outputs=64,\
            kernel_size=[3,3],stride=[1,1],padding='VALID', \
            biases_initializer=None,scope=myScope+'_conv3')
        self.conv4 = slim.convolution2d( \
            inputs=self.conv3,num_outputs=h_size,\
            kernel_size=[7,7],stride=[1,1],padding='VALID', \
            biases_initializer=None,scope=myScope+'_conv4')
        
        self.trainLength = tf.placeholder(dtype=tf.int32)
        #We take the output from the final convolutional layer and send it to a recurrent layer.
        #The input must be reshaped into [batch x trace x units] for rnn processing, 
        #and then returned to [batch x units] when sent through the upper levles.
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])
        self.convFlat = tf.reshape(slim.flatten(self.conv4),[self.batch_size,self.trainLength,h_size])
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn,self.rnn_state = tf.nn.dynamic_rnn(\
                inputs=self.convFlat,cell=rnn_cell,dtype=tf.float32,initial_state=self.state_in,scope=myScope+'_rnn')
        self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])
        
        #The output from the recurrent player is then split into separate Value and Advantage streams
        self.streamA,self.streamV = tf.split(self.rnn,2,1)
        self.AW = tf.Variable(tf.random_normal([h_size//2,4]))
        self.VW = tf.Variable(tf.random_normal([h_size//2,1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)
        
        self.salience = tf.gradients(self.Advantage,self.imageIn)
        #Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        self.predict = tf.argmax(self.Qout,1)

        self.Temp = tf.placeholder(shape=[None],dtype=tf.float32)
        self.Q_dist = slim.softmax(self.Qout/self.Temp)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,4,dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        
        #In order to only propogate accurate gradients through the network, we will mask the first
        #half of the losses for each trace as per Lample & Chatlot 2016
        self.maskA = tf.zeros([self.batch_size,self.trainLength//2])
        self.maskB = tf.ones([self.batch_size,self.trainLength//2])
        self.mask = tf.concat([self.maskA,self.maskB],1)
        self.mask = tf.reshape(self.mask,[-1])
        self.loss = tf.reduce_mean(self.td_error * self.mask)
        
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

class experience_buffer():
    def __init__(self, buffer_size = 1000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []
        self.buffer.append(experience)
            
    def sample(self,batch_size,trace_length):
        sampled_episodes = random.sample(self.buffer,batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0,len(episode)+1-trace_length)
            sampledTraces.append(episode[point:point+trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces,[batch_size*trace_length,5])

class DQRNLearner():
    BATCH_SIZE = 4 #How many experience traces to use for each training step.
    TRACE_LENGTH = 8 #How long each experience trace will be when training
    UPDATE_FREQ = 5 #How often to perform a training step.
    Y = .99 #Discount factor on the target Q-values
    INITIAL_RANDOM_ACTION_PROB = 1 #Starting chance of random action
    FINAL_RANDOM_ACTION_PROB = 0.01 #Final chance of random action
    ANNELING_STEPS = 100000 #How many steps of training to reduce startE to endE.
    PRE_TRAIN_STEPS = 10000 #How many steps of random actions before training begins.
    LOAD_MODEL = False #Whether to load a saved model.
    PATH = "./drqn" #The path to save our model to.
    H_SIZE = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
    
    SAVE_EVERY_X_STEPS = 1000 #Number of epidoes to periodically save for analysis
    TAU = 0.001

    def __init__(self, scope, checkpoint_path="deep_qrnn_spaceshooter_networks", playback_mode=False, verbose_logging=False):
        tf.reset_default_graph()
        #We define the cells for the primary and target q-networks
        self.cell = tf.contrib.rnn.BasicLSTMCell(num_units=DQRNLearner.H_SIZE,state_is_tuple=True)
        self.cellT = tf.contrib.rnn.BasicLSTMCell(num_units=DQRNLearner.H_SIZE,state_is_tuple=True)
        
        self.mainQN = Qnetwork(DQRNLearner.H_SIZE,self.cell,scope+'_main')
        self.targetQN = Qnetwork(DQRNLearner.H_SIZE,self.cellT,scope+'_target')

        self.state = (np.zeros([1,DQRNLearner.H_SIZE]),np.zeros([1,DQRNLearner.H_SIZE])) #Reset the recurrent layer's hidden state

        self.trainables = tf.trainable_variables()
        self.targetOps = self.updateTargetGraph(self.trainables,DQRNLearner.TAU)

        self.myBuffer = experience_buffer()
        self.episode_buffer = []

        #Set the rate of random action decrease. 
        self.random_action_prob = DQRNLearner.INITIAL_RANDOM_ACTION_PROB
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.session.run(tf.global_variables_initializer())

        self.checkpoint_path = checkpoint_path
        #Make a path for our model to be saved in.      
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)

        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Loaded checkpoints %s" % checkpoint.model_checkpoint_path)
        elif playback_mode:
            raise Exception("Could not load checkpoints for playback")

        #create lists to contain total rewards and steps per episode
        self.rList = []
        self.rAll = 0
        self.total_steps = 0
        self.episodes = 0

        self.last_action = None
        self.game_state = None

    def get_keys_pressed(self,game_state,reward,terminal):
        game_state = cv2.resize(game_state,(84, 84))
        game_state = self.processState(game_state)

        if self.last_action != None:
            self.train(self.game_state, self.last_action, game_state, reward, terminal)

        if terminal:
            #Add the episode to the experience buffer
            bufferArray = np.array(self.episode_buffer)
            self.episode_buffer = list(zip(bufferArray))
            self.myBuffer.add(self.episode_buffer)
            self.rList.append(self.rAll)

            self.episodes += 1
            #Periodically save the model. 
            summary_length = 100
            if self.episodes % summary_length == 0 and self.total_steps != 0:
                self.saver.save(self.session, self.checkpoint_path + '/network', global_step=self.total_steps)
                print ("Saved Model")
                print (self.episodes,self.total_steps,np.mean(self.rList[-summary_length:]), self.random_action_prob)
            
            h_size = DQRNLearner.H_SIZE
            self.state = (np.zeros([1,h_size]),np.zeros([1,h_size])) #Reset the recurrent layer's hidden state
            self.last_action = None
            self.game_state = None
            self.episode_buffer = []
            self.rAll = 0
            return None

        # action = np.random.randint(0,4)
        # if np.random.rand(1) < self.random_action_prob or self.total_steps < DQRNLearner.PRE_TRAIN_STEPS:
        #     state1 = self.session.run(self.mainQN.rnn_state,\
        #         feed_dict={self.mainQN.scalarInput:[game_state/255.0],self.mainQN.trainLength:1,self.mainQN.state_in:self.state,self.mainQN.batch_size:1})
        # else:
        t = [self.random_action_prob]
        q_out, action, state1 = self.session.run([self.mainQN.Q_dist, self.mainQN.predict, self.mainQN.rnn_state],\
                feed_dict={self.mainQN.scalarInput:[game_state/255.0],self.mainQN.trainLength:1,self.mainQN.state_in:self.state,self.mainQN.batch_size:1,self.mainQN.Temp:t})

        # Use this for action selection.
        action_value = np.random.choice(q_out[0],p=q_out[0])
        action = np.argmax(q_out[0] == action_value)

        self.state = state1
        self.last_action = action
        self.game_state = game_state

        key_action = self.key_presses_from_action(action)
            
        return key_action

            ##PASS THROUGH LAST ACTION HERE INTO a
            # s1P,r,d = env.step(a)
            # s1 = processState(s1P)
            # total_steps += 1

            # last_state, action, reward, new_state, terminal
            # episodeBuffer.append(np.reshape(np.array([s,a,r,s1,d]),[1,5]))
    def train(self,game_state,action,game_state1,reward,terminal):
        # print('Total steps: '+str(self.total_steps))        
        self.total_steps += 1
        self.episode_buffer.append(np.reshape(np.array([game_state,action,reward,game_state1,terminal]),[1,5]))

        if self.total_steps > DQRNLearner.PRE_TRAIN_STEPS:
            if self.random_action_prob > DQRNLearner.FINAL_RANDOM_ACTION_PROB:
                self.random_action_prob -= (DQRNLearner.INITIAL_RANDOM_ACTION_PROB - DQRNLearner.FINAL_RANDOM_ACTION_PROB) / DQRNLearner.ANNELING_STEPS
                # print('Random action prob: '+str(self.random_action_prob))

            batch_size = DQRNLearner.BATCH_SIZE
            h_size = DQRNLearner.H_SIZE
            trace_length = DQRNLearner.TRACE_LENGTH
            y = DQRNLearner.Y
            if self.total_steps % (DQRNLearner.UPDATE_FREQ) == 0:
                self.updateTarget(self.targetOps)
                #Reset the recurrent layer's hidden state
                state_train = (np.zeros([batch_size,h_size]),np.zeros([batch_size,h_size])) 
                
                trainBatch = self.myBuffer.sample(DQRNLearner.BATCH_SIZE,DQRNLearner.TRACE_LENGTH) #Get a random batch of experiences.
                #Below we perform the Double-DQN update to the target Q-values
                Q1 = self.session.run(self.mainQN.predict,feed_dict={\
                    self.mainQN.scalarInput:np.vstack(trainBatch[:,3]/255.0),\
                    self.mainQN.trainLength:trace_length,self.mainQN.state_in:state_train,self.mainQN.batch_size:batch_size})

                Q2 = self.session.run(self.targetQN.Qout,feed_dict={\
                    self.targetQN.scalarInput:np.vstack(trainBatch[:,3]/255.0),\
                    self.targetQN.trainLength:trace_length,self.targetQN.state_in:state_train,self.targetQN.batch_size:batch_size})

                end_multiplier = -(trainBatch[:,4] - 1)
                doubleQ = Q2[range(batch_size*trace_length),Q1]
                targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                #Update the network with our target values.
                self.session.run(self.mainQN.updateModel, \
                    feed_dict={self.mainQN.scalarInput:np.vstack(trainBatch[:,0]/255.0),self.mainQN.targetQ:targetQ,\
                    self.mainQN.actions:trainBatch[:,1],self.mainQN.trainLength:DQRNLearner.TRACE_LENGTH,\
                    self.mainQN.state_in:state_train,self.mainQN.batch_size:DQRNLearner.BATCH_SIZE})
        
        self.rAll += reward

    #These functions allows us to update the parameters of our target network with those of the primary network.
    def updateTargetGraph(self,tfVars,tau):
        total_vars = len(tfVars)
        op_holder = []
        for idx,var in enumerate(tfVars[0:total_vars//2]):
            op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
        return op_holder

    def updateTarget(self,op_holder):
        for op in op_holder:
            self.session.run(op)
        total_vars = len(tf.trainable_variables())
        a = tf.trainable_variables()[0].eval(session=self.session)
        b = tf.trainable_variables()[total_vars//2].eval(session=self.session)
        if a.all() == b.all():
            # print("Target Set Success")
            None
        else:
            print("Target Set Failed")

    def processState(self,state1):
        return np.reshape(state1,[21168])

    @staticmethod
    def key_presses_from_action(action_set):
        if action_set == 0:
            return [pygame.K_DOWN]
        elif action_set == 1:
            return [pygame.K_UP]
        elif action_set == 2:
            return [pygame.K_RIGHT]
        elif action_set == 3:
            return [pygame.K_LEFT]
        # elif action_set[4] == 1:
        #     return [pygame.K_SPACE]
        elif action_set == 4:
            return []
        raise Exception("Unexpected action")





