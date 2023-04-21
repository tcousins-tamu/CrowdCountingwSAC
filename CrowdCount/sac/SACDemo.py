#This is a soft actor critic implementation in accordance with this video: https://www.youtube.com/watch?v=ioidsRlf79o
#This exercise was done so that I would get further understanding of the soft actor critic algorithm

import numpy as np

import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

#Step One: generating the replay buffer
class ReplayBuffer():
    #NOTE - In this implementation we are working in a continuous environment, so num_actions corresponds to the number of components of an action
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0 #Keeps track of first available memerory
        self.state_memory = np.zeros((self.mem_size, *input_shape)) 
        self.new_state_memory = np.zeros((self.mem_size, *input_shape)) #Keeps track of resultant state
        self.action_memory = np.zeros((self.mem_size, n_actions))#Keeps track of actions 
        self.reward_memory = np.zeros(self.mem_size)#Keeps track of received rewards
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)#Used for setting tghe value of the terminal states
        
    def store_transistion(self, state, action, reward, state_, done):
        """This function will store the transition and relevant information into the replay buffer, to be accessed later

        Args:
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            state_ (_type_): _description_
            done (function): _description_
        """
        index = self.mem_cntr%self.mem_size
        
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        """Samples batch size from the replay buffer

        Args:
            batch_size (_type_): _description_

        Returns:
            _type_: _description_
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        
        batch = np.random.choice(max_mem, batch_size)
        
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, dones
    
#Step Two: Create the Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims, fc2_dims, name = "critic", chkpt_dir = "tmp/sac"):
        """_summary_

        Args:
            beta (_type_): learning rate
            input_dims (_type_): _description_
            n_actions (_type_): _description_
            fc1_dims (_type_): _description_
            fc2_dims (_type_): _description_
            name (_type_): _description_
            chkpt_dir (str, optional): _description_. Defaults to "tmp/sac".
        """
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_sac")
        
        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims) #defines a linear layer (y=mx+b)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1) #when I go back and implement this for the Librenet, this will cast back into an action space
        
        #May want to use SGD as our network
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cude:0' if T.cuda_is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        
        q = self.q(action_value)
        return q
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        
#Step Three: Create the Value Network
class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims = 256, fc2_dims = 256, name = "value", chkpt_dir = "tmp/sac"):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpointdir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_sac")
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        
        v = self.v(state_value)
        return v
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        
#Step Four: Create the actor network
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, fc2_dims=256, n_actions = 2, name = 'actor', chkpt_dir = "tmp/sac"):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        
    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        
        sigma = T.clamp(sigma, min = self.reparam_noise, max = 1)
        return mu, sigma
    
    #This is associated with a continuous action space
    def sample_normal(self, state, reparameterize = True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)
        
        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
            
        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim = True)
        
        return action, log_probs
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        
#Step Five: Implementing the agent that will be using these networks. This will be the equivalent of Librenet in the code
#Im writing it such taht I can make a comparison
class Agent():
    def __init__(self, alpha=.0003, beta = .0003, input_dims = [8], env = None, gamma = .99, n_actions = 2, max_size = 100000, tau = .005, layer1_size  = 256, layer2_size = 256, batch_size = 256, reward_scale = 2):
        """Ballin

        Args:
            alpha (float, optional): _description_. Defaults to .0003.
            beta (float, optional): _description_. Defaults to .0003.
            input_dims (list, optional): _description_. Defaults to [8].
            env (_type_, optional): _description_. Defaults to None.
            gamma (float, optional): _description_. Defaults to .99.
            n_actions (int, optional): _description_. Defaults to 2.
            max_size (int, optional): _description_. Defaults to 100000.
            tau (float, optional): _description_. Defaults to .005.
            layer1_size (int, optional): _description_. Defaults to 256.
            layer2_size (int, optional): _description_. Defaults to 256.
            batch_size (int, optional): _description_. Defaults to 256.
            reward_scale (int, optional): _description_. Defaults to 2.
        """
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        
        #Actor Network
        self.actor = ActorNetwork(alpha, input_dims, n_actions = n_actions, name = "actor", max_action = env.action_space.high)
        
        #Critic Network
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions = n_actions, name = "critic_1")
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions = n_actions, name = "critic_2")
        
        #Value Network
        self.value = ValueNetwork(beta, input_dims, name = 'value')
        self.target_value = ValueNetwork(beta, input_dims, name = 'target_value')
        
        self.scale = reward_scale
        self.update_network_parameters(tau = 1)
        
    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device) #sending to the GPU, guess this is not needed to be done later
        actions, _ = self.actor.sample_normal(state, reparameterize = False)
        
        return actions.cpu().detach().numpy()[0] #detach to prevent tracking the gradients
    

    def remember(self, state, action, reward, new_state, done):
        #send to replay buffer, this is done in the Librenet stuff separately I believe
        self.memory.store_transistion(state, action, reward, new_state, done)
        
    def update_network_parameters(self, tau = None):
        if tau is None:
            tau = self.tau
        
        #returns the weights and biases of each Value Network along with its layers
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()
        
        target_value_state_dict = dict(target_value_params)
        value_state_dict  = dict(value_params)
        
        #Going through each layer and performing a SOFT update
        for name in value_state_dict:
            value_state_dict[name]  = tau*value_state_dict[name].clone() + \
                (1-tau)*target_value_state_dict[name].clone()
                
        self.target_value.load_state_dict(value_state_dict)
    
    def save_models(self):
        #Will need to change the name when I bring it over to Librenet
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        
    def load_models(self):
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done = \
            self.memory.sample(self.batch_size)
        
        #converting to tensors, remember that pytorch is annoying about dtypes
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype = T.float).to(self.actor.device)
        state = T.tensor(state, dtype = T.float).to(self.actor.device)
        action = T.tensor(action, dtype = T.float).to(self.actor.device)
        
        #VALUE NETWORK LOSS
        #flattening
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done]=0.0
        
        #get actions and log probs NOTE log probs are not needed for our implementation
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy) #prevents overestimation, similiar to double DQN
        critic_value = critic_value.view(-1)
        
        #Loss Calculation
        self.value_optimizer.zero_grad() #zeroes out the gradients that pytorch calculates by default
        value_target = critic_value - log_probs #computing the gradients with respect to the model
        value_loss = .5*F.mse_loss(value, value_target)
        value_loss.backward(retain_graph = True) #we need to retain it for followup calcs
        self.value.optimizer.step()
        
        #ACTOR NETWORK LOSS
        action, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1) #remember that this will be removed in the Librenet calc
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value #again, something that will be changed for librenet
        actor_loss  =T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph = True)
        self.actor.optmizer.step()
        
        #Critic network loss 
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward +self.gamma*value_ #handles inclusion of ENTROPY IN LOSS FUNCTION THIS IS WHERE IT BECOMES RELEVANT
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = .5*F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = .5*F.mse_loss(q2_old_policy, q_hat)
        
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        self.update_network_parameters()