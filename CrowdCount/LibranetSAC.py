import torch.nn as nn
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint   
import numpy as np
import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import random 

#Helper Functions
def sample_n_unique(sampling_f, n1):
    res = []
    while len(res) < n1:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

#Backbone (do not touch)
class VGG16_BackBone(nn.Module):
    def __init__(self):
        super(VGG16_BackBone, self).__init__()
        self.layer0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)        
        self.layer1 = nn.ReLU(inplace=True)        
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.layer3 = nn.ReLU(inplace=True)            
        self.layer4 = nn.MaxPool2d(kernel_size=2, stride=2) 
# =============================================================================        
        self.layer5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.layer6 = nn.ReLU(inplace=True)      
        self.layer7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  
        self.layer8 = nn.ReLU(inplace=True)              
        self.layer9 = nn.MaxPool2d(kernel_size=2, stride=2) 
# =============================================================================        
        self.layer10 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1) 
        self.layer11 = nn.ReLU(inplace=True)      
        self.layer12 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1) 
        self.layer13 = nn.ReLU(inplace=True)      
        self.layer14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1) 
        self.layer15 = nn.ReLU(inplace=True)              
        self.layer16 = nn.MaxPool2d(kernel_size=2, stride=2) 
               
# =============================================================================        
        self.layer17 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1) 
        self.layer18 = nn.ReLU(inplace=True)      
        self.layer19 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.layer20 = nn.ReLU(inplace=True)      
        self.layer21 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) 
        self.layer22 = nn.ReLU(inplace=True)              
        self.layer23 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
# =============================================================================
        self.layer24 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) 
        self.layer25 = nn.ReLU(inplace=True)   
        self.layer26 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.layer27 = nn.ReLU(inplace=True)    
        self.layer28 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) 
        self.layer29 = nn.ReLU(inplace=True)
        self.layer30 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data,0.01)
                
    def forward(self, x):
        x = checkpoint(self.layer0,x)         
        x = self.layer1(x)  
        x = checkpoint(self.layer2,x)  
        x = self.layer3(x)  
        x = checkpoint(self.layer4,x)  
        x = checkpoint(self.layer5,x)  
        x = self.layer6(x)  
        x = checkpoint(self.layer7,x)  
        x = self.layer8(x)  
        x = checkpoint(self.layer9,x)  
        x = checkpoint(self.layer10,x)  
        x = self.layer11(x)  
        x = checkpoint(self.layer12,x)  
        x = self.layer13(x)  
        x = checkpoint(self.layer14,x)  
        x = self.layer15(x)  
        x = checkpoint(self.layer16,x)   
                 
        x = checkpoint(self.layer17,x)             
        x = self.layer18(x)  
        x = checkpoint(self.layer19,x) 
        x = self.layer20(x)  
        x = checkpoint(self.layer21,x)  
        x = self.layer22(x)          
        x = checkpoint(self.layer23,x)  
                         
        x = checkpoint(self.layer24,x)  
        x = self.layer25(x)  
        x = checkpoint(self.layer26,x)  
        x = self.layer27(x)  
        x = checkpoint(self.layer28,x)  
        x = self.layer29(x)    
        x = checkpoint(self.layer30,x)  
        return x

#SECTION -  Replay Buffer (try to avoid modifying)
class ReplayBuffer(object):
    def __init__(self, size, vector_len_fv,vector_len_hv,batch_size):
        
        self.size = size
        self.batch_size=batch_size
        self.next_idx      = 0
        self.num_in_buffer = 0
        self.state_fv  = np.zeros((size, vector_len_fv))
        self.state_hv  = np.zeros((size, vector_len_hv))
        self.action     = np.zeros((size,1))
        self.reward     = np.zeros((size,1))
        self.next_state_hv  = np.zeros((size, vector_len_hv))
        self.done       = np.zeros((size,1))
        self.flag_full =0

    def can_sample(self):
        
        return self.flag_full>0

    def out(self):
        
        assert self.can_sample()
        
        idxes = sample_n_unique(lambda: random.randint(0, 
                            self.size  - 2), self.batch_size)
        state_fv_batch  = self.state_fv[idxes]
        state_hv_batch  = self.state_hv[idxes]
        next_state_hv_batch  = self.next_state_hv[idxes]
        act_batch   = self.action[idxes]
        rew_batch   = self.reward[idxes]
        done_mask   = self.done[idxes]

        return state_fv_batch,state_hv_batch, act_batch, rew_batch,next_state_hv_batch, done_mask

    def put(self, state_fv,state_hv, action, reward,  next_state_hv,  done):
            
        length=len(state_fv)
        
        if self.size-self.num_in_buffer>length:
            
            self.state_fv[self.num_in_buffer:self.num_in_buffer+length,:]  = state_fv
            self.state_hv[self.num_in_buffer:self.num_in_buffer+length,:]  = state_hv
            self.action[self.num_in_buffer:self.num_in_buffer+length,:]  = action
            self.reward[self.num_in_buffer:self.num_in_buffer+length,:]  = reward
            self.next_state_hv[self.num_in_buffer:self.num_in_buffer+length,:]  = next_state_hv
            self.done[self.num_in_buffer:self.num_in_buffer+length,:]  = done
            
            self.num_in_buffer=self.num_in_buffer+length
            
        else:
            
            self.flag_full=1
            buffer_int=self.size-self.num_in_buffer
            self.state_fv[self.num_in_buffer:self.size,:]  = state_fv[0:buffer_int,:]
            self.state_hv[self.num_in_buffer:self.size,:]  = state_hv[0:buffer_int,:]
            self.action[self.num_in_buffer:self.size,:]  = action[0:buffer_int,:]
            self.reward[self.num_in_buffer:self.size,:]  = reward[0:buffer_int,:]
            self.next_state_hv[self.num_in_buffer:self.size,:]  = next_state_hv[0:buffer_int,:]
            self.done[self.num_in_buffer:self.size,:]  = done[0:buffer_int,:]
            
            buffer_int2=length-buffer_int
            self.state_fv[0:buffer_int2,:]  = state_fv[buffer_int:length,:]
            self.state_hv[0:buffer_int2,:]  = state_hv[buffer_int:length,:]
            self.action[0:buffer_int2,:]  = action[buffer_int:length,:]
            self.reward[0:buffer_int2,:]  = reward[buffer_int:length,:]
            self.next_state_hv[0:buffer_int2,:]  = next_state_hv[buffer_int:length,:]
            self.done[0:buffer_int2,:]  = done[buffer_int:length,:]
            
            self.num_in_buffer =buffer_int2

#SECTION - Supporting networks (in progress)
#Step Two: Create the Critic Network
#REVIEW - Ensure that the only changes necessary have been made
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, name = "critic", chkpt_dir = "tmp/sac"):
        """_summary_

        Args:
            beta (_type_): learning rate
            input_dims (_type_): Dimensionality of the state
            n_actions (_type_): Dimensionality of action space
            name (_type_): name for the checkpoint file
            chkpt_dir (str, optional): _description_. Defaults to "tmp/sac".
        """
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_sac")
        print("THESE ARE THE DIMENSIONS OF THE CRITIC NETWORK: ", input_dims)
        self.ACTION_NUMBER = n_actions
        self.layer1 = nn.Conv2d(in_channels=input_dims, out_channels=1024, kernel_size=1, padding=0)        
        self.layer2 = nn.ReLU(inplace=True)  
        self.layer3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, padding=0)        
        self.layer4 = nn.ReLU(inplace=True)  
        self.layer5 = nn.Conv2d(in_channels=1024, out_channels=n_actions, kernel_size=1, padding=0) #Final Layer maps back to action space     
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data,0.01)
                
        #May want to use SGD as our network
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x, hv):
        x = [x,hv]        
        x = T.cat(x,1)
        #del hv not deleting in the critic network, because we use it several times, 
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x) 
        x = self.layer5(x) 
        return x
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
           
#Step Three: Create the Value Network 
#REVIEW - I made no changes to the implementation in this section,  ensure correctness
class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims = 256, fc2_dims = 256, name = "value", chkpt_dir = "tmp/sac"):
        """Contains the value network implementation for SAC. 

        Args:
            beta (_type_): learning rate
            input_dims (_type_): dimensionality of the states
            fc1_dims (int, optional): Dimensions of the first layer. Defaults to 256. (I lowered it for testing)
            fc2_dims (int, optional): Dimensions of the second layer. Defaults to 256. (I lowered it for testing)
            name (str, optional): name for saving of model. Defaults to "value".
            chkpt_dir (str, optional): where checkpoints are saved. Defaults to "tmp/sac".
        """
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_sac")
        
        #NOTE - I dont know why this is, however they always add 512 to input dimensions
        # print("THESE ARE THE INPUT DIMENSIONS FOR VALUE: ", input_dims)
        # self.fc1 = nn.Linear(input_dims, fc1_dims)
        # self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        # self.v = nn.Linear(fc2_dims, 1)
        #TODO - Ensure that these dimensions are correct, the program did not like linear layers
        
        self.layer1 = nn.Conv2d(in_channels=input_dims, out_channels=1024, kernel_size=1, padding=0)        
        self.layer2 = nn.ReLU(inplace=True)  
        self.layer3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, padding=0)        
        self.layer4 = nn.ReLU(inplace=True)  
        self.layer5 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, padding=0)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x, hv):
        # state_value = self.fc1(state)
        # state_value = F.relu(state_value)
        # state_value = self.fc2(state_value)
        # state_value = F.relu(state_value)
        
        # v = self.v(state_value)
        # return v
        x = [x,hv]        
        x = T.cat(x,1)
        #del hv not deleting in the critic network, because we use it several times, 
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x) 
        x = self.layer5(x) 
        return x
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        
#Step Four: Create the actor network, not to be confused with the agent
#REVIEW - In particular, I think that the sample function needs to be looked at
class ActorNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions = 2, name = 'actor', chkpt_dir = "tmp/sac"):
        """Contains the implementation of the actor network

        Args:
            beta (_type_): learning rate
            input_dims (_type_): dimensionality of the state input
            n_actions (int, optional): Size of the action space. Defaults to 2. #NOTE - unkown reason behind the default value here, look into it later
            name (str, optional): name for saving. Defaults to 'actor'.
            chkpt_dir (str, optional): location of save. Defaults to "tmp/sac".
        """
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        
        self.ACTION_NUMBER = n_actions
        self.layer1 = nn.Conv2d(in_channels=input_dims, out_channels=1024, kernel_size=1, padding=0)        
        self.layer2 = nn.ReLU(inplace=True)  
        self.layer3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, padding=0)        
        self.layer4 = nn.ReLU(inplace=True)  
        self.layer5 = nn.Conv2d(in_channels=1024, out_channels=n_actions, kernel_size=1, padding=0)         
        
        #NOTE - These were needed for a continuous action space, that is the case no longer
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data,0.01)
        
        #The biggest difference here is that we change have the optimizers in the networks
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        
    def forward(self, x, hv):
        x = [x,hv]        
        x = T.cat(x,1)
        #del hv, not deleting here either, we will handle it in code
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x) 
        x = self.layer5(x) 
        return x
    
    #NOTE - The sample normal function was just replaced with a sample function, that will returnt the highest value for now
    #TODO - Determine if there needs to be a probabilistic decision for action choice
    def sample(self, state):
        #NOTE - I dont know if Im bringing it back to the cpu properly or whether to call argmax
        aValues = self.forward(state).cpu().detach().numpy()
        maxAction = np.argmax(aValues)
        return maxAction
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        
#SECTION - LibreNet Agent (in progress)
class LibraNetSAC(nn.Module):
    def __init__(self, parameters):
        super(LibraNetSAC, self).__init__()  
        #Weights definition
        #If youre gonna update some of these, dont forget to update the parameters['action_number']
        Action1 = -10
        Action2 = -5
        Action3 = -2
        Action4 = -1
        Action5 = 1
        Action6 = 2
        Action7 = 5
        Action8 = 10
        Action9 = 999
        self.A = [Action1,Action2,Action3,Action4,Action5,Action6,Action7,Action8,Action9]
        self.A_mat = np.array(self.A)
        
        self.A_mat_h_w = np.expand_dims(np.expand_dims(self.A_mat, 1), 2)
        
        #Inverse discretization vector
        #TODO - Find where this vector is used
        self.class2num = np.zeros(parameters['Interval_N'])
        for i in range(1, parameters['Interval_N']):
            if i == 1:
                lower = 0
            else:
                lower = np.exp((i - 2) * parameters['step_log'] + parameters['start_log'])
            upper = np.exp((i - 1) * parameters['step_log'] + parameters['start_log'])
            self.class2num[i] = (lower + upper) / 2
        
        #NOTE parameters added for SAC.
        #parameters created elsewhere:
            #1. replay buffer, created in train.py
            #2. Batch size, used where replay buffer is created
        #TODO - update the parameters to include these items
        self.alpha = parameters['ALPHA']
        self.gamma = parameters['GAMMA']
        self.tau = parameters['TAU']
        self.scale = parameters['SCALE']
        self.beta = parameters['BETA']
        #self.n_actions = len(self.A) #for now, we will use parameters['ACTION_NUMBER']
        
        #Creating the actor, critic, and value networks
        #TODO - Figure out how to get the dimensions of the states and put it in input_dims
        #For now, we will assume based on initial DQN creation that it is HV_NUMBER+512
        self.actor = ActorNetwork(self.alpha, parameters['HV_NUMBER']+512, n_actions = parameters['ACTION_NUMBER'], name = "actor")
        
        self.critic_1 = CriticNetwork(self.beta, parameters['HV_NUMBER']+512, n_actions = parameters['ACTION_NUMBER'], name = "critic_1")
        self.critic_2 = CriticNetwork(self.beta, parameters['HV_NUMBER']+512, n_actions = parameters['ACTION_NUMBER'], name = "critic_2")
        
        self.value = ValueNetwork(self.beta, parameters['HV_NUMBER']+512, name = "value")
        self.target_value = ValueNetwork(self.beta, parameters['HV_NUMBER']+512, name = "target_value")
        
        #Network definition, figure out where the backbone fits into the equation
        self.backbone = VGG16_BackBone()      
        
        #TODO - Ensure that this function is setup properly for this framework 
        #for the initialization, tau = 1, in accordance with the SAC video I watched
        self.update_network_parameters(tau = 1)
        
        # self.DQN = DQN(parameters['ACTION_NUMBER'], parameters['HV_NUMBER'])
        # self.DQN_faze = DQN(parameters['ACTION_NUMBER'], parameters['HV_NUMBER'])
        
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

    def get_Q(self, feature=None, history_vectory=None):
        return self.actor(feature,history_vectory) * 100
    
    def get_feature( self, im_data=None):
        return self.backbone(im_data)
    

#I actually do not know what this function does, might need to investigate later
def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                #print torch.sum(m.weight)
                m.weight.data.normal_(0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.ConvTranspose2d):                
                #print torch.sum(m.weight)
                m.weight.data.normal_(0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, dev)
