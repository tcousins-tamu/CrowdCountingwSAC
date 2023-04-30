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

class LibraNet(nn.Module):
    def __init__(self, parameters):
        super(LibraNet, self).__init__()  
        #Weights definition
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
        self.class2num = np.zeros(parameters['Interval_N'])
        for i in range(1, parameters['Interval_N']):
            if i == 1:
                lower = 0
            else:
                lower = np.exp((i - 2) * parameters['step_log'] + parameters['start_log'])
            upper = np.exp((i - 1) * parameters['step_log'] + parameters['start_log'])
            self.class2num[i] = (lower + upper) / 2
        
        #Network definition
        self.backbone = VGG16_BackBone()      
        self.actor = Actor(parameters['ACTION_NUMBER'], parameters['HV_NUMBER'])
        
        #self.tau = parameters['TAU']
        
        self.v = CriticV(parameters['HV_NUMBER'])
        self.v_target = CriticV(parameters['HV_NUMBER'])
        
        self.q1 = CriticQ(parameters['ACTION_NUMBER'], parameters['HV_NUMBER'])
        self.q2 = CriticQ(parameters['ACTION_NUMBER'], parameters['HV_NUMBER'])
        
        #optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = parameters['lr'])
        self.v_optimizer = optim.Adam(self.v.parameters(),lr = parameters['lr'])
        self.vtgt_optimizer = optim.Adam(self.v_target.parameters(), lr = parameters['lr'])
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr = parameters['lr'])
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr = parameters['lr'])

        # self.update_network_parameters(tau = 1)
                
    def get_feature( self, im_data=None):
        return self.backbone(im_data)
    
    #Im not positive on the use of this function, so Im leaving it here for the soft actor critic update when we get round to it
    # def update_network_parameters(self, tau = None):
    #     if tau is None:
    #         tau = self.tau
        
    #     #returns the weights and biases of each Value Network along with its layers
    #     target_value_params = self.v_target.named_parameters()
    #     value_params = self.v.named_parameters()
        
    #     target_value_state_dict = dict(target_value_params)
    #     value_state_dict  = dict(value_params)
        
    #     #Going through each layer and performing a SOFT update
    #     for name in value_state_dict:
    #         value_state_dict[name]  = tau*value_state_dict[name].clone() + \
    #             (1-tau)*target_value_state_dict[name].clone()

    def get_Q(self, feature=None, history_vectory=None):
        return self.actor(feature, history_vectory)

class Actor(nn.Module):
    def __init__(self, ACTION_NUMBER, HV_NUMBER):
        super(Actor, self).__init__()
        self.layer1 = nn.Conv2d(HV_NUMBER+512, 1024, kernel_size=1, padding=0)
        self.layer2 = nn.ReLU(inplace=True)  
        self.layer3 = nn.Conv2d(1024, 1024, kernel_size=1, padding=0)
        self.layer4 = nn.ReLU(inplace=True)  
        self.layer5 = nn.Conv2d(1024, ACTION_NUMBER, kernel_size=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data,0.01)


    def forward(self, x, hv):
        x = [x,hv]        
        x = T.cat(x,1)
        
        x = self.layer1(x) 
        x = self.layer2(x) 
        
        x = self.layer3(x) 
        x = self.layer4(x) 
                
        x = self.layer5(x) 
        return x
    
    
class CriticQ(nn.Module):
    def __init__(self, ACTION_NUMBER, HV_NUMBER):
        super(CriticQ, self).__init__()
        #NOTE - Modifiying this a bit to be closer to how it is done in base libranet
        self.layer1 = nn.Conv2d(in_channels = HV_NUMBER+512, out_channels = 1024, kernel_size=1, padding=0)
        self.layer2 = nn.ReLU(inplace=True)  
        self.layer3 = nn.Conv2d(1024, 1024, kernel_size=1, padding=0)
        self.layer4 = nn.ReLU(inplace=True)
        self.layer5 = nn.Conv2d(1024, ACTION_NUMBER, kernel_size=1, padding=0)
        self.layer6 = nn.ReLU(inplace=True)
        self.layer7 = nn.Conv2d(ACTION_NUMBER, 1, kernel_size=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data,0.01)

    #NOTE - Returned to naming conventions outlined in the paper
    def forward(self, x, hv):
        x = [x, hv]
        x = T.cat(x, 1)   
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x
    
    
class CriticV(nn.Module):
    def __init__(self, HV_NUMBER):
        super(CriticV, self).__init__()
        #NOTE - HV Number is super short
        # self.conv1 = nn.Conv2d(HV_NUMBER, 1024, kernel_size=1, padding=0)
        # self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, padding=0)
        # self.conv3 = nn.Conv2d(1024, 1024, kernel_size=1, padding=0)
        # self.fc4 = nn.Linear(512+1024, 512)
        # self.fc5 = nn.Linear(512, 1)
        self.layer1 = nn.Conv2d(in_channels = HV_NUMBER, out_channels = 1024, kernel_size=1, padding=0)
        self.layer2 = nn.ReLU(inplace=True)  
        self.layer3 = nn.Conv2d(1024, 1024, kernel_size=1, padding=0)
        self.layer4 = nn.ReLU(inplace=True)
        self.layer5 = nn.Conv2d(1024, 1, kernel_size=1, padding=0)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data,0.01)

    def forward(self, hv):
        x = self.layer1(hv)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x



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
