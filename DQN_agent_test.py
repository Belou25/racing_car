import gymnasium as gym
import cv2
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import warnings
import random
import pylab
warnings.filterwarnings("ignore", category=DeprecationWarning)

if torch.cuda.is_available() : 
    device = torch.device("cuda")
elif torch.backends.mps.is_available() :
    device = torch.device("mps")
else : 
    device = torch.device("cpu")


EPISODES = 1500


class DQN_agent :
    def __init__(self):

        #Initialize env and agent
        self.action_size=5
        self.state_size= (96, 96, 4)

        #Initilize super parameter
        self.learning_rate = 0.00025 
        self.epsilon = 1.
        self.espilon_start, self.epsilon_end = 1.0 , 0.1
        self.exploration_step = 1000000. 
        self.epsilon_decay=(self.espilon_start-self.epsilon_end) / self.exploration_step
        self.discount_factor = 0.99
        
        #Memory
        self.batch_size = 32 
        self.train_start = 5000 
        self.memory=deque(maxlen = 100000) 
        self.update_target_rate = 10000 
        self.no_op_steps = 50

        #Suivi performance
        self.avg_q_max, self.avg_loss = 0, 0

        #Definition du nn
        self.model=DQN(self.action_size, self.state_size).to(device)
        self.model.load_state_dict(torch.load('weights_save/model3_weights.pth', weights_only=True, map_location=device))
        self.model.eval()
        
    def get_action(self, history):
        self.model.eval()
        q_value=self.model(history.to(device)) 
        self.model.train()
        return torch.argmax(q_value, dim=1).item()  # Retourne l'action avec la plus grande Q-value, dim=1 recherche le maximum parmi les actions pour chaque observation [observations,actions]
        

class DQN(nn.Module):
    def __init__(self, action_size, state_size):
        super(DQN, self).__init__()

        #Set Up des convolutions
        self.conv1 = nn.Conv2d(in_channels=state_size[2], out_channels=16, kernel_size=8, stride=4) # [N, 4, 84, 84] -> [N, 16, 20, 20]
        #In_channels nombre de frame, out_channels : nombre de frame en sortie, Kernel_size : taille du filtre appliqué, stride : déplacement de pixel après chaque appliquation de filtre
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # [N, 16, 20, 20] -> [N, 32, 9, 9]
        # output_size = (input_size - kernel_size) / (stride) + 1 = 23
        #Set up des connections aux layers
        self.fc1 = nn.Linear(9 * 9 * 32, 256)
        self.fc2 = nn.Linear(256, action_size)
        self.to(device)

    def forward(self, x):
        #Appliquation du NN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, 9 * 9 * 32)) #nn.flatten()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def pre_processing(observe):
    img = cv2.resize(observe, dsize = (84, 84))
    state= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
     #Type d'interpolation en faisant la moyenne des pixels autour
    
    return state

if __name__=="__main__":

    env = gym.make("CarRacing-v3", continuous=False, render_mode="human")
    env.reset()
    env.render()
    agent = DQN_agent()
    Episode, Score, Tuile =[], [], []
    global_step=0

    for i in range(1):

        observe = env.reset()
        
        #Initialisation
        done=False
        dead=False
        
        action = 3 #freiner
        step = 0
        score = 0
        agent.avg_loss, agent.avg_q_max = 0, 0 
        #Attendre que des frames soient généré avant de pre_process
        for _ in range(np.random.randint(1, agent.no_op_steps)):
            observe, _, _, _, _ = env.step(action)
        
        # Preprocesses the frames and stacks 4 states to use as input.
        state = pre_processing(observe)
        history = np.stack((state,state,state,state), axis=2) #Une colonnes = une image
        history = np.reshape([history], (1, 84, 84, 4))
        history = torch.tensor(history, dtype=torch.float32).to(device) #On met history dans une liste pour le préparer a entrer dans le NN
        history = history.permute(0,3,2,1)
        
        
        while not done:
            
            #add step
            global_step+=1
            step+=1

            #action choice
            action = agent.get_action(history)

            #Get step info
            observe, reward, terminated, truncated, info = env.step(action) 
            
            state = pre_processing(observe)
            
            #Next_state
            next_state = torch.tensor(state, dtype=torch.float32).to(device)
            next_state = torch.reshape(next_state, (1, 84, 84, 1))
            next_state = next_state.permute(0, 3, 2, 1)

            #Update History
            next_history = torch.cat((next_state, history[:, :3, :, :]), dim=1)
            
            #Test if loop done
            done = truncated or terminated

            #Tuile Visited
            tuile_visited_nb=env.unwrapped.tile_visited_count

            score+=reward 
            
            history = next_history
            
         
                    
    print(f"episode {i} , score {score} , tuile_visited = {tuile_visited_nb}, global_step {global_step} , average_q {agent.avg_q_max/float(step)} , average_loss {agent.avg_loss/float(step)} ")
