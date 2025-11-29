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

if torch.cuda.is_available() : 
    device = torch.device("cuda")
elif torch.backends.mps.is_available() :
    device = torch.device("mps")
else : 
    device = torch.device("cpu")


EPISODES = 1500

class DQN_agent :
    def __init__(self):

        #Memory
        self.batch_size = 32 #Batch size possible à 512
        self.train_start =  5000
        self.memory=deque(maxlen = 100000)
        self.update_target_rate = 10000 
        self.no_op_steps = 50
        
        #Initialize env and agent
        self.action_size=5
        self.state_size= (96, 96, 4)

        #Initilize super parameter
        self.learning_rate = 0.00025 #Attention : à changer si modification du batch size
        self.epsilon = 1.
        self.espilon_start, self.epsilon_end = 1.0 , 0.1
        self.exploration_step = 1000000. 
        self.epsilon_decay=(self.espilon_start-self.epsilon_end) / self.exploration_step
        self.discount_factor = 0.99
        
        

        #Suivi performance
        self.avg_q_max, self.avg_loss = 0, 0

        #Definition du nn
        self.model=DQN(self.action_size, self.state_size).to(device)
        #self.model.load_state_dict(torch.load('weights_save/model3_weights.pth', weights_only=True, map_location=device))
        self.target_model = DQN(self.action_size, self.state_size).to(device)
        self.update_target_model()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss().to(device)

        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def get_action(self, history):
        """
        if np.random.rand() <= self.epsilon :
            return random.randrange(self.action_size)
        else : 
            q_value=self.model(history.to(device)) 
            return torch.argmax(q_value, dim=1).item()  # Retourne l'action avec la plus grande Q-value, dim=1 recherche le maximum parmi les actions pour chaque observation [observations,actions]
        """
        q_value=self.model(history.to(device)) 
        return torch.argmax(q_value, dim=1).item()
    
    def append_sample(self, history, action, reward, next_history, dead) :
        history_cpu = history.cpu()
        next_history_cpu = next_history.cpu()
        self.memory.append((history_cpu, action, reward, next_history_cpu, dead)) #Stocker en tuple (même groupe) dans la classe, history est de taille (1, 4, 96, 96)

    def train_model(self) :

        """
        if self.epsilon> self.epsilon_end :
            self.epsilon -= self.epsilon_decay
        """
        
        batch = random.sample(self.memory, self.batch_size) #random.sample(population, k-echantillons) 

        #Pas de np.array() avec pytorch 
        state = torch.stack([sample[0][0] for sample in batch]).to(device) #1er [0] accède à "history" dans le tuple, 2e[0] prend la première image sur les 4
        actions = torch.tensor([sample[1] for sample in batch], dtype=torch.long).to(device)
        rewards = torch.tensor([sample[2] for sample in batch], dtype=torch.float32).to(device)
        next_state = torch.stack([sample[3][0] for sample in batch]).to(device)
        dead = torch.tensor([sample[4] for sample in batch], dtype=torch.long).to(device)

        target_predicts = self.target_model(next_state)
        max_q = torch.amax(target_predicts, dim=1)
        # Target-update based on Bellman equation.
        targets = rewards + (1 - dead) * self.discount_factor * max_q #Dead=False=0, Dead=True=1 
        
        '''PYTORCH n'a pas de .fit() tout est fait manuellement'''
        #Forward pass : On calcule les prédictions du modèle avec self.model(inputs).
         
        predicted_q = self.model(state)
        predicted_q_gather = predicted_q.gather(1, actions.view(-1, 1))#gather extrait les valeurs de chaque indices actions prédites à partir de state
        '''
        self.model(history) :

        Le modèle (self.model) prend les données d'entrée history(=state) et effectue une prédiction.
        La sortie est un tenseur de forme (batch_size, num_actions), où chaque élément représente la Q-valeur estimée pour chaque action possible.
        
        actions.view(-1, 1) :

        actions est une liste des actions choisies pour chaque élément du batch. Par exemple, [2, 3, 0] signifie que l'action choisie dans le premier échantillon est 2, dans le deuxième échantillon est 3, etc.
        .view(-1, 1) change la forme du tenseur actions de (batch_size,) à (batch_size, 1), pour être compatible avec gather.
                
        gather(1, actions.view(-1, 1)) :

        La méthode .gather(dim, index) extrait les valeurs le long de la dimension spécifiée (dim=1 ici).
        dim=1 signifie qu'on récupère les valeurs le long des colonnes, et index fournit les indices des valeurs à extraire.
        Pour chaque ligne du batch, .gather() extrait la Q-valeur correspondant à l'action choisie.

        '''
        #Calcul de la perte : On utilise une fonction de perte (ici MSELoss).
        loss = self.criterion(predicted_q_gather.squeeze(), targets) #.squeeze() supprime les dimensions de taille 1
        
        #Backward pass : On calcule les gradients avec loss.backward()
        self.optimizer.zero_grad()
        loss.backward()

        #Optimisation : On met à jour les poids du modèle avec self.optimizer.step().
        self.optimizer.step()

        self.avg_loss += loss.item()
        self.avg_q_max += torch.amax(predicted_q).item()
        

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
    
    return state , img
 

if __name__=="__main__":

    env = gym.make("CarRacing-v3", continuous=False)
    env.reset()
    agent = DQN_agent()
    Episode, Score, Tuile =[], [], []
    global_step=0
    #recorder = VideoRecorder(filename='save_video/video_racing_car.mp4', frame_size=(84, 84), fps=30)
    for i in range(1, EPISODES + 1):

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
        state, img = pre_processing(observe)
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
            
            state, img = pre_processing(observe)
            
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

            #Update the memory
            agent.append_sample(history, action, reward, next_history, terminated)

            #Train le model
            if len(agent.memory)>= agent.train_start : 
                agent.train_model()

            if global_step % agent.update_target_rate == 0 :
                agent.update_target_model()
            
            history = next_history
            
            if done :

                print(f"episode {i} , score {score} , tuile_visited = {tuile_visited_nb}, global_step {global_step} , average_q {agent.avg_q_max/float(step)} , average_loss {agent.avg_loss/float(step)} ")
                
                Score.append(score)
                Episode.append(i)
                Tuile.append(tuile_visited_nb)

                # Première figure : Score
                pylab.figure(1)  # Crée ou sélectionne la figure 1
                pylab.plot(Episode, Score, 'b')  # Trace le graphique pour les scores
                pylab.title("Score par épisode")  # Ajoute un titre (optionnel)
                pylab.xlabel("Épisode")  # Ajoute un label pour l'axe des X
                pylab.ylabel("Score")  # Ajoute un label pour l'axe des Y
                pylab.savefig("save_graph/racing_car4_score.png")  # Sauvegarde la figure 1

                # Deuxième figure : Tuile
                pylab.figure(2)  # Crée ou sélectionne la figure 2
                pylab.plot(Episode, Tuile, 'r')  # Trace le graphique pour les tuiles
                pylab.title("Tuiles visitées par épisode")  # Ajoute un titre (optionnel)
                pylab.xlabel("Épisode")  # Ajoute un label pour l'axe des X
                pylab.ylabel("Tuiles visitées")  # Ajoute un label pour l'axe des Y
                pylab.savefig("save_graph/racing_car4_tuile.png")  # Sauvegarde la figure 2


        if i % 25 == 0 :
            torch.save(agent.model.state_dict(), "weights_save/model4_weights.pth")
