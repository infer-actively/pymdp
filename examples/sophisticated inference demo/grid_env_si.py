#Importing needed modules
import numpy as np
import matplotlib.pyplot as plt
import os

class grid_environment():

    def __init__(self, path = "grid_si.txt", epi_length_limit = 10000):
        
        mdppath = 'mdp_si.txt'
        
        #storing the file as strings line by line
        mdpdata=[]
        
        #Saving arm true means to the array-band (indices indicates arms)
        mdp = open(str(mdppath), "r")
        for x in mdp:
            mdpdata.append(x)
        mdp.close()
        #Closing mdp file.
        
        #Determining the nature of mdp
        mdptype1=[]
        for word in mdpdata[-2].split():
            try:
                mdptype1.append(str(word))
            except (ValueError, IndexError):
                pass
        mdptype=mdptype1[1]
        
        #Discount factor
        gamma1=[]
        for word in mdpdata[-1].split():
            try:
                gamma1.append(float(word))
            except (ValueError, IndexError):
                pass
        
        self.gamma=float(gamma1[0])
        
        #Number of states
        states=[]
        for word in mdpdata[0].split():
            try:
                states.append(int(word))
            except (ValueError, IndexError):
                pass
        self.numS=int(states[0])
        #Number of actions
        actions=[]
        for word in mdpdata[1].split():
            try:
                actions.append(int(word))
            except (ValueError, IndexError):
                pass
        self.numA=int(actions[0])
        #Start state
        start=[]
        for word in mdpdata[2].split():
            try:
                start.append(int(word))
            except (ValueError, IndexError):
                pass
            
        self.startS=int(start[0])
        #Terminal states for episodic mdps
        if(mdptype=='episodic'):
            terminal=[]
            for word in mdpdata[3].split():
                try:
                    terminal.append(int(word))
                except (ValueError, IndexError):
                    pass
            self.no_of_termS = len(terminal)
            self.terS = terminal
        
        #T-matrix dimensions numS*numS*numA
        #R-matrix dimensions numS*numA*numS
        
        self.T = np.zeros((self.numS,self.numA,self.numS))
        self.R = np.zeros((self.numS,self.numA,self.numS))
        if(mdptype=='episodic'):
            for i in range(len(self.terS)):
                self.T[self.terS[i],:,self.terS[i]]=1
        
        for i in range(4,len(mdpdata)-2):
            trans=[]
            for word in mdpdata[i].split():
                try:
                    trans.append(float(word))
                except (ValueError, IndexError):
                    pass
            trans
            s1=int(trans[0])
            ac=int(trans[1])
            s2=int(trans[2])
            r=float(trans[3])
            p=float(trans[4])
            self.T[s1,ac,s2]=p
            self.R[s1,ac,s2]=r
        
        #useful variables
        #numS,numA,startS,terS,mdptype,gamma,T,R
        self.path = path
        self.current_state = self.startS
        self.end_state = self.terS
        self.info = None
        self.tau = 0
        self.tau_limit = epi_length_limit
        self.termination = False
        self.truncation = False
        
        #Encoder
        #Importing needed modules

        # #Reading arguments for program from shell call
        gridpath = path

        #storing the file as strings line by line
        griddata=[]

        #Saving arm true means to the array-band (indices indicates arms)
        grid = open(str(gridpath), "r")
        for x in grid:
            griddata.append(x)
        grid.close()
        #Closing mdp file

        #Grid Representation
        grid=[]
        for i in range(len(griddata)):    
            row=[]
            for word in griddata[i].split():
                try:
                    row.append(int(word))
                except (ValueError, IndexError):
                    pass
            grid.append(row)

        n=len(grid)

        self.allstates=[]
        nonterm=[]
        self.validstates=[]
        endstates=[]
        startstate=[]

        for i in range(n):
            for j in range(n):
                
                self.allstates.append((i,j))
                if(grid[i][j]!=1):
                    self.validstates.append((i,j))
                if(grid[i][j]!=1 and grid[i][j]!=3):
                    nonterm.append((i,j))
                if(grid[i][j]==2):
                    startstate.append((i,j))
                if(grid[i][j]==3):
                    endstates.append((i,j))

    #to get the state number corresponding to the coordinate of a valis state
    def ctostates(self,x,y):
        s=0
        for i in range(self.numS):
            if(self.validstates[i][0]==x and self.validstates[i][1]==y):
                break
            s=s+1
        return(s)
    
    def ctoallstates(self,x,y):
        s=0
        for i in range(self.numS):
            if(self.allstates[i][0]==x and self.allstates[i][1]==y):
                break
            s=s+1
        return(s)

    def statestoc(self,s):
        return [self.validstates[s][0], self.validstates[s][1]]
    
    def allstatestoc(self,s):
        return [self.allstates[s][0], self.allstates[s][1]]
        
    def render(self, animation_save = False, N = 0, tau = 0):
        plt.figure(figsize=(3.5, 3.5))
        grid = np.loadtxt(self.path, dtype = int)
        [x,y] = self.statestoc(self.current_state)
        [sx, sy] = self.statestoc(self.startS)
        grid[sx,sy] = 0
        grid[x][y] = 4
        plt.imshow(grid, cmap=plt.cm.CMRmap, interpolation='nearest') #
        plt.xticks([]), plt.yticks([])
        if(animation_save == True):
            # Figure figure when rendered in environment
            plt.savefig(f'./img_N_{N}/img_{tau}.png')
        else:
            plt.show()
        
    def render_c_matrix(self, c):
        plt.figure(figsize=(3.5, 3.5))
        if c is not None:
            grid = c
        plt.imshow(grid, cmap=plt.cm.CMRmap, interpolation='nearest') #
        plt.xticks([]), plt.yticks([])
        plt.show()
        
    def reset(self, seed = 10):
        self.current_state = self.startS
        self.termination = False
        self.truncation = False
        self.tau = 0
        return self.current_state, self.info
        
    def step(self, action):
        self.tau += 1
        n_s = np.argmax(self.T[self.current_state, action, :])
        reward = self.R[self.current_state, action, n_s]
        if(reward == 10):
            self.termination = True
        if(self.tau > self.tau_limit):
            self.truncation = True
        self.current_state = n_s
        return n_s, reward, self.termination, self.truncation, self.info
        
    def get_trueB(self):
        true_B = np.zeros((self.numS, self.numS, self.numA))
        for i in range(self.numS):
            for j in range(self.numA):
                true_B[:,i,j] = self.T[i,j,:]
        return true_B
