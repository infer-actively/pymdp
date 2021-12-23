import os
import sys
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as patches
import matplotlib.animation as animation

import scipy.ndimage as ndimage

path = pathlib.Path(os.getcwd())
sys.path.append(str(path) + '/')

from pymdp.agent import Agent
from pymdp import utils, maths

from IPython.display import HTML, Image

# set up dimensionalities of the generative model and environment

grid_dims = [5, 7] # dimensions of the grid
num_grid_points = np.prod(grid_dims)

# create a look-up table mapping linear indices to (y, x) tuples
grid = np.arange(num_grid_points).reshape(grid_dims)
it = np.nditer(grid, flags=["multi_index"])

loc_list = []
while not it.finished:
    loc_list.append(it.multi_index)
    it.iternext()

cue1_location = (2, 0)
# cue2_locations = [(1, 3), (1, 5), (3, 3), (3, 5)]
cue2_locations = [(0, 2), (1, 3), (3, 3), (4, 2)]

reward_conditions = ["TOP", "BOTTOM"]
# reward_locations = [(0, 4), (4, 4)]
reward_locations = [(1, 5), (3, 5)]

num_states = [num_grid_points, len(cue2_locations), len(reward_conditions)]

cue1_names = ['Null', 'Cue 1', 'Cue 2', 'Cue 3', 'Cue 4'] # different possible cue identities at cue1 location
cue2_names = ['Null', 'reward_on_top', 'reward_on_bottom']

reward_names = ['Null', '+5', '-10']

num_obs = [num_grid_points, len(cue1_names), len(cue2_names), len(reward_names)]

# A matrix
A_m_shapes = [ [o_dim] + num_states for o_dim in num_obs]
A = utils.obj_array_zeros(A_m_shapes)

# make the location observation only depend on the location state (proprioceptive observation modality)
A[0] = np.tile(np.expand_dims(np.eye(num_grid_points), (-2, -1)), (1, 1, num_states[1], num_states[2]))

# make the cue1 observation depend on the location (being at cue1_location) and the true location of cue2
A[1][0,:,:,:] = 1.0 # default makes Null the most likely observation everywhere

for i, cue_loc2_i in enumerate(cue2_locations):
    A[1][0,loc_list.index(cue1_location),i,:] = 0.0
    A[1][i+1,loc_list.index(cue1_location),i,:] = 1.0

# make the cue2 observation depend on the location (being at the correct cue2_location) and the reward condition
A[2][0,:,:,:] = 1.0 # default makes Null the most likely observation everywhere

for i, cue_loc2_i in enumerate(cue2_locations):

    # if the cue2-location is the one you're currently at, then you get a signal about where the reward is
    A[2][0,loc_list.index(cue_loc2_i),i,:] = 0.0 
    A[2][1,loc_list.index(cue_loc2_i),i,0] = 1.0
    A[2][2,loc_list.index(cue_loc2_i),i,1] = 1.0
    
# make the reward observation depend on the location (being at reward location) and the reward condition
A[3][0,:,:,:] = 1.0 # default makes Null the most likely observation everywhere

A[3][0,loc_list.index(reward_locations[0]),:,:] = 0.0
A[3][1,loc_list.index(reward_locations[0]),:,0] = 1.0
A[3][2,loc_list.index(reward_locations[0]),:,1] = 1.0

A[3][0,loc_list.index(reward_locations[1]),:,:] = 0.0
A[3][1,loc_list.index(reward_locations[1]),:,1] = 1.0
A[3][2,loc_list.index(reward_locations[1]),:,0] = 1.0

# B matrix

num_controls = [5, 1, 1]
B_f_shapes = [ [ns, ns, num_controls[f]] for f, ns in enumerate(num_states)]

B = utils.obj_array_zeros(B_f_shapes)
actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

for action_id, action_label in enumerate(actions):

  for curr_state, grid_location in enumerate(loc_list):

    y, x = grid_location

    if action_label == "UP":
      next_y = y - 1 if y > 0 else y 
      next_x = x
    elif action_label == "DOWN":
      next_y = y + 1 if y < (grid_dims[0]-1) else y 
      next_x = x
    elif action_label == "LEFT":
      next_x = x - 1 if x > 0 else x 
      next_y = y
    elif action_label == "RIGHT":
      next_x = x + 1 if x < (grid_dims[1]-1) else x 
      next_y = y
    elif action_label == "STAY":
      next_x = x
      next_y = y

    new_location = (next_y, next_x)
    next_state = loc_list.index(new_location)
    B[0][next_state, curr_state, action_id] = 1.0

B[1][:,:,0] = np.eye(num_states[1])
B[2][:,:,0] = np.eye(num_states[2])

# C vector
C = utils.obj_array_zeros(num_obs)

C[3][1] = 2.0
C[3][2] = -4.0

# D vector
D = utils.obj_array_uniform(num_states)
D[0] = utils.onehot(loc_list.index((0,0)), num_grid_points)

class GridWorldEnv():
    
    def __init__(self,starting_loc = (4,0), cue1_loc = (2, 0), cue2 = 'Cue 1', reward_condition = 'TOP'):

        self.init_loc = starting_loc
        self.current_location = self.init_loc

        self.cue1_loc = cue1_loc
        self.cue2_name = cue2
        self.cue2_loc_names = ['Cue 1', 'Cue 2', 'Cue 3', 'Cue 4']
        self.cue2_loc = cue2_locations[self.cue2_loc_names.index(self.cue2_name)]

        self.reward_condition = reward_condition
        print(f'Starting location is {self.init_loc}, Reward condition is {self.reward_condition}, cue is located in {self.cue2_name}')
    
    def step(self,action_label):

        (Y, X) = self.current_location

        if action_label == "UP": 
          
          Y_new = Y - 1 if Y > 0 else Y
          X_new = X

        elif action_label == "DOWN": 

          Y_new = Y + 1 if Y < (grid_dims[0]-1) else Y
          X_new = X

        elif action_label == "LEFT": 
          Y_new = Y
          X_new = X - 1 if X > 0 else X

        elif action_label == "RIGHT": 
          Y_new = Y
          X_new = X +1 if X < (grid_dims[1]-1) else X

        elif action_label == "STAY":
          Y_new, X_new = Y, X 
        
        self.current_location = (Y_new, X_new) # store the new grid location

        loc_obs = self.current_location # agent always directly observes the grid location they're in 

        if self.current_location == self.cue1_loc:
          cue1_obs = self.cue2_name
        else:
          cue1_obs = 'Null'

        if self.current_location == self.cue2_loc:
          cue2_obs = cue2_names[reward_conditions.index(self.reward_condition)+1]
        else:
          cue2_obs = 'Null'
        
        if self.current_location == reward_locations[0]:
          if self.reward_condition == 'TOP':
            reward_obs = '+5'
          else:
            reward_obs = '-10'
        elif self.current_location == reward_locations[1]:
          if self.reward_condition == 'BOTTOM':
            reward_obs = '+5'
          else:
            reward_obs = '-10'
        else:
          reward_obs = 'Null'

        return loc_obs, cue1_obs, cue2_obs, reward_obs

    def reset(self):
        self.current_location = self.init_loc
        print(f'Re-initialized location to {self.init_loc}')
        loc_obs = self.current_location
        cue1_obs = 'Null'
        cue2_obs = 'Null'
        reward_obs = 'Null'

        return loc_obs, cue1_obs, cue2_obs, reward_obs
    
my_agent = Agent(A = A, B = B, C = C, D = D, policy_len = 4)

my_env = GridWorldEnv(starting_loc = (0,0), cue1_loc = (2, 0), cue2 = 'Cue 1', reward_condition = 'TOP')
# my_env = GridWorldEnv(starting_loc = (0,0), cue1_loc = (2, 0), cue2 = 'Cue 3', reward_condition = 'BOTTOM')

loc_obs, cue1_obs, cue2_obs, reward_obs = my_env.reset()
  
history_of_locs = [loc_obs]
history_of_beliefs = []
obs = [loc_list.index(loc_obs), cue1_names.index(cue1_obs), cue2_names.index(cue2_obs), reward_names.index(reward_obs)]
for t in range(12):

    qs = my_agent.infer_states(obs)
    
    history_of_beliefs.append(qs)

    my_agent.infer_policies()
    chosen_action_id = my_agent.sample_action()

    movement_id = int(chosen_action_id[0])

    choice_action = actions[movement_id]

    loc_obs, cue1_obs, cue2_obs, reward_obs = my_env.step(choice_action)

    obs = [loc_list.index(loc_obs), cue1_names.index(cue1_obs), cue2_names.index(cue2_obs), reward_names.index(reward_obs)]

    history_of_locs.append(loc_obs)

num_points_interp = 10

# Obtain evenly-spaced ratios between 0 and 1
interpolation_weights = np.linspace(0, 1, num_points_interp)

all_points = []

for ii in range(len(history_of_locs)-1):
  loc1, loc2 = np.array(history_of_locs[ii]), np.array(history_of_locs[ii+1])
  interpolated = [(1-l)*loc1 + (l)*loc2 for l in interpolation_weights] # Generate arrays for interpolation using ratios

  all_points.append(np.asarray(interpolated))

all_points = np.vstack(all_points)

fig, ax = plt.subplots(figsize=(16, 10))

X, Y = np.meshgrid(np.arange(grid_dims[1]+1), np.arange(grid_dims[0]+1))
h = ax.pcolormesh(X, Y, np.ones(grid_dims), edgecolors='k', vmin = 0, vmax = 30, linewidth=3, cmap = 'coolwarm')
ax.invert_yaxis()

reward_top = ax.add_patch(patches.Rectangle((reward_locations[0][1],reward_locations[0][0]),1.0,1.0,linewidth=10,edgecolor=[0.5, 0.5, 0.5],facecolor='none'))
reward_bottom = ax.add_patch(patches.Rectangle((reward_locations[1][1],reward_locations[1][0]),1.0,1.0,linewidth=10,edgecolor=[0.5, 0.5, 0.5],facecolor='none'))

mouse_img = plt.imread('mouse_img.png')

down_mouse_im = OffsetImage(mouse_img, zoom=0.05)

right_mouse = ndimage.rotate(mouse_img, 90, reshape=True)
right_mouse_im = OffsetImage(right_mouse, zoom=0.05)

left_mouse = ndimage.rotate(mouse_img, -90, reshape=True)
left_mouse_im = OffsetImage(left_mouse, zoom=0.05)

up_mouse = ndimage.rotate(mouse_img, 180, reshape=True)
up_mouse_im = OffsetImage(up_mouse, zoom=0.05)

ab_down_mouse = AnnotationBbox(down_mouse_im, (all_points[0,1] + 0.5, all_points[0,0] + 0.5), xycoords='data', frameon=False)
ab_right_mouse = AnnotationBbox(right_mouse_im, (all_points[0,1] + 0.5, all_points[0,0] + 0.5), xycoords='data', frameon=False)
ab_left_mouse = AnnotationBbox(left_mouse_im, (all_points[0,1] + 0.5, all_points[0,0] + 0.5), xycoords='data', frameon=False)
ab_up_mouse = AnnotationBbox(up_mouse_im, (all_points[0,1] + 0.5, all_points[0,0] + 0.5), xycoords='data', frameon=False)

on_axis_mouse_down = ax.add_artist(ab_down_mouse)
on_axis_mouse_right = ax.add_artist(ab_right_mouse)
on_axis_mouse_right.set_visible(False)
on_axis_mouse_left = ax.add_artist(ab_left_mouse)
on_axis_mouse_left.set_visible(False)
on_axis_mouse_up = ax.add_artist(ab_up_mouse)
on_axis_mouse_up.set_visible(False)

heading_names = ['down', 'right', 'left', 'up']

cheese_img = plt.imread('cheese_img.png')
cheese_img = OffsetImage(cheese_img, zoom=0.03)

shock_img = plt.imread('shock_img.png')
shock_img = OffsetImage(shock_img, zoom=0.13)

cue1_loc, cue2_loc, reward_condition = my_env.cue1_loc, my_env.cue2_loc, my_env.reward_condition
cue_grid = np.ones(grid_dims)
cue_grid[cue1_loc[0],cue1_loc[1]] = 15.0
for loc_ii in cue2_locations:
  row_coord, column_coord = loc_ii
  cue_grid[row_coord, column_coord] = 5.0
h.set_array(cue_grid.ravel())

reward_loc = reward_locations[0] if reward_condition == "TOP" else reward_locations[1]

cue1_rect = ax.add_patch(patches.Rectangle((cue1_loc[1],cue1_loc[0]),1.0,1.0,linewidth=15,edgecolor=[0.2, 0.7, 0.6],facecolor='none'))
cue2_rect = ax.add_patch(patches.Rectangle((cue2_loc[1],cue2_loc[0]),1.0,1.0,linewidth=15,edgecolor=[0.2, 0.7, 0.6],facecolor='none'))
cue2_rect.set_visible(False)

qmark_offsets = [0.4, 0.6]

def init():
    # ax.set_title("Epistemic chaining", fontsize = 36)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    ax.text(cue1_loc[1]+0.3, cue1_loc[0]+0.6, 'Cue 1', fontsize = 20)

    for loc_ii in cue2_locations:
      y, x = loc_ii
      ax.text(x+qmark_offsets[0], y+qmark_offsets[1], "?", fontsize = 55, color='k')

    fig.tight_layout()
    # return h, mouse_collection[0], mouse_collection[1], mouse_collection[2], mouse_collection[3]
    return h, on_axis_mouse_down, on_axis_mouse_right, on_axis_mouse_left, on_axis_mouse_up

def update(frame):
    ydata, xdata = frame

    if (ydata, xdata) == cue1_loc:
      cue1_rect.set_visible(False)
      cue2_rect.set_visible(True)

      cue_grid[cue2_loc[0], cue2_loc[1]] = 15.0
      h.set_array(cue_grid.ravel())
      for text_obj in ax.texts:
        text_pos = text_obj.get_position()
        if (text_pos[1] - qmark_offsets[1], text_pos[0] - qmark_offsets[0]) == cue2_loc:
          text_obj.set_position( (cue2_loc[1]+0.3, cue2_loc[0]+0.6) )
          text_obj.set_text("Cue 2")
          text_obj.set_fontsize(20) 
          text_obj.set_color('k') 
        elif text_obj.get_text() == "?":
          text_obj.set_visible(False)

    if (ydata, xdata) == cue2_loc:

      cue2_rect.set_visible(False)

      if reward_condition == "TOP":
        reward_top.set_edgecolor('g')
        reward_top.set_facecolor('g')
        reward_bottom.set_edgecolor([0.7, 0.2, 0.2])
        reward_bottom.set_facecolor([0.7, 0.2, 0.2])

        ab_cheese = AnnotationBbox(cheese_img, (reward_locations[0][1]+0.5,reward_locations[0][0]+0.5), xycoords='data', frameon=False)
        an_cheese = ax.add_artist(ab_cheese)

        an_cheese.set_zorder(2)

        ab_shock = AnnotationBbox(shock_img, (reward_locations[1][1]+0.5,reward_locations[1][0]+0.5), xycoords='data', frameon=False)
        an_shock = ax.add_artist(ab_shock)

        an_shock.set_zorder(2)

      elif reward_condition == "BOTTOM":

        ab_cheese = AnnotationBbox(cheese_img, (reward_locations[1][1]+0.5,reward_locations[1][0]+0.5), xycoords='data', frameon=False)
        an_cheese = ax.add_artist(ab_cheese)
        an_cheese.set_zorder(2)

        ab_shock = AnnotationBbox(shock_img, (reward_locations[0][1]+0.5,reward_locations[0][0]+0.5), xycoords='data', frameon=False)
        an_shock = ax.add_artist(ab_shock)
        an_shock.set_zorder(2)

        reward_bottom.set_edgecolor('g')
        reward_bottom.set_facecolor('g')
        reward_top.set_edgecolor([0.7, 0.2, 0.2])
        reward_top.set_facecolor([0.7, 0.2, 0.2])
    
    # for ii, mouse_artist in enumerate(mouse_collection):
    #   if mouse_artist.get_visible():
    #     visible_artist_idx = ii
    #     current_mouse_heading = heading_names[visible_artist_idx]

    # if mouse_collection[visible_artist_idx].xy[0] < xdata: # if the next move is to the right
    #   which_visible = mouse_collection[1]
    #   invisible_artist_idx = [0, 2, 3]
    # elif mouse_collection[visible_artist_idx].xy[0] > xdata: # if the next move is to the left
    #   which_visible = mouse_collection[2]
    #   invisible_artist_idx = [0, 1, 3]
    # elif mouse_collection[visible_artist_idx].xy[1] < ydata: # if the current move is to move up
    #   which_visible = mouse_collection[3]
    #   invisible_artist_idx = [0, 1, 2]
    # elif mouse_collection[visible_artist_idx].xy[1] > ydata: # if the current move is to move down
    #   which_visible = mouse_collection[0]
    #   invisible_artist_idx = [1, 2, 3]
    # else:
    #   which_visible = mouse_collection[visible_artist_idx]
    #   invisible_artist_idx = list(set([0, 1, 2, 3]) - set([visible_artist_idx]))
    
    # which_visible.set_visible(True)
    # which_visible.xy = (xdata+0.5, ydata + 0.5)
    # which_visible.xybox = (xdata+0.5, ydata + 0.5)
    # which_visible.set_zorder(3)

    mouse_collection = [on_axis_mouse_down, on_axis_mouse_right, on_axis_mouse_left, on_axis_mouse_up]

    for ii, mouse_artist in enumerate(mouse_collection):
        if mouse_artist.get_visible():
          visible_artist_idx = ii
    
    if mouse_collection[visible_artist_idx].xy[0] < (xdata + 0.5): # if the next move is to the right
      on_axis_mouse_right.set_visible(True)
      on_axis_mouse_right.xy = (xdata+0.5, ydata + 0.5)
      on_axis_mouse_right.xybox = (xdata+0.5, ydata + 0.5)
      on_axis_mouse_right.set_zorder(3)

      on_axis_mouse_left.set_visible(False)
      on_axis_mouse_down.set_visible(False)
      on_axis_mouse_up.set_visible(False)
    elif mouse_collection[visible_artist_idx].xy[0] > (xdata + 0.5): # if the next move is to the left
      on_axis_mouse_left.set_visible(True)
      on_axis_mouse_left.xy = (xdata+0.5, ydata + 0.5)
      on_axis_mouse_left.xybox = (xdata+0.5, ydata + 0.5)
      on_axis_mouse_left.set_zorder(3)

      on_axis_mouse_right.set_visible(False)
      on_axis_mouse_down.set_visible(False)
      on_axis_mouse_up.set_visible(False)
    elif mouse_collection[visible_artist_idx].xy[1] > (ydata + 0.5): # if the current move is to move up
      on_axis_mouse_up.set_visible(True)
      on_axis_mouse_up.xy = (xdata+0.5, ydata + 0.5)
      on_axis_mouse_up.xybox = (xdata+0.5, ydata + 0.5)
      on_axis_mouse_up.set_zorder(3)

      on_axis_mouse_right.set_visible(False)
      on_axis_mouse_down.set_visible(False)
      on_axis_mouse_left.set_visible(False)
    elif mouse_collection[visible_artist_idx].xy[1] < (ydata + 0.5): # if the current move is to move down
      on_axis_mouse_down.set_visible(True)
      on_axis_mouse_down.xy = (xdata+0.5, ydata + 0.5)
      on_axis_mouse_down.xybox = (xdata+0.5, ydata + 0.5)
      on_axis_mouse_down.set_zorder(3)

      on_axis_mouse_right.set_visible(False)
      on_axis_mouse_up.set_visible(False)
      on_axis_mouse_left.set_visible(False)
    else:
      if visible_artist_idx == 0:
        # on_axis_mouse_down.set_visible(True)
        on_axis_mouse_down.xy = (xdata+0.5, ydata + 0.5)
        on_axis_mouse_down.xybox = (xdata+0.5, ydata + 0.5)
        on_axis_mouse_down.set_zorder(3)
      elif visible_artist_idx == 1:
        # on_axis_mouse_right.set_visible(True)
        on_axis_mouse_right.xy = (xdata+0.5, ydata + 0.5)
        on_axis_mouse_right.xybox = (xdata+0.5, ydata + 0.5)
        on_axis_mouse_right.set_zorder(3)
      elif visible_artist_idx == 2:
        # on_axis_mouse_left.set_visible(True)
        on_axis_mouse_left.xy = (xdata+0.5, ydata + 0.5)
        on_axis_mouse_left.xybox = (xdata+0.5, ydata + 0.5)
        on_axis_mouse_left.set_zorder(3)
      elif visible_artist_idx == 3:
        # on_axis_mouse_up.set_visible(True)
        on_axis_mouse_up.xy = (xdata+0.5, ydata + 0.5)
        on_axis_mouse_up.xybox = (xdata+0.5, ydata + 0.5)
        on_axis_mouse_up.set_zorder(3)

    # for invis_ii in invisible_artist_idx:
    #   mouse_collection[invis_ii].set_visible(False)

    # on_axis_mouse.xy = (xdata+0.5, ydata + 0.5)
    # on_axis_mouse.xybox = (xdata+0.5, ydata + 0.5)
      
    # on_axis_mouse.set_zorder(3)

    # return h, mouse_collection[0], mouse_collection[1], mouse_collection[2], mouse_collection[3]
    # return h, which_visible
    return h, on_axis_mouse_down, on_axis_mouse_right, on_axis_mouse_left, on_axis_mouse_up
# anim = FuncAnimation(fig, update, frames=np.array(history_of_locs), init_func=init, blit=True)
anim = FuncAnimation(fig, update, frames=all_points, init_func=init, blit=True)

anim.save('.github/chained_cue_navigation_v1.gif', writer = 'pillow', fps = 15)

# anim.save('.github/chained_cue_navigation_v2.gif', writer = 'pillow', fps = 15)
