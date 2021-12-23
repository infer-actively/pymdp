
# %%
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib as mpl

import scipy.ndimage as ndimage





# y = np.linspace(-10, 10, num=1000)
# x = np.linspace(-10, 10, num=1000)
# X, Y = np.meshgrid(x, y)
# C = np.ones((1000, 1000)) * float('nan')

# # intantiate empty plot (values = nan)
# pcmesh = plt.pcolormesh(X, Y, C, vmin=-100, vmax=100, shading='flat')

# %%
grid_dims = [5, 7] # dimensions of the grid

fig, ax = plt.subplots(figsize=(16, 10))
# h = sns.heatmap(0.1 + np.zeros(grid_dims), cmap='viridis', vmax=1., vmin=0.0, linewidth = 20.0, cbar=False, ax=ax)
# h = ax.imshow(0.4 + np.zeros(grid_dims), interpolation='none', cmap='viridis', vmax=1., vmin=0.0)
# ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
# h = ax.pcolormesh(np.zeros(grid_dims), edgecolors='k', linewidth=3)
X, Y = np.meshgrid(np.arange(grid_dims[1]+1), np.arange(grid_dims[0]+1))
h = ax.pcolormesh(X, Y, np.ones(grid_dims), edgecolors='k', vmin = -5, vmax = 20, linewidth=3)
ax.add_patch(patches.Rectangle((4,3),1.0,1.0,linewidth=10,edgecolor=[0.5, 0.5, 0.5],facecolor='none'))
ax.add_patch(patches.Rectangle((4,1),1.0,1.0,linewidth=10,edgecolor=[0.5, 0.5, 0.5],facecolor='none'))
ax.invert_yaxis()
# cue_grid = np.ones(grid_dims)
# cue_grid[2,0] = 10.0
# h.set_array(cue_grid.ravel())
# plt.draw()

# plt.text(0.3, 2.6, 'Cue 1', fontsize = 20)


all_points = np.zeros((2,2))

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

plt.draw()
# %%

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

y, x = np.meshgrid(np.linspace(-10, 10,100), np.linspace(-10, 10,100))

z = np.sin(x)*np.sin(x)+np.sin(y)*np.sin(y)

v = np.linspace(-10, 10,100)
t = np.sin(v)*np.sin(v)
tt = np.cos(v)*np.cos(v)
###########

fig = plt.figure(figsize=(16, 8),facecolor='white')
gs = gridspec.GridSpec(5, 2)
ax1 = plt.subplot(gs[0,0])

line, = ax1.plot([],[],'b-.',linewidth=2)
ax1.set_xlim(-10,10)
ax1.set_ylim(0,1)
ax1.set_xlabel('time')
ax1.set_ylabel('amplitude')
ax1.set_title('Oscillationsssss')
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

#############################
ax2 = plt.subplot(gs[1:3,0])
quad1 = ax2.pcolormesh(x,y,z,shading='gouraud')
ax2.set_xlabel('time')
ax2.set_ylabel('amplitude')
cb2 = fig.colorbar(quad1,ax=ax2)

#########################
ax3 = plt.subplot(gs[3:,0])
quad2 = ax3.pcolormesh(x, y, z,shading='gouraud')
ax3.set_xlabel('time')
ax3.set_ylabel('amplitude')
cb3 = fig.colorbar(quad2,ax=ax3)

############################
ax4 = plt.subplot(gs[:,1])
line2, = ax4.plot(v,tt,'b',linewidth=2)
ax4.set_xlim(-10,10)
ax4.set_ylim(0,1)

def init():
    line.set_data([],[])
    line2.set_data([],[])
    quad1.set_array([])
    return line,line2,quad1

def animate(iter):
    t = np.sin(2*v-iter/(2*np.pi))*np.sin(2*v-iter/(2*np.pi))
    tt = np.cos(2*v-iter/(2*np.pi))*np.cos(2*v-iter/(2*np.pi))
    z = np.sin(x-iter/(2*np.pi))*np.sin(x-iter/(2*np.pi))+np.sin(y)*np.sin(y)
    line.set_data(v,t)
    quad1.set_array(z.ravel())
    line2.set_data(v,tt)
    return line,line2,quad1

gs.tight_layout(fig)

anim = animation.FuncAnimation(fig,animate,frames=100,interval=50,blit=False,repeat=True)
plt.show()
# %%
