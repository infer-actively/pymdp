#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scene Construction / Visual Search Environment

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

from pymdp.envs import Env
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


scenes = {}

scenes['flee'] = np.array([["bird", "cat"], 
                    ["null", "null"]])
scenes['feed'] = np.array([["bird", "seed"], 
                    ["null", "null"]])
scenes['wait'] = np.array([["bird", "null"], 
                    ["null", "seed"]])

graphics_folder = "MDP_search_graphics"


fig, ax = plt.subplots(figsize=(16, 10))

img = plt.imread(os.path.join(graphics_folder,'bird.png'))
im = OffsetImage(img, zoom=0.35)

ab = AnnotationBbox(im, (0.75, 0.75), xycoords='data', frameon=False)
an = ax.add_artist(ab)

class SceneConstruction(Env):

    def __init__(self, scenes=None):
        if scenes is None:
            self.scenes = self._construct_default_scenes()
        else:
            self.scenes = scenes

    def display_scene_array(self):

        img_arr = scenes[self.scene]

        for row_i, img_names in enumerate(img_arr):
            for col_i, img_name in enumerate(img_names):
                print(f'Row: {row_i}, Column: {col_i}, image: {img_name}')

    def _construct_default_scenes(self):
        scenes = {}

        scenes['flee'] = [["bird", "cat"], 
                          ["null", "null"]]
        scenes['feed'] = [["bird", "seed"], 
                          ["null", "null"]]
        scenes['wait'] = [["bird", "null"], 
                          ["null", "seed"]]

        return scenes
