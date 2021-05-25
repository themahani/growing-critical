#!/usr/bin/env python

""" This is the implementaion of the model in Python """

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


class Neural_Network(object):
    """ the neural network model """
    def __init__(self, size, neuron_population):
        self.size = size    # the size of the area of the square canvas
        self.num = neuron_population    # the population of the neuron in canvas
        self.dim = 2        # dimension of the canvas
        self.pos = np.random.random(size=(self.num, self.dim)) * self.size      # random initial position for the neuron in the canvas

    
    def timestep(self):
        """ evolve the system one time step """
        pass


    def calc_mutual_area(self):
        """
        find the mutual area(2D) or volume (3D) of the neurons to find 
        the interaction coefficients.
        """
        pass
