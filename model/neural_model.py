#!/usr/bin/env python

""" This is the implementaion of the model in Python """

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import squareform, pdist


class NeuralNetwork:
    """ the neural network model """
    def __init__(self, size=1, neuron_population=100):
        self.size = size    # the size of the area of the square canvas
        self.num = neuron_population    # the population of the neuron in canvas
        self.dim = 2        # dimension of the canvas
        # hash table to store data of neurons: x, y, z, radius, fired
        # x -> float
        # y -> float
        # z -> float
        # radius -> float
        self.neurons = np.recarray((self.num,),
            dtype=[('x', float), ('y', float), ('z', float), ('radius', float)])

        self.neurons['x'] = np.random.random(size=self.num) * self.size      # random initial position for the neuron in the canvas
        self.neurons['y'] = np.random.random(size=self.num) * self.size      # random initial position for the neuron in the canvas

        if self.dim == 3:
            self.neurons['z'] = np.random.random(size=self.num) * self.size      # random initial position for the neuron in the canvas
        else:
            self.neurons['z'] = np.zeros(shape=self.num)    # all zero z-axis

        self.neurons['radius'] = np.random.random(size=self.num) * 0.05 * self.size    # list of neuron radii as disks (2D) or spheres (3D)
        pos = np.vstack((np.vstack((self.neurons['x'], self.neurons['y'])), self.neurons['z']))
        self.dist_mat = squareform(pdist(pos.T))     # distance matrix of the neurons
        self.mutual_area = self.calc_mutual_area()      # calculate mutual area of disks(2D) or volume of spheres (3D)

    def timestep(self):
        """ evolve the system one time step """
        pass


    def calc_mutual_area(self):
        """
        find the mutual area(2D) or volume (3D) of the neurons to find
        the interaction coefficients.
        """
        mask = self.dist_mat < self.neurons['radius'] + self.neurons['radius'].T    # find which neurons have intersections
        _mutual_area = np.zeros(shape=(self.num, self.num)) # initialize
        r1 = np.tile(self.neurons['radius'], (self.num, 1))      # matrix of radii
        r2 = r1.T           # transpose of radii as r2
        for i in range(self.size):
            for j in range(self.size):
                if mask[i, j] == True and i != j:   # if has intersection and not self
                    _mutual_area[i, j] = 0.5 * \
                            np.sqrt((-self.dist_mat[i, j] + r1[i, j] + r2[i, j]) *\
                            (self.dist_mat[i, j] + r1[i, j] - r2[i, j]) *\
                            (self.dist_mat[i, j] - r1[i, j] + r2[i, j]) *\
                            (self.dist_mat[i, j] + r1[i, j] + r2[i, j]))
        print("mutual area")
        print(_mutual_area)
        return _mutual_area


    def display(self):
        """ plot the neurons on the canvas as disks """
        r = self.neurons['radius']
        x = self.neurons['x']
        y = self.neurons['y']
        phi = np.linspace(0.0,2*np.pi,100)

        na=np.newaxis

        # the first axis of these arrays varies the angle,
        # the second varies the circles
        x_line = x[na,:]+r[na,:]*np.sin(phi[:,na])
        y_line = y[na,:]+r[na,:]*np.cos(phi[:,na])

        plt.plot(x_line,y_line,'b-')
        plt.title(f"neuron membrane of size {self.size} and population {self.num}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


def test():
    """ function to test the system """
    network = NeuralNetwork()
    network.display()


if __name__ == '__main__':
    test()
