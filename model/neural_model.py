#!/usr/bin/env python

""" This is the implementaion of the model in Python """

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import squareform, pdist


class NeuralNetwork:
    """ the neural network model """
    def __init__(self, size=1, neuron_population=100, f0=0.01):
        self.size = size    # the size of the area of the square canvas
        self.num = neuron_population    # the population of the neuron in canvas
        self.dim = 2        # dimension of the canvas
        # hash table to store data of neurons: x, y, z, radius, fired
        # x -> float
        # y -> float
        # z -> float
        # radius -> float
        self.neurons = np.recarray((self.num,),
            dtype=[('x', float), ('y', float), ('z', float), ('radius', float),
                ('f_i', float)])

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

        self.f0 = f0        # f0
        self.neurons['f_i'] = f0    # initial firing rate
        self.f_sat = 2

        self.fired = np.zeros(self.num, dtype=bool)     # see if neurons are fired
        self.tau = 10 ** -2     # decay constant for firing rates
        self._h = 10 ** -3      # time step
        self.g = 500            # correlation coefficient of mutual area (Hz)


    def update_fire_rate(self):
        """ update the firing rate of each neuron """
        # homogenious part
        self.neurons['f_i'] += (self.f0 - self.neurons['f_i']) / self.tau * self._h
        # inhomogenious part
        if np.sum(self.fired) > 0:
            self.mutual_area = self.calc_mutual_area()  # update mutual area
            self.neurons['f_i'] += np.sum(self.mutual_area[self.fired]) * self.g
        else:
            pass


    def timestep(self):
        """ evolve the system one time step """
        r_dot0 = 10 ** -6
        # decide which neurons fire at this timestep
        self.fired = np.random.random(size=self.num) < self.neurons['f_i'] * self._h
        self.update_fire_rate()     # update fire rate
        self.neurons['radius'] += r_dot0 * self._h   # homogenious increment
        self.neurons['radius'][self.fired] -= r_dot0 / self.f_sat   # inhomogenious decrement


    def calc_mutual_area(self):
        """
        find the mutual area(2D) or volume (3D) of the neurons to find
        the interaction coefficients.
        """
        def func(d, r1, r2):
            """ function for mutual area """
            if d < r1 + r2:     # have intersection
                if d < np.absolute(r1 - r2):    # one inside the other
                    if r1 > r2:  # r1 is in r2
                        return np.pi * r1 ** 2
                    else:
                        return np.pi * r2 ** 2
                else:
                    return 0.5 * np.sqrt((-d + r1 + r2) *\
                            (-d - r1 + r2) * (-d + r1 - r2) * (d + r1 + r2))
            else:
                return 0

        _mutual_area = np.zeros(shape=(self.num, self.num)) # initialize
        r1 = np.tile(self.neurons['radius'], (self.num, 1))      # matrix of radii
        r2 = r1.T           # transpose of radii as r2

        for i in range(self.num):
            for j in range(self.num):
                _mutual_area[i, j] = func(self.dist_mat[i, j],
                                          r1[i, j], r2[i, j])
        return _mutual_area


    def display(self, color):
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

        plt.plot(x_line,y_line,f'{color}-')
        plt.title(f"neuron membrane of size {self.size} and population {self.num}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def animate_system(self):
        def animate(i):
            """ animate for FuncAnimation """
            for _ in range(10):
                self.timestep()

            r = self.neurons['radius']
            x_line = x[na,:]+r[na,:]*np.sin(phi[:,na])
            y_line = y[na,:]+r[na,:]*np.cos(phi[:,na])
            ax.clear()
            ax.plot(x_line, y_line, 'b-')
            plt.title(f"step {i}, neuron population = {self.num}")

            return 0

        r = self.neurons['radius']
        x = self.neurons['x']
        y = self.neurons['y']
        phi = np.linspace(0.0,2*np.pi,100)

        na=np.newaxis

        # the first axis of these arrays varies the angle,
        # the second varies the circles
        x_line = x[na,:]+r[na,:]*np.sin(phi[:,na])
        y_line = y[na,:]+r[na,:]*np.cos(phi[:,na])

        fig, ax = plt.subplots()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.plot(x_line, y_line, 'b-')

        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)

        ani = FuncAnimation(fig, animate, interval=10, blit=False,
                save_count=500)
        plt.show()



def test():
    """ function to test the system """
    from time import time
    network = NeuralNetwork(neuron_population=100)
    network.display('b')

    start = time()
    num = 500000
    array = np.zeros(num)
    rand = np.random.randint(network.num)

    for i in range(num):
        print(f"\rstep {i}", end='')
        network.timestep()
        array[i] = np.sum(network.mutual_area[rand]) * network.tau * network.g

    print(f"runtime = {time() - start}")

    network.display('r')

    plt.plot(np.linspace(1, num, num), array)
    plt.show()


if __name__ == '__main__':
    test()
