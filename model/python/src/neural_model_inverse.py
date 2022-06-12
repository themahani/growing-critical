#!/usr/bin/env python

""" This is the implementaion of the model in Python """

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import squareform, pdist


class NeuralNetwork:
    """Form the neural network model."""
    def __init__(self, size=1, neuron_population: int=100, tau: float=0.01,
            f0: float=0.01, f_sat: float=2, g: float=500, k: float=10 ** -6,
            timestep: float=0.001) -> None:
        self.size = size    # the size of the area of the square canvas
        self.num = neuron_population    # the population of the neuron in canvas
        self.dim = 2        # dimension of the canvas

        """
        hash table to store data of neurons: x, y, z, radius, fired
        x -> float
        y -> float
        z -> float
        radius -> float
        f_i -> float
        """
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

        self.f0 = f0        # f0 (Hz)
        self.neurons['f_i'] = f0    # initial firing rate
        self.f_sat = f_sat      # f_sat (Hz)

        self.fired = np.zeros(self.num, dtype=bool)     # see if neurons are fired
        self.tau = tau     # decay constant for firing rates
        self._h = timestep      # time step
        self.g = g            # correlation coefficient of mutual area (Hz)
        self.k = k
        self.f_sat = f_sat


        self.time_ax = np.arange(0, 500 ,timestep)  # initialize the time axis
        self.f_list = np.arange(0.01, 10, .1)   # list of various initial f
        cpdf_list = []  # initialize cpdf
        for f0 in self.f_list:
            isi = NeuralNetwork.pdf(self.time_ax, tau=tau, f_0=self.f0, f0=f0)
            cpdf_list.append(np.cumsum(isi) * self._h)
        self.cpdf = np.array(cpdf_list)


    @staticmethod
    def pdf(t: np.ndarray, tau:float = 0.01, f_0:float = 0.01, f0:float = 1.) -> np.ndarray:
        """Return the Probability Distribution Fucntion of spike time intervals

        ...
        Parameters
        ----------
        t
            time in seconds
        tau : float, default=0.01
            time constant in seconds
        f_0 : float, default=0.01
            base fire rate of neurons
        f0
            fire rate of neurons at the current time

        """
        return (f_0 + (f0 - f_0) * np.exp(-t / tau)) \
            * np.exp(-f_0 * t - tau * (f0 - f_0) * (1 - np.exp(-t / tau)))

    @staticmethod
    def nearest_value(arr: np.ndarray, value: float):
        """find the index of the nearest value in array to `value`
        Args:
            arr: np.ndarray
                array to find the index in (assuming it is sorted)
            value: float
                value to check the nearest values for in `arr`

        Returns:
            ind:
                array of indices for the nearest values in `arr`
        """
        ind = np.searchsorted(arr, value, side='left')
        if ind == len(arr):
            return

        mask = np.where(np.abs(arr[ind] - value) > np.abs(arr[ind-1] - value))[0]
        if mask:
            return ind-1
        else:
            return ind


    def _find_next_spike(self):
        """Find the next spike time and the neuron that fires

        Returns
        -------
        ind: int
            index of the neuron that fires the next spike
        """
        max_time = self.time_ax.shape() # initialize as the latest time possible
        rand = np.random.uniform(0, 1, self.num)    # generate random for time
        for i in range(self.num):   # do this for each neuron
            f_ind = NeuralNetwork.nearest_value(self.f_list,
                self.neurons['f_i'][i]) # find the best f0 for f_i of this neuron
            rand_time_index = NeuralNetwork.nearest_value(self.cpdf[f_ind],
                rand[i])    # find the next spike of neuron i
            if rand_time_index == None: # if rand_time greater than time axis, ignore it
                continue
            if rand_time_index < max_time:
                max_time = rand_time_index  # keep the min time

        return max_time * self._h


    def update_fire_rate(self):
        """Update the firing rate of each neuron."""
        # homogenious part
        self.neurons['f_i'] += (self.f0 - self.neurons['f_i']) / self.tau * self._h
        # inhomogenious part
        if np.sum(self.fired) > 0:  # if at least 1 neuron fired
            self.mutual_area = self.calc_mutual_area()  # update mutual area
            self.neurons['f_i'][self.fired] += np.sum(self.mutual_area[self.fired]) * self.g    # inhomogenious increment of f_i
        else:
            pass


    def timestep(self):
        """Evolve the system one time step."""
        # decide which neurons fire at this timestep
        self.fired = np.random.random(size=self.num) < self.neurons['f_i'] * self._h
        self.update_fire_rate()     # update fire rate
        self.neurons['radius'] += self.k * self._h   # homogenious increment
        self.neurons['radius'][self.fired] -= self.k / self.f_sat   # inhomogenious decrement

    @staticmethod
    def func(d, r1, r2) -> float:
        """Calculate the intersection area of two circles and return it.

        ...
        Parameters
        ----------
        d
            Distance of the centers of two circles
        r1
            Radius of the 1st circle
        r2
            Radius of the second circle
        """
        if d < r1 + r2:     # have intersection
            if d < np.absolute(r1 - r2):    # one inside the other
                if r1 < r2:  # r1 is in r2
                    return np.pi * r1 ** 2
                else:
                    return np.pi * r2 ** 2
            else:
                part1 = r1 ** 2 * np.arccos((d*d + r1*r1 - r2*r2) / (2*d*r1))
                part2 = r2 ** 2 * np.arccos((d*d - r1*r1 + r2*r2) / (2*d*r2))
                part3 =  0.5 * np.sqrt((-d + r1 + r2) *\
                        (d - r1 + r2) * (d + r1 - r2) * (d + r1 + r2))
                return part1 + part2 - part3
        else:
            return 0

    def calc_mutual_area(self) -> np.ndarray:
        """Calculate the mutual area(2D) or volume (3D) of all neurons."""

        _mutual_area = np.zeros(shape=(self.num, self.num),dtype=float) # initialize
        r1 = np.tile(self.neurons['radius'], (self.num, 1))      # matrix of radii
        r2 = r1.T           # transpose of radii as r2

        for i in range(self.num):
            for j in range(i):
                _mutual_area[i, j] = NeuralNetwork.func(self.dist_mat[i, j],
                        r1[i, j], r2[i, j])
                _mutual_area[j, i] = NeuralNetwork.func(self.dist_mat[i, j],
                        r1[i, j], r2[i, j])
            _mutual_area[i, i] = 0.0
        return _mutual_area


    def display(self, color):
        """Plot the neurons on the canvas as disks."""
        # preparations
        r = self.neurons['radius']
        x = self.neurons['x']
        y = self.neurons['y']
        fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)

        neurons_circles = []
        for i in range(self.num):
            neurons_circles.append(plt.Circle(xy=(x[i], y[i]), radius=r[i],
                alpha=0.2, color=color))
            ax.add_patch(neurons_circles[i])

        ax.set_title(f"neuron membrane of size {self.size} and population {self.num}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-0.5, self.size + 0.5)
        ax.set_ylim(-0.5, self.size + 0.5)
        plt.show()

    def animate_system(self, color):
        def animate(i):
            """Animate for FuncAnimation."""
            for _ in range(10):
                self.timestep()

            r = self.neurons['radius']
            for ind in range(self.num):
                neurons_circles[ind].radius = r[ind]

            ax.set_title(f"step {i}, neuron population = {self.num}")

            return 0

        # preparations
        r = self.neurons['radius']
        x = self.neurons['x']
        y = self.neurons['y']
        fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)

        neurons_circles = []
        for i in range(self.num):
            neurons_circles.append(plt.Circle(xy=(x[i], y[i]), radius=r[i],
                alpha=0.2, color=color))
            ax.add_patch(neurons_circles[i])

        # ax.set_title(f"neuron membrane of size {self.size} and population {self.num}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-0.5, self.size + 0.5)
        ax.set_ylim(-0.5, self.size + 0.5)
        ani = FuncAnimation(fig, animate, interval=10, blit=False,
                save_count=500)
        plt.show()

    def render(self, duration: float=10**5, progress: bool=True) -> None:
        """Render the model for a set duration.

        ...
        Parameters
        ----------
        duration
            in seconds
        progress
            Assign True is you want to see the progress bar.
        """
        self.display('r')  # display the initial state of the system
        from time import time
        print(f"Requested to render the model for {duration} seconds\n"
                f"The value for model timestep: {self._h}\n"
                f"Beginning the render process...\n")

        n_steps = int(duration // self._h)
        interval = 1000 # take a sample every interval
        leng = n_steps // interval  # the number of intervals to loop
        arr = np.zeros(shape=(leng, self.num), dtype=float) # the sample
        if progress:    # optional progress log
            start = time()
            for i in range(leng):
                print("\rProgress: %.2f " % (i / leng * 100.0), end='')
                for _ in range(interval):   # loop on interval
                    self.timestep()
                arr[i] = np.sum(self.calc_mutual_area(), axis=0) * self.tau \
                        * self.g    # take sample
            end = time()
        else:
            start = time()
            for i in range(leng):
                for _ in range(interval):   # loop over interval
                    self.timestep()
                arr[i] = np.sum(self.calc_mutual_area(), axis=0) * self.tau \
                        * self.g    # take sample
            end = time()

        print(f"\n\nRendered the model in {end-start} seconds CPU time.")
        np.save("data.npy", arr)    # save the sample data

        self.display('r')



def test():
    """Test the system."""
    from time import time   # to calc runtime of the program
    network = NeuralNetwork(neuron_population=100)
    network.animate_system('b')

    start = time()
    duration = 5 * 10 ** 2
    num = int(duration // network._h)
    interval = 100
    array = np.zeros((num // interval, network.num))

    for i in range(num // interval):
        print(f"\rstep {i} / {num // interval}", end='')
        for _ in range(interval):
            network.timestep()
        array[i] = np.sum(network.mutual_area, axis=0) * network.tau * network.g

    print(f"runtime = {time() - start}")

    network.display('r')

    print(array[-1])
    plt.plot(np.linspace(1, num // interval, num // interval), array)
    plt.show()


if __name__ == '__main__':
    test()
