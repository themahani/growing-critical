#!/usr/bin/env python

""" This is the implementaion of the model in Python """

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import squareform, pdist


class NeuralNetwork:
    """Form the neural network model."""
    def __init__(self, size=1, neuron_population: int=100, tau: float=0.01,
            f0: float=0.01, f_sat: float=2, g: float=500, k: float=1e-6,
            timestep: float=0.001, random_seed: int=20) -> None:
        self.size = size    # the size of the area of the square canvas
        self._num = neuron_population    # the population of the neuron in canvas
        self.dim = 2        # dimension of the canvas

        """
        hash table to store data of neurons: x, y, z, radius, fired
        x -> float
        y -> float
        z -> float
        radius -> float
        f_i -> float
        """
        self.neurons = np.recarray((self._num,),
            dtype=[('x', float), ('y', float), ('z', float), ('radius', float),
                ('f_i', float)])

        self.neurons['x'] = np.random.random(size=self._num) * self.size      # random initial position for the neuron in the canvas
        self.neurons['y'] = np.random.random(size=self._num) * self.size      # random initial position for the neuron in the canvas

        if self.dim == 3:
            self.neurons['z'] = np.random.random(size=self._num) * self.size      # random initial position for the neuron in the canvas
        else:
            self.neurons['z'] = np.zeros(shape=self._num)    # all zero z-axis

        self.neurons['radius'] = np.random.random(size=self._num) * 0.05 * self.size    # list of neuron radii as disks (2D) or spheres (3D)
        pos = np.vstack((np.vstack((self.neurons['x'], self.neurons['y'])), self.neurons['z']))
        self.dist_mat = squareform(pdist(pos.T))     # distance matrix of the neurons
        self.mutual_area = self.calc_mutual_area()      # calculate mutual area of disks(2D) or volume of spheres (3D)

        self.f0 = f0        # f0 (Hz)
        self.neurons['f_i'] = np.random.uniform(0, 1, self._num)    # initial firing rate
        self.f_sat = f_sat      # f_sat (Hz)

        self.fired = np.zeros(self._num, dtype=bool)     # see if neurons are fired
        self.tau = tau     # decay constant for firing rates
        self._h = timestep      # time step
        self.g = g            # correlation coefficient of mutual area (Hz)
        self.k = k
        self.f_sat = f_sat


        self.MAX_TIME = 500   # the latest possible spike time in seconds
        self.time_ax = np.arange(0, self.MAX_TIME ,timestep)  # initialize the time axis
        self.f_list = np.arange(0.01, 50, .1)   # list of various initial f
        cpdf_list = []  # initialize cpdf
        for f0 in self.f_list:
            isi = NeuralNetwork.pdf(self.time_ax, tau=tau, f_0=self.f0, f0=f0)  # generate isi distribution using this f0
            cpdf_list.append(np.cumsum(isi) * self._h)  # add the cpdf to list
        self.cpdf = np.array(cpdf_list)

        self.current_time = 0   # initialize system time
        # create pseudo-random number generator (helps with recreating results)
        self.rng = np.random.default_rng(seed=random_seed)


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

        mask = np.abs(arr[ind] - value) > np.abs(arr[ind-1] - value)
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
        min_time_ind = self.MAX_TIME // self._h    # initialize as the latest time possible
        rand = self.rng.uniform(0, 1, self._num)    # generate random for time
        neuron_ind = 0
        for i in range(self._num):   # do the following for each neuron
            f_ind = NeuralNetwork.nearest_value(self.f_list,
                self.neurons['f_i'][i]) # find the best f0 for f_i of this neuron
            if f_ind is None:
                continue    # if f_ind higher than the max value, try next neuron
            rand_time_index = NeuralNetwork.nearest_value(self.cpdf[f_ind],
                rand[i])    # find the next spike of neuron i
            if rand_time_index is None: # if rand_time greater than time axis, ignore it
                continue
            if rand_time_index < min_time_ind:
                min_time_ind = rand_time_index  # keep the min time
                neuron_ind = i      # record the neuron that fired

        return min_time_ind * self._h, neuron_ind

    def _update_fire_rate(self, duration: float, fired_neuron: int):
        """Update the firing rate of the neurons until the given duration"""
        self.mutual_area = self.calc_mutual_area()  # update mutual area

        # inhomogenious update
        self.neurons['f_i'] += self.mutual_area[fired_neuron] * self.g * self._h
        # homogenious update
        self.neurons['f_i'] = self.f0 - (self.f0 - self.neurons['f_i']) \
            * np.exp(-duration / self.tau)

    def _update_radius(self, duration: float, fired_neuron: int) -> None:
        """Update the radius of all neurons for the given duration"""
        # shrink the fired neuron in one time step
        self.neurons['radius'][fired_neuron] -= self.k / self.f_sat * self._h
        # global evolution of firing rates
        self.neurons['radius'] += self.k * duration


    def evolve(self, until: float):
        """Evolve the system to the next spike."""
        time_limit = self.current_time + until
        while self.current_time < time_limit:
            next_spike_time, fired_neuron = self._find_next_spike()
            self._update_fire_rate(next_spike_time, fired_neuron)   # update the firing rate
            self._update_radius(next_spike_time, fired_neuron)  # update radius
            self.current_time += next_spike_time    # update system time

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

        _mutual_area = np.zeros(shape=(self._num, self._num),dtype=float) # initialize
        r1 = np.tile(self.neurons['radius'], (self._num, 1))      # matrix of radii
        r2 = r1.T           # transpose of radii as r2

        for i in range(self._num):
            for j in range(i):
                _mutual_area[i, j] = NeuralNetwork.func(self.dist_mat[i, j],
                        r1[i, j], r2[i, j])
                _mutual_area[j, i] = NeuralNetwork.func(self.dist_mat[i, j],
                        r1[i, j], r2[i, j])
            _mutual_area[i, i] = 0.0
        return _mutual_area


    def display(self, color, save:bool = True, f_name:str = "") -> None:
        """Plot the neurons on the canvas as disks."""
        # preparations
        r = self.neurons['radius']
        x = self.neurons['x']
        y = self.neurons['y']
        fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)

        neurons_circles = []
        for i in range(self._num):
            neurons_circles.append(plt.Circle(xy=(x[i], y[i]), radius=r[i],
                alpha=0.2, color=color))
            ax.add_patch(neurons_circles[i])

        ax.set_title(f"neuron membrane of size {self.size} and population {self._num}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-0.5, self.size + 0.5)
        ax.set_ylim(-0.5, self.size + 0.5)
        if save:
            if f_name == "":
                f_name = f"time_{self.current_time}"
            plt.savefig(f_name + ".jpg", bbox_inches='tight', dpi=300)
        else:
            plt.show()

    def animate_system(self, color):
        def animate(i):
            """Animate for FuncAnimation."""
            self.evolve(100)    # evolve the system for at least 100 seconds

            r = self.neurons['radius']
            for ind in range(self._num):
                neurons_circles[ind].radius = r[ind]

            ax.set_title(f"Current System Time = {self.current_time:.2f}, Neuron Population = {self._num}")

            return 0

        # preparations
        r = self.neurons['radius']
        x = self.neurons['x']
        y = self.neurons['y']
        fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)

        neurons_circles = []
        for i in range(self._num):
            neurons_circles.append(plt.Circle(xy=(x[i], y[i]), radius=r[i],
                alpha=0.2, color=color))
            ax.add_patch(neurons_circles[i])

        # ax.set_title(f"neuron membrane of size {self.size} and population {self._num}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-0.5, self.size + 0.5)
        ax.set_ylim(-0.5, self.size + 0.5)
        ani = FuncAnimation(fig, animate, interval=1000, blit=False,
                save_count=500)
        plt.show()

    def render(self, duration: float=10**5, interval:int = 1000,
            progress: bool=True) -> None:
        """Render the model for a set duration.

        ...
        Parameters
        ----------
        duration
            in seconds
        progress
            Assign True is you want to see the progress bar.
        interval
            The system time interval between data aquisition (in seconds)
        """
        self.display('r', save=True, f_name='start')  # display the initial state of the system
        from time import time
        print(f"Requested to render the model for {duration} seconds\n"
                f"The value for model timestep: {self._h}\n"
                f"Beginning the render process...\n")

        arr = []
        fire_rate = []
        if progress:    # optional progress log
            start = time()
            while self.current_time < duration:
                self.evolve(interval)   # evolve for interval seconds
                arr.append(np.sum(self.mutual_area, axis=0) * self.tau * self.g)    # record mean mutual
                fire_rate.append(self.neurons['f_i'])   # record the firing rate of neurons
                print(f"\rProgress: {self.current_time:.2f} / {duration} seconds ", end='')
            end = time()
        else:
            start = time()
            while self.current_time < duration:
                self.evolve(interval)   # evolve for interval seconds
                arr.append(np.sum(self.mutual_area, axis=0) * self.tau * self.g)    # record mean mutual
                fire_rate.append(self.neurons['f_i'])   # record the firing rate of neurons
            end = time()

        print(f"\n\nRendered the model in {end-start} seconds CPU time.")

        np.save("mean_mutual_area.npy", np.array(arr))    # save the sample data
        np.save("firing_rage.npy", np.array(fire_rate))     # save the fire_rate data
        self.display('b', save=True, f_name='end')      # display the final state of the system



def test():
    """Test the system."""
    seed = eval(input("Enter seed:\n>> "))
    network = NeuralNetwork(neuron_population=100, k=1e-6, random_seed=seed)

    network.render(5e5, progress=True, interval=500)


if __name__ == '__main__':
    test()
