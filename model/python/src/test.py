#!/usr/bin/env python

"""Test the class with a minimal system and predictable results"""

import numpy as np
import matplotlib.pyplot as plt

from neural_model_inverse import NeuralNetwork


def main():
    """main body"""
    pop = 5
    net = NeuralNetwork(neuron_population=pop, random_seed=0,
        prefix=f"testrun-pop{pop}-")
    net.render(50.e5, interval=1000)


if __name__ == "__main__":
    main()
