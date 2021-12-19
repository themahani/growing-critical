#!/usr/bin/env python

""" test the neural_model module """

from os import path
import sys
sys.path.append(path.join(path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from numpy.testing import assert_allclose

from neural_model import NeuralNetwork

def test_circles_intersection1():
    assert NeuralNetwork.func(10, 2, 3) == 0

def test_circles_intersection2():
    assert_allclose(NeuralNetwork.func(0, 1, 4), np.pi, 1e-5)

def test_circles_intersection3():
    assert_allclose(NeuralNetwork.func(0, 4, 1), np.pi, 1e-5)
