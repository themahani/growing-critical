#!/usr/bin/env python

""" test the neural_model module """

from os import path
import sys
sys.path.append(path.join(path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from numpy.testing import assert_allclose

from neural_model import NeuralNetwork
