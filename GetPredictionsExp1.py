import PhysicalObject
import VisualWorld
from SimpleListener import *
from Bias import *
from ComplexListener import *
from Speaker import *

# Build a set of biases
PersonBias = Bias(['color', 'size', 'category'], [0.5, 0.2, 0.1])

# Load objects
execfile('ExperimentLoader.py')


