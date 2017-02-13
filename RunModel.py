import PhysicalObject
from VisualWorld import *
from SimpleListener import *
from Bias import *
from Speaker import *

# Build a set of biases
PersonBias = Bias(['color', 'size', 'subcategory'], [0.5, 0.2, 0.1])

# Create a couple of objects for different trials
T1_greencar = PhysicalObject.PhysicalObject("car", "green", "color")
T1_blackcar = PhysicalObject.PhysicalObject("car", "black", "color")
T1_redcar = PhysicalObject.PhysicalObject("car", "red", "color")
T1_yellowhat = PhysicalObject.PhysicalObject("hat", "yellow", "color")

# Create the visual world
World = VisualWorld([T1_greencar, T1_blackcar, T1_redcar, T1_yellowhat])

# Create a speaker
MySpeaker = Speaker(World, PersonBias)

MySpeaker.Communicate(T1_yellowhat)
