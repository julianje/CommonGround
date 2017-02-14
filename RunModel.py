import PhysicalObject
import VisualWorld
from SimpleListener import *
from Bias import *
from ComplexListener import *
from Speaker import *

# Build a set of biases
PersonBias = Bias(['color', 'size', 'subcategory'], [0.5, 0.2, 0.1])

# Create a couple of objects for different trials
T1_greencar = PhysicalObject.PhysicalObject("car", "green", "color")
T1_blackcar = PhysicalObject.PhysicalObject("car", "black", "color")
T1_redcar = PhysicalObject.PhysicalObject("car", "red", "color")
T1_yellowhat = PhysicalObject.PhysicalObject("hat", "yellow", "color")

# Create the visual world
World = VisualWorld.VisualWorld([T1_greencar, T1_blackcar, T1_redcar, T1_yellowhat])

# Create a speaker
MySpeaker = Speaker(World, PersonBias)

MySpeaker.Communicate(T1_yellowhat)

# Create a couple of objects for different trials
#T2_bigsofa = PhysicalObject.PhysicalObject("sofa", "big", "size")
#T2_smallsofa = PhysicalObject.PhysicalObject("sofa", "small", "size")
#T2_dog = PhysicalObject.PhysicalObject("dog", "None", "size")
#T2_plant = PhysicalObject.PhysicalObject("plant", "None", "size")
#
# Create the visual world
#World = VisualWorld.VisualWorld([T2_bigsofa, T2_smallsofa, T2_dog, T2_plant])
#
# Create a speaker
#MySpeaker = Speaker(World, PersonBias)
#
# MySpeaker.Communicate(T2_dog)

# Test listener model
# Build a set of biases
PersonBias = Bias(['color', 'size', 'subcategory'], [0.5, 0.2, 0.1])

# Create a couple of objects for different trials
T1_greencar = PhysicalObject.PhysicalObject("car", "green", "color")
T1_blackcar = PhysicalObject.PhysicalObject("car", "black", "color")
T1_redcar = PhysicalObject.PhysicalObject("car", "red", "color")
T1_yellowhat = PhysicalObject.PhysicalObject("hat", "yellow", "color")

# Create the visual world
World = VisualWorld.VisualWorld([T1_greencar, T1_blackcar, T1_redcar, T1_yellowhat])

CGPrior = [0, 0, 0.5, 0]  # the red car may not be common ground.

MyListener = ComplexListener(PersonBias, World, CGPrior)
res = MyListener.JointInference(['car', 'green'])
res = MyListener.JointInference(['car', None], 10000)
