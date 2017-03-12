import ObjectFeature as OF
import PhysicalObject as PO
import VisualWorld as VW
import SimpleListener as SL
import ComplexListener as CL
import Utterance as UT
import Bias
import Belief as BF
import Speaker
import SupportingFunctions as SF

#####################################
# Two objects with one feature each #
#####################################

rose = PO.PhysicalObject("flower", [OF.ObjectFeature("red", "color")], "rose")
sunflower = PO.PhysicalObject(
    "flower", [OF.ObjectFeature("yellow", "color")], "sunflower")

World = VW.VisualWorld([rose, sunflower])

# prior over the visual world
MyCGPrior = BF.Belief(2, [rose.Id, sunflower.Id], [0.5, 0.5])

# prior over the speaker's production biases. In this case, only color matters
# Use the Beta build in supporting functions
BiasPriors = [SF.BuildBeta(5, 3, "color", 0.5)]

MyListener = CL.ComplexListener(World, MyCGPrior, BiasPriors)

Utterance_flower = UT.Utterance("flower")  # Ambiguous
Utterance_yellowflower = UT.Utterance(
    "flower", [OF.ObjectFeature("yellow", "color")])  # Ambiguous

MyListener.Infer(Utterance_flower)
res = MyListener.ComputePosterior()

#####################################
# Two objects with two features each #
#####################################

rose = PO.PhysicalObject("flower", [OF.ObjectFeature(
    "yellow", "color"), OF.ObjectFeature("small", "size")], "rose")
sunflower = PO.PhysicalObject("flower", [OF.ObjectFeature(
    "yellow", "color"), OF.ObjectFeature("big", "size")], "sunflower")

World = VW.VisualWorld([rose, sunflower])

# prior over the visual world
MyCGPrior = BF.Belief(2, [rose.Id, sunflower.Id], [0.5, 0.5])

# prior over the speaker's production biases. In this case, only color matters
# Use the Beta build in supporting functions
BiasPriors = [SF.BuildBeta(5, 3, "color"), SF.BuildBeta(3, 5, "size")]

MyListener = CL.ComplexListener(World, MyCGPrior, BiasPriors)

Utterance_flower = UT.Utterance("flower")  # Ambiguous
Utterance_yellowflower = UT.Utterance(
    "flower", [OF.ObjectFeature("yellow", "color")])  # Ambiguous

MyListener.Infer(Utterance_flower)
res = MyListener.ComputePosterior()

#####################################
#        Color paradigm test        #
#####################################

import ObjectFeature as OF
import PhysicalObject as PO
import VisualWorld as VW
import SimpleListener as SL
import ComplexListener as CL
import Utterance as UT
import Bias
import Belief as BF
import Speaker
import Filter
import SupportingFunctions as SF

# Build a filter for the hypothesis space
MyFilter = Filter.Filter(3)  # Common ground must have three objects.

T1_TL = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "TL")
T1_TR = PO.PhysicalObject("triangle", [OF.ObjectFeature("green", "color")], "TR")
T1_BL = PO.PhysicalObject("triangle", [OF.ObjectFeature("blue", "color")], "BL")
T1_BR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("yellow", "color")], "BR")

World_T1 = VW.VisualWorld([T1_TL, T1_TR, T1_BL, T1_BR])
CGPrior = BF.Belief(4, [T1_TL.Id, T1_TR.Id, T1_BL.Id, T1_BR.Id], [0.5, 0.5, 0.5, 0.5])
BiasPriors = [SF.BuildBeta(2, 1, "color")]
T1_Utterance = UT.Utterance("triangle")

MyListener = CL.ComplexListener(World_T1, CGPrior, BiasPriors, MyFilter)

#T0Res = MyListener.ComputePosterior()
MyListener.Infer(T1_Utterance)
T1Res = MyListener.ComputePosterior(0)

T2_TL = PO.PhysicalObject("square", [OF.ObjectFeature("black", "color")], "TL")
T2_TR = PO.PhysicalObject("square", [OF.ObjectFeature("green", "color")], "TR")
T2_BL = PO.PhysicalObject("triangle", [OF.ObjectFeature("blue", "color")], "BL")
T2_BR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("yellow", "color")], "BR")

World_T2 = VW.VisualWorld([T2_TL, T2_TR, T2_BL, T2_BR])
MyListener.ChangeVisualWorld(World_T2)

T2_Utterance = UT.Utterance("square", [OF.ObjectFeature("black", "color")])

MyListener.Infer(T2_Utterance)
T2Res = MyListener.ComputePosterior(1)

T3_TL = PO.PhysicalObject("square", [OF.ObjectFeature("green", "color")], "TL")
T3_TR = PO.PhysicalObject("circle", [OF.ObjectFeature("blue", "color")], "TR")
T3_BL = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "BL")
T3_BR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("yellow", "color")], "BR")

World_T3 = VW.VisualWorld([T3_TL, T3_TR, T3_BL, T3_BR])
MyListener.ChangeVisualWorld(World_T3)

T3_Utterance = UT.Utterance("circle")

MyListener.Infer(T3_Utterance)
T3Res = MyListener.ComputePosterior(2)

T4_TL = PO.PhysicalObject("square", [OF.ObjectFeature("yellow", "color")], "TL")
T4_TR = PO.PhysicalObject("triangle", [OF.ObjectFeature("blue", "color")], "TR")
T4_BL = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "BL")
T4_BR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("green", "color")], "BR")

World_T4 = VW.VisualWorld([T4_TL, T4_TR, T4_BL, T4_BR])
MyListener.ChangeVisualWorld(World_T4)

T4_Utterance = UT.Utterance("triangle")

MyListener.Infer(T4_Utterance)
T4Res = MyListener.ComputePosterior(3)

T5_TL = PO.PhysicalObject("square", [OF.ObjectFeature("yellow", "color")], "TL")
T5_TR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("green", "color")], "TR")
T5_BL = PO.PhysicalObject("rectangle", [OF.ObjectFeature("red", "color")], "BL")
T5_BR = PO.PhysicalObject("triangle", [OF.ObjectFeature("blue", "color")], "BR")

World_T5 = VW.VisualWorld([T5_TL, T5_TR, T5_BL, T5_BR])
MyListener.ChangeVisualWorld(World_T5)

T5_Utterance = UT.Utterance("rectangle")

MyListener.Infer(T5_Utterance)
T5Res = MyListener.ComputePosterior(4)

#SF.PrintCSV(T0Res, "0")
SF.PrintCSV(T1Res, "1", True)
SF.PrintCSV(T2Res, "2", False)
SF.PrintCSV(T3Res, "3", False)
SF.PrintCSV(T4Res, "4", False)
SF.PrintCSV(T5Res, "5", False)

# VERSION WITH ALL COLOR #
##########################
##########################
##########################


import ObjectFeature as OF
import PhysicalObject as PO
import VisualWorld as VW
import SimpleListener as SL
import ComplexListener as CL
import Utterance as UT
import Bias
import Belief as BF
import Speaker
import Filter
import SupportingFunctions as SF

# Build a filter for the hypothesis space
MyFilter = Filter.Filter(3)  # Common ground must have three objects.

T1_TL = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "TL")
T1_TR = PO.PhysicalObject("triangle", [OF.ObjectFeature("green", "color")], "TR")
T1_BL = PO.PhysicalObject("triangle", [OF.ObjectFeature("blue", "color")], "BL")
T1_BR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("yellow", "color")], "BR")

World_T1 = VW.VisualWorld([T1_TL, T1_TR, T1_BL, T1_BR])
CGPrior = BF.Belief(4, [T1_TL.Id, T1_TR.Id, T1_BL.Id, T1_BR.Id], [0.5, 0.5, 0.5, 0.5])
BiasPriors = [SF.BuildBeta(2, 1, "color")]
T1_Utterance = UT.Utterance("triangle", [OF.ObjectFeature("green", "color")])

MyListener = CL.ComplexListener(World_T1, CGPrior, BiasPriors, MyFilter)

#T0Res = MyListener.ComputePosterior()
MyListener.Infer(T1_Utterance)
T1Res = MyListener.ComputePosterior(0)

T2_TL = PO.PhysicalObject("square", [OF.ObjectFeature("black", "color")], "TL")
T2_TR = PO.PhysicalObject("square", [OF.ObjectFeature("green", "color")], "TR")
T2_BL = PO.PhysicalObject("triangle", [OF.ObjectFeature("blue", "color")], "BL")
T2_BR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("yellow", "color")], "BR")

World_T2 = VW.VisualWorld([T2_TL, T2_TR, T2_BL, T2_BR])
MyListener.ChangeVisualWorld(World_T2)

T2_Utterance = UT.Utterance("square", [OF.ObjectFeature("black", "color")])

MyListener.Infer(T2_Utterance)
T2Res = MyListener.ComputePosterior(1)

T3_TL = PO.PhysicalObject("square", [OF.ObjectFeature("green", "color")], "TL")
T3_TR = PO.PhysicalObject("circle", [OF.ObjectFeature("blue", "color")], "TR")
T3_BL = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "BL")
T3_BR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("yellow", "color")], "BR")

World_T3 = VW.VisualWorld([T3_TL, T3_TR, T3_BL, T3_BR])
MyListener.ChangeVisualWorld(World_T3)

T3_Utterance = UT.Utterance("circle", [OF.ObjectFeature("blue", "color")])

MyListener.Infer(T3_Utterance)
T3Res = MyListener.ComputePosterior(2)

T4_TL = PO.PhysicalObject("square", [OF.ObjectFeature("yellow", "color")], "TL")
T4_TR = PO.PhysicalObject("triangle", [OF.ObjectFeature("blue", "color")], "TR")
T4_BL = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "BL")
T4_BR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("green", "color")], "BR")

World_T4 = VW.VisualWorld([T4_TL, T4_TR, T4_BL, T4_BR])
MyListener.ChangeVisualWorld(World_T4)

T4_Utterance = UT.Utterance("triangle", [OF.ObjectFeature("blue", "color")])

MyListener.Infer(T4_Utterance)
T4Res = MyListener.ComputePosterior(3)

T5_TL = PO.PhysicalObject("square", [OF.ObjectFeature("yellow", "color")], "TL")
T5_TR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("green", "color")], "TR")
T5_BL = PO.PhysicalObject("rectangle", [OF.ObjectFeature("red", "color")], "BL")
T5_BR = PO.PhysicalObject("triangle", [OF.ObjectFeature("blue", "color")], "BR")

World_T5 = VW.VisualWorld([T5_TL, T5_TR, T5_BL, T5_BR])
MyListener.ChangeVisualWorld(World_T5)

T5_Utterance = UT.Utterance("rectangle", [OF.ObjectFeature("green", "color")])

MyListener.Infer(T5_Utterance)
T5Res = MyListener.ComputePosterior(4)

#SF.PrintCSV(T0Res, "0")
SF.PrintCSV(T1Res, "1", True)
SF.PrintCSV(T2Res, "2", False)
SF.PrintCSV(T3Res, "3", False)
SF.PrintCSV(T4Res, "4", False)
SF.PrintCSV(T5Res, "5", False)

