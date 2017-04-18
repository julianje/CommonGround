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
import Filter

###########################
####### SEQUENCE 1 ########
###########################

T1_Utterance = UT.Utterance("triangle")
T2_Utterance = UT.Utterance("square", [OF.ObjectFeature("green", "color")])
T3_Utterance = UT.Utterance("circle")
T4_Utterance = UT.Utterance("rectangle")
T5_Utterance = UT.Utterance("star")
T6_Utterance = UT.Utterance("rectangle")

# General parameters
# Build a filter for the hypothesis space
# First parameter is minimum number of objects the speaker may see.
MyFilter = Filter.Filter(3, 3)  # Common ground must have three objects (one object is not seen).
BiasPriors = [SF.BuildBeta(0.39, 0.32, "color")]  # Parameters estimated from participant data.
SpeakerRationalityNoise = 0.055  # Fit from participant data.


T1_TL = PO.PhysicalObject("rectangle", [OF.ObjectFeature("yellow", "color")], "TL")
T1_TR = PO.PhysicalObject("triangle", [OF.ObjectFeature("green", "color")], "TR")
T1_BL = PO.PhysicalObject("triangle", [OF.ObjectFeature("blue", "color")], "BL")
T1_BR = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "BR")
CGPrior = BF.Belief(4, [T1_TL.Id, T1_TR.Id, T1_BL.Id, T1_BR.Id], [0.5, 0.5, 0.5, 0.5])  # Priors from common ground
World_T1 = VW.VisualWorld([T1_TL, T1_TR, T1_BL, T1_BR])
MyListener = CL.ComplexListener(World_T1, CGPrior, BiasPriors, MyFilter, SpeakerRationalityNoise)
MyListener.Infer(T1_Utterance)
T1Res = MyListener.ComputePosterior(0)

T2_TL = PO.PhysicalObject("square", [OF.ObjectFeature("green", "color")], "TL")
T2_TR = PO.PhysicalObject("square", [OF.ObjectFeature("yellow", "color")], "TR")
T2_BL = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "BL")
T2_BR = PO.PhysicalObject("star", [OF.ObjectFeature("blue", "color")], "BR")
World_T2 = VW.VisualWorld([T2_TL, T2_TR, T2_BL, T2_BR])
MyListener.ChangeVisualWorld(World_T2)
MyListener.Infer(T2_Utterance)
T2Res = MyListener.ComputePosterior(1)

T3_TL = PO.PhysicalObject("star", [OF.ObjectFeature("green", "color")], "TL")
T3_TR = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "TR")
T3_BL = PO.PhysicalObject("circle", [OF.ObjectFeature("yellow", "color")], "BL")
T3_BR = PO.PhysicalObject("sqaure", [OF.ObjectFeature("blue", "color")], "BR")
World_T3 = VW.VisualWorld([T3_TL, T3_TR, T3_BL, T3_BR])
MyListener.ChangeVisualWorld(World_T3)
MyListener.Infer(T3_Utterance)
T3Res = MyListener.ComputePosterior(2)

T4_TL = PO.PhysicalObject("circle", [OF.ObjectFeature("blue", "color")], "TL")
T4_TR = PO.PhysicalObject("triangle", [OF.ObjectFeature("yellow", "color")], "TR")
T4_BL = PO.PhysicalObject("star", [OF.ObjectFeature("green", "color")], "BL")
T4_BR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("red", "color")], "BR")
World_T4 = VW.VisualWorld([T4_TL, T4_TR, T4_BL, T4_BR])
MyListener.ChangeVisualWorld(World_T4)
MyListener.Infer(T4_Utterance)
T4Res = MyListener.ComputePosterior(3)

T5_TL = PO.PhysicalObject("square", [OF.ObjectFeature("red", "color")], "TL")
T5_TR = PO.PhysicalObject("star", [OF.ObjectFeature("blue", "color")], "TR")
T5_BL = PO.PhysicalObject("rectangle", [OF.ObjectFeature("yellow", "color")], "BL")
T5_BR = PO.PhysicalObject("circle", [OF.ObjectFeature("green", "color")], "BR")
World_T5 = VW.VisualWorld([T5_TL, T5_TR, T5_BL, T5_BR])
MyListener.ChangeVisualWorld(World_T5)
MyListener.Infer(T5_Utterance)
T5Res = MyListener.ComputePosterior(4)

T6_TL = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "TL")
T6_TR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("green", "color")], "TR")
T6_BL = PO.PhysicalObject("rectangle", [OF.ObjectFeature("blue", "color")], "BL")
T6_BR = PO.PhysicalObject("star", [OF.ObjectFeature("yellow", "color")], "BR")
World_T6 = VW.VisualWorld([T6_TL, T6_TR, T6_BL, T6_BR])
MyListener.ChangeVisualWorld(World_T6)
MyListener.Infer(T6_Utterance)
T6Res = MyListener.ComputePosterior(5)

#SF.PrintCSV(T0Res, "0")
SF.PrintCSV(T1Res, "1-1", True)
SF.PrintCSV(T2Res, "1-2", False)
SF.PrintCSV(T3Res, "1-3", False)
SF.PrintCSV(T4Res, "1-4", False)
SF.PrintCSV(T5Res, "1-5", False)
SF.PrintCSV(T6Res, "1-6", False)

###########################
####### SEQUENCE 2 ########
###########################

T1_Utterance = UT.Utterance("triangle")
T2_Utterance = UT.Utterance("square", [OF.ObjectFeature("green", "color")])
T3_Utterance = UT.Utterance("rectangle")
T4_Utterance = UT.Utterance("star")
T5_Utterance = UT.Utterance("circle", [OF.ObjectFeature("yellow", "color")])
T6_Utterance = UT.Utterance("rectangle")

T1_TL = PO.PhysicalObject("rectangle", [OF.ObjectFeature("yellow", "color")], "TL")
T1_TR = PO.PhysicalObject("triangle", [OF.ObjectFeature("green", "color")], "TR")
T1_BL = PO.PhysicalObject("triangle", [OF.ObjectFeature("blue", "color")], "BL")
T1_BR = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "BR")
CGPrior = BF.Belief(4, [T1_TL.Id, T1_TR.Id, T1_BL.Id, T1_BR.Id], [0.5, 0.5, 0.5, 0.5])  # Priors from common ground
World_T1 = VW.VisualWorld([T1_TL, T1_TR, T1_BL, T1_BR])
MyListener = CL.ComplexListener(World_T1, CGPrior, BiasPriors, MyFilter, SpeakerRationalityNoise)
MyListener.Infer(T1_Utterance)
T1Res = MyListener.ComputePosterior(0)

T2_TL = PO.PhysicalObject("square", [OF.ObjectFeature("green", "color")], "TL")
T2_TR = PO.PhysicalObject("square", [OF.ObjectFeature("yellow", "color")], "TR")
T2_BL = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "BL")
T2_BR = PO.PhysicalObject("star", [OF.ObjectFeature("blue", "color")], "BR")
World_T2 = VW.VisualWorld([T2_TL, T2_TR, T2_BL, T2_BR])
MyListener.ChangeVisualWorld(World_T2)
MyListener.Infer(T2_Utterance)
T2Res = MyListener.ComputePosterior(1)

T3_TL = PO.PhysicalObject("circle", [OF.ObjectFeature("blue", "color")], "TL")
T3_TR = PO.PhysicalObject("triangle", [OF.ObjectFeature("yellow", "color")], "TR")
T3_BL = PO.PhysicalObject("star", [OF.ObjectFeature("green", "color")], "BL")
T3_BR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("red", "color")], "BR")
World_T3 = VW.VisualWorld([T3_TL, T3_TR, T3_BL, T3_BR])
MyListener.ChangeVisualWorld(World_T3)
MyListener.Infer(T3_Utterance)
T3Res = MyListener.ComputePosterior(2)

T4_TL = PO.PhysicalObject("star", [OF.ObjectFeature("blue", "color")], "TL")
T4_TR = PO.PhysicalObject("circle", [OF.ObjectFeature("green", "color")], "TR")
T4_BL = PO.PhysicalObject("square", [OF.ObjectFeature("red", "color")], "BL")
T4_BR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("yellow", "color")], "BR")
World_T4 = VW.VisualWorld([T4_TL, T4_TR, T4_BL, T4_BR])
MyListener.ChangeVisualWorld(World_T4)
MyListener.Infer(T4_Utterance)
T4Res = MyListener.ComputePosterior(3)

T5_TL = PO.PhysicalObject("square", [OF.ObjectFeature("green", "color")], "TL")
T5_TR = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "TR")
T5_BL = PO.PhysicalObject("star", [OF.ObjectFeature("blue", "color")], "BL")
T5_BR = PO.PhysicalObject("circle", [OF.ObjectFeature("yellow", "color")], "BR")
World_T5 = VW.VisualWorld([T5_TL, T5_TR, T5_BL, T5_BR])
MyListener.ChangeVisualWorld(World_T5)
MyListener.Infer(T5_Utterance)
T5Res = MyListener.ComputePosterior(4)

T6_TL = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "TL")
T6_TR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("green", "color")], "TR")
T6_BL = PO.PhysicalObject("rectangle", [OF.ObjectFeature("blue", "color")], "BL")
T6_BR = PO.PhysicalObject("star", [OF.ObjectFeature("yellow", "color")], "BR")
World_T6 = VW.VisualWorld([T6_TL, T6_TR, T6_BL, T6_BR])
MyListener.ChangeVisualWorld(World_T6)
MyListener.Infer(T6_Utterance)
T6Res = MyListener.ComputePosterior(5)

#SF.PrintCSV(T0Res, "0")
SF.PrintCSV(T1Res, "2-1", True)
SF.PrintCSV(T2Res, "2-2", False)
SF.PrintCSV(T3Res, "2-3", False)
SF.PrintCSV(T4Res, "2-4", False)
SF.PrintCSV(T5Res, "2-5", False)
SF.PrintCSV(T6Res, "2-6", False)

###########################
####### SEQUENCE 3 ########
###########################

T1_Utterance = UT.Utterance("triangle")
T2_Utterance = UT.Utterance("square", [OF.ObjectFeature("green", "color")])
T3_Utterance = UT.Utterance("rectangle", [OF.ObjectFeature("red", "color")])
T4_Utterance = UT.Utterance("star", [OF.ObjectFeature("blue", "color")])
T5_Utterance = UT.Utterance("circle", [OF.ObjectFeature("yellow", "color")])
T6_Utterance = UT.Utterance("rectangle")

T1_TL = PO.PhysicalObject("rectangle", [OF.ObjectFeature("yellow", "color")], "TL")
T1_TR = PO.PhysicalObject("triangle", [OF.ObjectFeature("green", "color")], "TR")
T1_BL = PO.PhysicalObject("triangle", [OF.ObjectFeature("blue", "color")], "BL")
T1_BR = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "BR")
CGPrior = BF.Belief(4, [T1_TL.Id, T1_TR.Id, T1_BL.Id, T1_BR.Id], [0.5, 0.5, 0.5, 0.5])  # Priors from common ground
World_T1 = VW.VisualWorld([T1_TL, T1_TR, T1_BL, T1_BR])
MyListener = CL.ComplexListener(World_T1, CGPrior, BiasPriors, MyFilter, SpeakerRationalityNoise)
MyListener.Infer(T1_Utterance)
T1Res = MyListener.ComputePosterior(0)

T2_TL = PO.PhysicalObject("square", [OF.ObjectFeature("green", "color")], "TL")
T2_TR = PO.PhysicalObject("square", [OF.ObjectFeature("yellow", "color")], "TR")
T2_BL = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "BL")
T2_BR = PO.PhysicalObject("star", [OF.ObjectFeature("blue", "color")], "BR")
World_T2 = VW.VisualWorld([T2_TL, T2_TR, T2_BL, T2_BR])
MyListener.ChangeVisualWorld(World_T2)
MyListener.Infer(T2_Utterance)
T2Res = MyListener.ComputePosterior(1)

T3_TL = PO.PhysicalObject("circle", [OF.ObjectFeature("blue", "color")], "TL")
T3_TR = PO.PhysicalObject("triangle", [OF.ObjectFeature("yellow", "color")], "TR")
T3_BL = PO.PhysicalObject("star", [OF.ObjectFeature("green", "color")], "BL")
T3_BR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("red", "color")], "BR")
World_T3 = VW.VisualWorld([T3_TL, T3_TR, T3_BL, T3_BR])
MyListener.ChangeVisualWorld(World_T3)
MyListener.Infer(T3_Utterance)
T3Res = MyListener.ComputePosterior(2)

T4_TL = PO.PhysicalObject("star", [OF.ObjectFeature("blue", "color")], "TL")
T4_TR = PO.PhysicalObject("circle", [OF.ObjectFeature("green", "color")], "TR")
T4_BL = PO.PhysicalObject("square", [OF.ObjectFeature("red", "color")], "BL")
T4_BR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("yellow", "color")], "BR")
World_T4 = VW.VisualWorld([T4_TL, T4_TR, T4_BL, T4_BR])
MyListener.ChangeVisualWorld(World_T4)
MyListener.Infer(T4_Utterance)
T4Res = MyListener.ComputePosterior(3)

T5_TL = PO.PhysicalObject("square", [OF.ObjectFeature("green", "color")], "TL")
T5_TR = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "TR")
T5_BL = PO.PhysicalObject("star", [OF.ObjectFeature("blue", "color")], "BL")
T5_BR = PO.PhysicalObject("circle", [OF.ObjectFeature("yellow", "color")], "BR")
World_T5 = VW.VisualWorld([T5_TL, T5_TR, T5_BL, T5_BR])
MyListener.ChangeVisualWorld(World_T5)
MyListener.Infer(T5_Utterance)
T5Res = MyListener.ComputePosterior(4)

T6_TL = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "TL")
T6_TR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("green", "color")], "TR")
T6_BL = PO.PhysicalObject("rectangle", [OF.ObjectFeature("blue", "color")], "BL")
T6_BR = PO.PhysicalObject("star", [OF.ObjectFeature("yellow", "color")], "BR")
World_T6 = VW.VisualWorld([T6_TL, T6_TR, T6_BL, T6_BR])
MyListener.ChangeVisualWorld(World_T6)
MyListener.Infer(T6_Utterance)
T6Res = MyListener.ComputePosterior(5)

#SF.PrintCSV(T0Res, "0")
SF.PrintCSV(T1Res, "3-1", True)
SF.PrintCSV(T2Res, "3-2", False)
SF.PrintCSV(T3Res, "3-3", False)
SF.PrintCSV(T4Res, "3-4", False)
SF.PrintCSV(T5Res, "3-5", False)
SF.PrintCSV(T6Res, "3-6", False)

###########################
####### SEQUENCE 4 ########
###########################

T1_Utterance = UT.Utterance("triangle")
T2_Utterance = UT.Utterance("star", [OF.ObjectFeature("blue", "color")])
T3_Utterance = UT.Utterance("rectangle")
T4_Utterance = UT.Utterance("square", [OF.ObjectFeature("green", "color")])
T5_Utterance = UT.Utterance("circle", [OF.ObjectFeature("yellow", "color")])
T6_Utterance = UT.Utterance("rectangle")


# General parameters
# Build a filter for the hypothesis space
MyFilter = Filter.Filter(3, 3)  # Common ground must have three objects (one object is not seen).
BiasPriors = [SF.BuildBeta(0.55, 0.82, "color")]  # Parameters estimated from participant data.

T1_TL = PO.PhysicalObject("rectangle", [OF.ObjectFeature("yellow", "color")], "TL")
T1_TR = PO.PhysicalObject("triangle", [OF.ObjectFeature("green", "color")], "TR")
T1_BL = PO.PhysicalObject("triangle", [OF.ObjectFeature("blue", "color")], "BL")
T1_BR = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "BR")
CGPrior = BF.Belief(4, [T1_TL.Id, T1_TR.Id, T1_BL.Id, T1_BR.Id], [0.5, 0.5, 0.5, 0.5])  # Priors from common ground
World_T1 = VW.VisualWorld([T1_TL, T1_TR, T1_BL, T1_BR])
MyListener = CL.ComplexListener(World_T1, CGPrior, BiasPriors, MyFilter, SpeakerRationalityNoise)
MyListener.Infer(T1_Utterance)
T1Res = MyListener.ComputePosterior(0)

T2_TL = PO.PhysicalObject("star", [OF.ObjectFeature("blue", "color")], "TL")
T2_TR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("green", "color")], "TR")
T2_BL = PO.PhysicalObject("square", [OF.ObjectFeature("red", "color")], "BL")
T2_BR = PO.PhysicalObject("circle", [OF.ObjectFeature("yellow", "color")], "BR")
World_T2 = VW.VisualWorld([T2_TL, T2_TR, T2_BL, T2_BR])
MyListener.ChangeVisualWorld(World_T2)
MyListener.Infer(T2_Utterance)
T2Res = MyListener.ComputePosterior(1)

T3_TL = PO.PhysicalObject("circle", [OF.ObjectFeature("blue", "color")], "TL")
T3_TR = PO.PhysicalObject("triangle", [OF.ObjectFeature("yellow", "color")], "TR")
T3_BL = PO.PhysicalObject("star", [OF.ObjectFeature("green", "color")], "BL")
T3_BR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("red", "color")], "BR")
World_T3 = VW.VisualWorld([T3_TL, T3_TR, T3_BL, T3_BR])
MyListener.ChangeVisualWorld(World_T3)
MyListener.Infer(T3_Utterance)
T3Res = MyListener.ComputePosterior(2)

T4_TL = PO.PhysicalObject("square", [OF.ObjectFeature("green", "color")], "TL")
T4_TR = PO.PhysicalObject("square", [OF.ObjectFeature("yellow", "color")], "TR")
T4_BL = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "BL")
T4_BR = PO.PhysicalObject("star", [OF.ObjectFeature("blue", "color")], "BR")
World_T4 = VW.VisualWorld([T4_TL, T4_TR, T4_BL, T4_BR])
MyListener.ChangeVisualWorld(World_T4)
MyListener.Infer(T4_Utterance)
T4Res = MyListener.ComputePosterior(3)

T5_TL = PO.PhysicalObject("square", [OF.ObjectFeature("green", "color")], "TL")
T5_TR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("red", "color")], "TR")
T5_BL = PO.PhysicalObject("star", [OF.ObjectFeature("blue", "color")], "BL")
T5_BR = PO.PhysicalObject("circle", [OF.ObjectFeature("yellow", "color")], "BR")
World_T5 = VW.VisualWorld([T5_TL, T5_TR, T5_BL, T5_BR])
MyListener.ChangeVisualWorld(World_T5)
MyListener.Infer(T5_Utterance)
T5Res = MyListener.ComputePosterior(4)

T6_TL = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "TL")
T6_TR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("green", "color")], "TR")
T6_BL = PO.PhysicalObject("rectangle", [OF.ObjectFeature("blue", "color")], "BL")
T6_BR = PO.PhysicalObject("star", [OF.ObjectFeature("yellow", "color")], "BR")
World_T6 = VW.VisualWorld([T6_TL, T6_TR, T6_BL, T6_BR])
MyListener.ChangeVisualWorld(World_T6)
MyListener.Infer(T6_Utterance)
T6Res = MyListener.ComputePosterior(5)

#SF.PrintCSV(T0Res, "0")
SF.PrintCSV(T1Res, "4-1", True)
SF.PrintCSV(T2Res, "4-2", False)
SF.PrintCSV(T3Res, "4-3", False)
SF.PrintCSV(T4Res, "4-4", False)
SF.PrintCSV(T5Res, "4-5", False)
SF.PrintCSV(T6Res, "4-6", False)

###########################
####### SEQUENCE 5 ########
###########################

T1_Utterance = UT.Utterance("triangle")
T2_Utterance = UT.Utterance("star", [OF.ObjectFeature("blue", "color")])
T3_Utterance = UT.Utterance("rectangle", [OF.ObjectFeature("red", "color")])
T4_Utterance = UT.Utterance("square")
T5_Utterance = UT.Utterance("circle", [OF.ObjectFeature("yellow", "color")])
T6_Utterance = UT.Utterance("rectangle")

T1_TL = PO.PhysicalObject("rectangle", [OF.ObjectFeature("yellow", "color")], "TL")
T1_TR = PO.PhysicalObject("triangle", [OF.ObjectFeature("green", "color")], "TR")
T1_BL = PO.PhysicalObject("triangle", [OF.ObjectFeature("blue", "color")], "BL")
T1_BR = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "BR")
CGPrior = BF.Belief(4, [T1_TL.Id, T1_TR.Id, T1_BL.Id, T1_BR.Id], [0.5, 0.5, 0.5, 0.5])  # Priors from common ground
World_T1 = VW.VisualWorld([T1_TL, T1_TR, T1_BL, T1_BR])
MyListener = CL.ComplexListener(World_T1, CGPrior, BiasPriors, MyFilter, SpeakerRationalityNoise)
MyListener.Infer(T1_Utterance)
T1Res = MyListener.ComputePosterior(0)

T2_TL = PO.PhysicalObject("star", [OF.ObjectFeature("blue", "color")], "TL")
T2_TR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("green", "color")], "TR")
T2_BL = PO.PhysicalObject("square", [OF.ObjectFeature("red", "color")], "BL")
T2_BR = PO.PhysicalObject("circle", [OF.ObjectFeature("yellow", "color")], "BR")
World_T2 = VW.VisualWorld([T2_TL, T2_TR, T2_BL, T2_BR])
MyListener.ChangeVisualWorld(World_T2)
MyListener.Infer(T2_Utterance)
T2Res = MyListener.ComputePosterior(1)

T3_TL = PO.PhysicalObject("circle", [OF.ObjectFeature("blue", "color")], "TL")
T3_TR = PO.PhysicalObject("triangle", [OF.ObjectFeature("yellow", "color")], "TR")
T3_BL = PO.PhysicalObject("star", [OF.ObjectFeature("green", "color")], "BL")
T3_BR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("red", "color")], "BR")
World_T3 = VW.VisualWorld([T3_TL, T3_TR, T3_BL, T3_BR])
MyListener.ChangeVisualWorld(World_T3)
MyListener.Infer(T3_Utterance)
T3Res = MyListener.ComputePosterior(2)

T4_TL = PO.PhysicalObject("square", [OF.ObjectFeature("green", "color")], "TL")
T4_TR = PO.PhysicalObject("star", [OF.ObjectFeature("blue", "color")], "TR")
T4_BL = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "BL")
T4_BR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("yellow", "color")], "BR")
World_T4 = VW.VisualWorld([T4_TL, T4_TR, T4_BL, T4_BR])
MyListener.ChangeVisualWorld(World_T4)
MyListener.Infer(T4_Utterance)
T4Res = MyListener.ComputePosterior(3)

T5_TL = PO.PhysicalObject("square", [OF.ObjectFeature("green", "color")], "TL")
T5_TR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("red", "color")], "TR")
T5_BL = PO.PhysicalObject("star", [OF.ObjectFeature("blue", "color")], "BL")
T5_BR = PO.PhysicalObject("circle", [OF.ObjectFeature("yellow", "color")], "BR")
World_T5 = VW.VisualWorld([T5_TL, T5_TR, T5_BL, T5_BR])
MyListener.ChangeVisualWorld(World_T5)
MyListener.Infer(T5_Utterance)
T5Res = MyListener.ComputePosterior(4)

T6_TL = PO.PhysicalObject("circle", [OF.ObjectFeature("red", "color")], "TL")
T6_TR = PO.PhysicalObject("rectangle", [OF.ObjectFeature("green", "color")], "TR")
T6_BL = PO.PhysicalObject("rectangle", [OF.ObjectFeature("blue", "color")], "BL")
T6_BR = PO.PhysicalObject("star", [OF.ObjectFeature("yellow", "color")], "BR")
World_T6 = VW.VisualWorld([T6_TL, T6_TR, T6_BL, T6_BR])
MyListener.ChangeVisualWorld(World_T6)
MyListener.Infer(T6_Utterance)
T6Res = MyListener.ComputePosterior(5)

#SF.PrintCSV(T0Res, "0")
SF.PrintCSV(T1Res, "5-1", True)
SF.PrintCSV(T2Res, "5-2", False)
SF.PrintCSV(T3Res, "5-3", False)
SF.PrintCSV(T4Res, "5-4", False)
SF.PrintCSV(T5Res, "5-5", False)
SF.PrintCSV(T6Res, "5-6", False)
