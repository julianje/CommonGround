# Same as GetModelPredictions,
# but the dummy trial is the first one.
# In the original stim set. The fourth image
# of the first sequence is now the first.
# For all other sequences the third trial is now the first.

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

Trials = 28

TL_A = [["yellow", "circle"], ["green", "triangle"], ["red", "circle"], ["red", "triangle"], ["yellow", "square"], ["blue", "square"], ["red", "circle"], ["green", "triangle"], ["yellow", "triangle"], ["yellow", "circle"], ["red", "square"], ["blue", "star"], ["blue", "triangle"], [
    "red", "circle"], ["yellow", "star"], ["red", "square"], ["red", "circle"], ["green", "star"], ["red", "circle"], ["red", "star"], ["red", "circle"], ["yellow", "star"], ["red", "circle"], ["yellow", "star"], ["green", "star"], ["green", "square"], ["yellow", "triangle"], ["red", "triangle"]]
TR_A = [["red", "star"], ["red", "square"], ["blue", "square"], ["blue", "circle"], ["blue", "circle"], ["yellow", "circle"], ["blue", "triangle"], ["yellow", "star"], ["red", "circle"], ["green", "star"], ["yellow", "star"], ["yellow", "circle"], ["green", "circle"], ["green", "star"], [
    "blue", "square"], ["blue", "star"], ["blue", "square"], ["blue", "circle"], ["green", "circle"], ["green", "circle"], ["blue", "star"], ["green", "star"], ["blue", "triangle"], ["red", "square"], ["red", "triangle"], ["yellow", "circle"], ["red", "square"], ["green", "triangle"]]
BL_A = [["green", "square"], ["yellow", "star"], ["yellow", "star"], ["yellow", "square"], ["red", "triangle"], ["green", "triangle"], ["yellow", "square"], ["red", "square"], ["blue", "star"], ["red", "triangle"], ["blue", "triangle"], ["red", "triangle"], ["yellow", "square"], ["yellow", "triangle"], [
    "red", "triangle"], ["yellow", "circle"], ["yellow", "square"], ["red", "square"], ["yellow", "star"], ["yellow", "circle"], ["yellow", "square"], ["red", "square"], ["yellow", "square"], ["green", "triangle"], ["blue", "star"], ["red", "square"], ["green", "star"], ["yellow", "square"]]
BR_A = [["blue", "triangle"], ["blue", "circle"], ["green", "triangle"], ["green", "star"], ["green", "star"], ["red", "star"], ["green", "star"], ["blue", "circle"], ["green", "square"], ["blue", "square"], ["green", "circle"], ["green", "square"], ["red", "star"], ["blue", "square"], [
    "green", "circle"], ["green", "triangle"], ["green", "triangle"], ["yellow", "circle"], ["blue", "triangle"], ["blue", "square"], ["green", "star"], ["blue", "triangle"], ["green", "square"], ["blue", "star"], ["yellow", "square"], ["blue", "triangle"], ["blue", "star"], ["blue", "star"]]

TL_B = [["yellow", "triangle"], ["red", "triangle"], ["blue", "triangle"], ["yellow", "triangle"], ["red", "circle"], ["red", "square"], ["green", "square"], ["yellow", "circle"], ["green", "circle"], ["green", "triangle"], ["green", "circle"], ["blue", "circle"], ["blue", "star"], ["green", "star"], [
    "red", "circle"], ["yellow", "triangle"], ["green", "square"], ["red", "star"], ["green", "circle"], ["green", "star"], ["blue", "star"], ["yellow", "triangle"], ["yellow", "star"], ["yellow", "triangle"], ["red", "square"], ["yellow", "circle"], ["yellow", "square"], ["green", "circle"]]
TR_B = [["green", "star"], ["blue", "star"], ["yellow", "circle"], ["red", "square"], ["green", "triangle"], ["blue", "square"], ["yellow", "triangle"], ["blue", "square"], ["blue", "triangle"], ["yellow", "star"], ["yellow", "triangle"], ["green", "square"], ["red", "circle"], ["yellow", "square"], [
    "blue", "triangle"], ["green", "square"], ["red", "star"], ["blue", "circle"], ["red", "triangle"], ["yellow", "triangle"], ["red", "triangle"], ["green", "triangle"], ["red", "triangle"], ["green", "circle"], ["green", "circle"], ["red", "triangle"], ["red", "star"], ["blue", "square"]]
BL_B = [["blue", "square"], ["yellow", "circle"], ["green", "square"], ["blue", "star"], ["blue", "triangle"], ["yellow", "triangle"], ["blue", "square"], ["red", "triangle"], ["red", "circle"], ["red", "square"], ["blue", "square"], ["red", "star"], ["yellow", "circle"], ["blue", "triangle"], [
    "yellow", "square"], ["blue", "star"], ["yellow", "star"], ["yellow", "triangle"], ["blue", "star"], ["red", "circle"], ["green", "square"], ["blue", "square"], ["blue", "circle"], ["blue", "square"], ["blue", "triangle"], ["green", "square"], ["blue", "square"], ["yellow", "star"]]
BR_B = [["red", "circle"], ["green", "square"], ["red", "star"], ["green", "circle"], ["yellow", "star"], ["green", "circle"], ["red", "star"], ["green", "triangle"], ["yellow", "square"], ["blue", "circle"], ["red", "star"], ["yellow", "square"], ["green", "triangle"], ["red", "triangle"], [
    "green", "circle"], ["red", "star"], ["blue", "circle"], ["green", "circle"], ["yellow", "star"], ["blue", "circle"], ["yellow", "square"], ["red", "star"], ["green", "circle"], ["red", "triangle"], ["yellow", "square"], ["blue", "circle"], ["green", "triangle"], ["red", "star"]]

Utterance_A = [["", "circle"], ["", "star"], ["", "triangle"], ["", "triangle"], ["", "triangle"], ["", "star"], ["", "circle"], ["", "square"], ["", "square"], ["yellow", "circle"], ["blue", "triangle"], ["green", "square"], ["blue", "triangle"], [
    "yellow", "triangle"], ["green", "circle"], ["red", "square"], ["", "square"], ["", "circle"], ["", "circle"], ["", "circle"], ["", "star"], ["", "star"], ["yellow", "square"], ["blue", "star"], ["green", "star"], ["red", "square"], ["blue", "star"], ["red", "triangle"]]
Utterance_B = [["", "triangle"], ["", "square"], ["red", "star"], ["blue", "star"], ["", "triangle"], ["", "square"], ["green", "square"], ["green", "triangle"], ["green", "circle"], ["green", "triangle"], ["red", "star"], ["", "square"], ["", "circle"], ["blue", "triangle"], [
    "red", "circle"], ["red", "star"], ["", "star"], ["", "circle"], ["blue", "star"], ["blue", "circle"], ["yellow", "square"], ["yellow", "triangle"], ["blue", "circle"], ["yellow", "triangle"], ["red", "square"], ["blue", "circle"], ["blue", "square"], ["yellow", "star"]]

Verbose = False

BetaParameters = [0.39, 0.32]
SpeakerRationalityNoise = 0.055

#BetaParameters = [10, 1]
#SpeakerRationalityNoise = 0.99

# This code will only gives us the referents for the trial on the right.
# To get the referents on the trial on the left The second part of the
# code does the same, but with a flipped order.
for i in range(Trials):
    if Verbose:
        print("Running trial " + str(i))
    MyFilter = Filter.Filter(3, 3)  # Common ground must have three objects.
    BiasPriors = [SF.BuildBeta(BetaParameters[0], BetaParameters[1], "color")]
    # Fit to participant data. This is the raw proportion. Using a beta
    # distribution the estimate would be 0.055. The raw estimate is 0.255
    RawUtteranceA = Utterance_A[i]
    RawUtteranceB = Utterance_B[i]
    if RawUtteranceA[0] != "":
        FirstUtterance = UT.Utterance(
            RawUtteranceA[1], [OF.ObjectFeature(RawUtteranceA[0], "color")])
    else:
        FirstUtterance = UT.Utterance(RawUtteranceA[1])
    if RawUtteranceB[0] != "":
        SecondUtterance = UT.Utterance(
            RawUtteranceB[1], [OF.ObjectFeature(RawUtteranceB[0], "color")])
    else:
        SecondUtterance = UT.Utterance(RawUtteranceB[1])
    TA_TL = PO.PhysicalObject(
        TL_A[i][1], [OF.ObjectFeature(TL_A[i][0], "color")], "TL")
    TA_TR = PO.PhysicalObject(
        TR_A[i][1], [OF.ObjectFeature(TR_A[i][0], "color")], "TR")
    TA_BL = PO.PhysicalObject(
        BL_A[i][1], [OF.ObjectFeature(BL_A[i][0], "color")], "BL")
    TA_BR = PO.PhysicalObject(
        BR_A[i][1], [OF.ObjectFeature(BR_A[i][0], "color")], "BR")
    CGPrior = BF.Belief(4, [TA_TL.Id, TA_TR.Id, TA_BL.Id, TA_BR.Id], [
                        0.5, 0.5, 0.5, 0.5])  # Priors from common ground
    World_TA = VW.VisualWorld([TA_TL, TA_TR, TA_BL, TA_BR])
    MyListener = CL.ComplexListener(
        World_TA, CGPrior, BiasPriors, MyFilter, SpeakerRationalityNoise)
    if Verbose:
        print("Reasoning about first utterance")
    MyListener.Infer(FirstUtterance)
    TARes = MyListener.ComputePosterior(0)
    TB_TL = PO.PhysicalObject(
        TL_B[i][1], [OF.ObjectFeature(TL_B[i][0], "color")], "TL")
    TB_TR = PO.PhysicalObject(
        TR_B[i][1], [OF.ObjectFeature(TR_B[i][0], "color")], "TR")
    TB_BL = PO.PhysicalObject(
        BL_B[i][1], [OF.ObjectFeature(BL_B[i][0], "color")], "BL")
    TB_BR = PO.PhysicalObject(
        BR_B[i][1], [OF.ObjectFeature(BR_B[i][0], "color")], "BR")
    World_TB = VW.VisualWorld([TB_TL, TB_TR, TB_BL, TB_BR])
    MyListener.ChangeVisualWorld(World_TB)
    if Verbose:
        print("Reasoning about second utterance")
    MyListener.Infer(SecondUtterance)
    TBRes = MyListener.ComputePosterior(1)
    if i == 0:
        SF.PrintCSV(TBRes, str(i), True)
    else:
        SF.PrintCSV(TBRes, str(i), False)

# Re-run everything but now with second utterance first. This gives
# the same belief inferences, but now prints the inferred referent of the first event
# (which is coded as the second) while accounting for the second.
print("Running inverted order to recover referent of first visual wolrd accounting for the second one.")
for i in range(Trials):
    if Verbose:
        print("Running trial " + str(i))
    MyFilter = Filter.Filter(3, 3)  # Common ground must have three objects.
    BiasPriors = [SF.BuildBeta(BetaParameters[0], BetaParameters[1], "color")]
    # Fit to participant data. This is the raw proportion. Using abeta
    # distribution the estimate would be 0.055
    RawUtteranceA = Utterance_B[i]
    RawUtteranceB = Utterance_A[i]
    if RawUtteranceA[0] != "":
        FirstUtterance = UT.Utterance(
            RawUtteranceA[1], [OF.ObjectFeature(RawUtteranceA[0], "color")])
    else:
        FirstUtterance = UT.Utterance(RawUtteranceA[1])
    if RawUtteranceB[0] != "":
        SecondUtterance = UT.Utterance(
            RawUtteranceB[1], [OF.ObjectFeature(RawUtteranceB[0], "color")])
    else:
        SecondUtterance = UT.Utterance(RawUtteranceB[1])
    TA_TL = PO.PhysicalObject(
        TL_B[i][1], [OF.ObjectFeature(TL_B[i][0], "color")], "TL")
    TA_TR = PO.PhysicalObject(
        TR_B[i][1], [OF.ObjectFeature(TR_B[i][0], "color")], "TR")
    TA_BL = PO.PhysicalObject(
        BL_B[i][1], [OF.ObjectFeature(BL_B[i][0], "color")], "BL")
    TA_BR = PO.PhysicalObject(
        BR_B[i][1], [OF.ObjectFeature(BR_B[i][0], "color")], "BR")
    CGPrior = BF.Belief(4, [TA_TL.Id, TA_TR.Id, TA_BL.Id, TA_BR.Id], [
                        0.5, 0.5, 0.5, 0.5])  # Priors from common ground
    World_TA = VW.VisualWorld([TA_TL, TA_TR, TA_BL, TA_BR])
    MyListener = CL.ComplexListener(
        World_TA, CGPrior, BiasPriors, MyFilter, SpeakerRationalityNoise)
    if Verbose:
        print("Reasoning about first utterance")
    MyListener.Infer(FirstUtterance)
    TARes = MyListener.ComputePosterior(0)
    TB_TL = PO.PhysicalObject(
        TL_A[i][1], [OF.ObjectFeature(TL_A[i][0], "color")], "TL")
    TB_TR = PO.PhysicalObject(
        TR_A[i][1], [OF.ObjectFeature(TR_A[i][0], "color")], "TR")
    TB_BL = PO.PhysicalObject(
        BL_A[i][1], [OF.ObjectFeature(BL_A[i][0], "color")], "BL")
    TB_BR = PO.PhysicalObject(
        BR_A[i][1], [OF.ObjectFeature(BR_A[i][0], "color")], "BR")
    World_TB = VW.VisualWorld([TB_TL, TB_TR, TB_BL, TB_BR])
    MyListener.ChangeVisualWorld(World_TB)
    if Verbose:
        print("Reasoning about second utterance")
    MyListener.Infer(SecondUtterance)
    TBRes = MyListener.ComputePosterior(1)
    if i == 0:
        SF.PrintCSV(TBRes, str(i), True)
    else:
        SF.PrintCSV(TBRes, str(i), False)
