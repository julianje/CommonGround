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

Length = 5

TL_Keys = [["large","square"],["small","rectangle"],["small","triangle"],["small","circle"],["large","triangle"]]
TR_Keys = [["large","circle"],["small","triangle"],["large","triangle"],["small","triangle"],["small","star"]]
BL_Keys = [["large","triangle"],["small","star"],["large","circle"],["small","rectangle"],["large","star"]]
BR_Keys = [["large","star"],["small","circle"],["large","square"],["large","rectangle"],["large","square"]]

Utterance_Unr = [["large","square"],["small","circle"],["small","triangle"],["large","rectangle"],["","star"]]
Utterance_Rel = [["","square"],["","circle"],["small","triangle"],["","rectangle"],["","star"]]

Utterance_List = Utterance_Rel

Verbose = False

BetaParameters = [0.39, 0.32]
SpeakerRationalityNoise = 0.055

MyFilter = Filter.Filter(3, 3)  # Common ground must have three objects.
BiasPriors = [SF.BuildBeta(BetaParameters[0], BetaParameters[1], "size")]
RawUtterance = Utterance_List[0]
if RawUtterance[0] != "":
    FirstUtterance = UT.Utterance(
        RawUtterance[1], [OF.ObjectFeature(RawUtterance[0], "size")])
else:
    FirstUtterance = UT.Utterance(RawUtterance[1])
TL = PO.PhysicalObject(
    TL_Keys[0][1], [OF.ObjectFeature(TL_Keys[0][0], "size")], "TL")
TR = PO.PhysicalObject(
    TR_Keys[0][1], [OF.ObjectFeature(TR_Keys[0][0], "size")], "TR")
BL = PO.PhysicalObject(
    BL_Keys[0][1], [OF.ObjectFeature(BL_Keys[0][0], "size")], "BL")
BR = PO.PhysicalObject(
    BR_Keys[0][1], [OF.ObjectFeature(BR_Keys[0][0], "size")], "BR")
CGPrior = BF.Belief(4, [TL.Id, TR.Id, BL.Id, BR.Id], [
                    0.5, 0.5, 0.5, 0.5])  # Priors from common ground
World = VW.VisualWorld([TL, TR, BL, BR])
MyListener = CL.ComplexListener(
    World, CGPrior, BiasPriors, MyFilter, SpeakerRationalityNoise)
if Verbose:
    print("Reasoning about first utterance")
MyListener.Infer(FirstUtterance)
TRes_A = MyListener.ComputePosterior(0)
SF.PrintCSV(TRes_A, "0", True)
TL = PO.PhysicalObject(
    TL_Keys[1][1], [OF.ObjectFeature(TL_Keys[1][0], "size")], "TL")
TR = PO.PhysicalObject(
    TR_Keys[1][1], [OF.ObjectFeature(TR_Keys[1][0], "size")], "TR")
BL = PO.PhysicalObject(
    BL_Keys[1][1], [OF.ObjectFeature(BL_Keys[1][0], "size")], "BL")
BR = PO.PhysicalObject(
    BR_Keys[1][1], [OF.ObjectFeature(BR_Keys[1][0], "size")], "BR")
World = VW.VisualWorld([TL, TR, BL, BR])
MyListener.ChangeVisualWorld(World)
if Verbose:
    print("Reasoning about second utterance")
RawUtterance = Utterance_List[1]
if RawUtterance[0] != "":
    SecondUtterance = UT.Utterance(
        RawUtterance[1], [OF.ObjectFeature(RawUtterance[0], "size")])
else:
    SecondUtterance = UT.Utterance(RawUtterance[1])
MyListener.Infer(SecondUtterance)
TRes_B = MyListener.ComputePosterior(1)
SF.PrintCSV(TRes_B, "1", False)

TL = PO.PhysicalObject(
    TL_Keys[2][1], [OF.ObjectFeature(TL_Keys[2][0], "size")], "TL")
TR = PO.PhysicalObject(
    TR_Keys[2][1], [OF.ObjectFeature(TR_Keys[2][0], "size")], "TR")
BL = PO.PhysicalObject(
    BL_Keys[2][1], [OF.ObjectFeature(BL_Keys[2][0], "size")], "BL")
BR = PO.PhysicalObject(
    BR_Keys[2][1], [OF.ObjectFeature(BR_Keys[2][0], "size")], "BR")
World = VW.VisualWorld([TL, TR, BL, BR])
MyListener.ChangeVisualWorld(World)
if Verbose:
    print("Reasoning about second utterance")
RawUtterance = Utterance_List[2]
if RawUtterance[0] != "":
    ThirdUtterance = UT.Utterance(
        RawUtterance[1], [OF.ObjectFeature(RawUtterance[0], "size")])
else:
    ThirdUtterance = UT.Utterance(RawUtterance[1])
MyListener.Infer(ThirdUtterance)
TRes_B = MyListener.ComputePosterior(2)
SF.PrintCSV(TRes_B, "2", False)

TL = PO.PhysicalObject(
    TL_Keys[3][1], [OF.ObjectFeature(TL_Keys[3][0], "size")], "TL")
TR = PO.PhysicalObject(
    TR_Keys[3][1], [OF.ObjectFeature(TR_Keys[3][0], "size")], "TR")
BL = PO.PhysicalObject(
    BL_Keys[3][1], [OF.ObjectFeature(BL_Keys[3][0], "size")], "BL")
BR = PO.PhysicalObject(
    BR_Keys[3][1], [OF.ObjectFeature(BR_Keys[3][0], "size")], "BR")
World = VW.VisualWorld([TL, TR, BL, BR])
MyListener.ChangeVisualWorld(World)
if Verbose:
    print("Reasoning about second utterance")
RawUtterance = Utterance_List[3]
if RawUtterance[0] != "":
    FourthUtterance = UT.Utterance(
        RawUtterance[1], [OF.ObjectFeature(RawUtterance[0], "size")])
else:
    FourthUtterance = UT.Utterance(RawUtterance[1])
MyListener.Infer(FourthUtterance)
TRes_B = MyListener.ComputePosterior(3)
SF.PrintCSV(TRes_B, "3", False)

TL = PO.PhysicalObject(
    TL_Keys[4][1], [OF.ObjectFeature(TL_Keys[4][0], "size")], "TL")
TR = PO.PhysicalObject(
    TR_Keys[4][1], [OF.ObjectFeature(TR_Keys[4][0], "size")], "TR")
BL = PO.PhysicalObject(
    BL_Keys[4][1], [OF.ObjectFeature(BL_Keys[4][0], "size")], "BL")
BR = PO.PhysicalObject(
    BR_Keys[4][1], [OF.ObjectFeature(BR_Keys[4][0], "size")], "BR")
World = VW.VisualWorld([TL, TR, BL, BR])
MyListener.ChangeVisualWorld(World)
if Verbose:
    print("Reasoning about second utterance")
RawUtterance = Utterance_List[4]
if RawUtterance[0] != "":
    FifthUtterance = UT.Utterance(
        RawUtterance[1], [OF.ObjectFeature(RawUtterance[0], "size")])
else:
    FifthUtterance = UT.Utterance(RawUtterance[1])
MyListener.Infer(FifthUtterance)
TRes_B = MyListener.ComputePosterior(4)
SF.PrintCSV(TRes_B, "4", False)
