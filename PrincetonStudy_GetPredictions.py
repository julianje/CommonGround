import PhysicalObject
import VisualWorld
from SimpleListener import *
from Bias import *
from ComplexListener import *
from Speaker import *
import Filter
import Belief as BF
import SupportingFunctions as SF

# Load objects, visual worlds, and utterances.
execfile('PrincetonStudy_ExperimentLoader.py')

MyFilter = Filter.Filter(3,4) # Three objects are visible
SpeakerRationalityNoise_Color = 0.047
SpeakerRationalityNoise_Size = 0.163
SpeakerRationalityNoise_Category = 0.22
ColorPriorParams = [0.43,0.82]
SizePriorParams = [0.28,1.18]
CategoryPriorParams = [0.29,0.22]

# Build priors depending on which corner you're questioning
TLSpot = [0.5, 1, 1, 1]
TRSpot = [1, 0.5, 1, 1]
BLSpot = [1, 1, 0.5, 1]
BRSpot = [1, 1, 1, 0.5]

CGPrior = BF.Belief(4,[COL_A_A.objects[0].Id,COL_A_A.objects[1].Id,COL_A_A.objects[2].Id,COL_A_A.objects[3].Id], TRSpot)
ProductionPriors = [SF.BuildBeta(ColorPriorParams[0], ColorPriorParams[1], "color")]
MyListener = ComplexListener(COL_A_A, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Color)
MyListener.Infer(COL_A_A_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "COL-A-1", True)

CGPrior = BF.Belief(4,[COL_A_B.objects[0].Id,COL_A_B.objects[1].Id,COL_A_B.objects[2].Id,COL_A_B.objects[3].Id], BRSpot)
ProductionPriors = [SF.BuildBeta(ColorPriorParams[0], ColorPriorParams[1], "color")]
MyListener = ComplexListener(COL_A_B, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Color)
MyListener.Infer(COL_A_B_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "COL-A-2", False)

CGPrior = BF.Belief(4,[COL_A_C.objects[0].Id,COL_A_C.objects[1].Id,COL_A_C.objects[2].Id,COL_A_C.objects[3].Id], BLSpot)
ProductionPriors = [SF.BuildBeta(ColorPriorParams[0], ColorPriorParams[1], "color")]
MyListener = ComplexListener(COL_A_C, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Color)
MyListener.Infer(COL_A_C_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "COL-A-3", False)

CGPrior = BF.Belief(4,[COL_A_D.objects[0].Id,COL_A_D.objects[1].Id,COL_A_D.objects[2].Id,COL_A_D.objects[3].Id], TLSpot)
ProductionPriors = [SF.BuildBeta(ColorPriorParams[0], ColorPriorParams[1], "color")]
MyListener = ComplexListener(COL_A_D, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Color)
MyListener.Infer(COL_A_D_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "COL-A-4", False)

CGPrior = BF.Belief(4,[COL_B_A.objects[0].Id,COL_B_A.objects[1].Id,COL_B_A.objects[2].Id,COL_B_A.objects[3].Id], BRSpot)
ProductionPriors = [SF.BuildBeta(ColorPriorParams[0], ColorPriorParams[1], "color")]
MyListener = ComplexListener(COL_B_A, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Color)
MyListener.Infer(COL_B_A_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "COL-B-1", True)

CGPrior = BF.Belief(4,[COL_B_B.objects[0].Id,COL_B_B.objects[1].Id,COL_B_B.objects[2].Id,COL_B_B.objects[3].Id], BLSpot)
ProductionPriors = [SF.BuildBeta(ColorPriorParams[0], ColorPriorParams[1], "color")]
MyListener = ComplexListener(COL_B_B, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Color)
MyListener.Infer(COL_B_B_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "COL-B-2", False)

CGPrior = BF.Belief(4,[COL_B_C.objects[0].Id,COL_B_C.objects[1].Id,COL_B_C.objects[2].Id,COL_B_C.objects[3].Id], TLSpot)
ProductionPriors = [SF.BuildBeta(ColorPriorParams[0], ColorPriorParams[1], "color")]
MyListener = ComplexListener(COL_B_C, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Color)
MyListener.Infer(COL_B_C_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "COL-B-3", False)

CGPrior = BF.Belief(4,[COL_B_D.objects[0].Id,COL_B_D.objects[1].Id,COL_B_D.objects[2].Id,COL_B_D.objects[3].Id], TRSpot)
ProductionPriors = [SF.BuildBeta(ColorPriorParams[0], ColorPriorParams[1], "color")]
MyListener = ComplexListener(COL_B_D, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Color)
MyListener.Infer(COL_B_D_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "COL-B-4", False)

CGPrior = BF.Belief(4,[COL_C_A.objects[0].Id,COL_C_A.objects[1].Id,COL_C_A.objects[2].Id,COL_C_A.objects[3].Id], BRSpot)
ProductionPriors = [SF.BuildBeta(ColorPriorParams[0], ColorPriorParams[1], "color")]
MyListener = ComplexListener(COL_C_A, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Color)
MyListener.Infer(COL_C_A_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "COL-C-1", True)

CGPrior = BF.Belief(4,[COL_C_B.objects[0].Id,COL_C_B.objects[1].Id,COL_C_B.objects[2].Id,COL_C_B.objects[3].Id], BLSpot)
ProductionPriors = [SF.BuildBeta(ColorPriorParams[0], ColorPriorParams[1], "color")]
MyListener = ComplexListener(COL_C_B, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Color)
MyListener.Infer(COL_C_B_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "COL-C-2", False)

CGPrior = BF.Belief(4,[COL_C_C.objects[0].Id,COL_C_C.objects[1].Id,COL_C_C.objects[2].Id,COL_C_C.objects[3].Id], TLSpot)
ProductionPriors = [SF.BuildBeta(ColorPriorParams[0], ColorPriorParams[1], "color")]
MyListener = ComplexListener(COL_C_C, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Color)
MyListener.Infer(COL_C_C_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "COL-C-3", False)

CGPrior = BF.Belief(4,[COL_C_D.objects[0].Id,COL_C_D.objects[1].Id,COL_C_D.objects[2].Id,COL_C_D.objects[3].Id], TRSpot)
ProductionPriors = [SF.BuildBeta(ColorPriorParams[0], ColorPriorParams[1], "color")]
MyListener = ComplexListener(COL_C_D, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Color)
MyListener.Infer(COL_C_D_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "COL-C-4", False)

CGPrior = BF.Belief(4,[COL_D_A.objects[0].Id,COL_D_A.objects[1].Id,COL_D_A.objects[2].Id,COL_D_A.objects[3].Id], BRSpot)
ProductionPriors = [SF.BuildBeta(ColorPriorParams[0], ColorPriorParams[1], "color")]
MyListener = ComplexListener(COL_D_A, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Color)
MyListener.Infer(COL_D_A_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "COL-D-1", True)

CGPrior = BF.Belief(4,[COL_D_B.objects[0].Id,COL_D_B.objects[1].Id,COL_D_B.objects[2].Id,COL_D_B.objects[3].Id], BLSpot)
ProductionPriors = [SF.BuildBeta(ColorPriorParams[0], ColorPriorParams[1], "color")]
MyListener = ComplexListener(COL_D_B, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Color)
MyListener.Infer(COL_D_B_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "COL-D-2", False)

CGPrior = BF.Belief(4,[COL_D_C.objects[0].Id,COL_D_C.objects[1].Id,COL_D_C.objects[2].Id,COL_D_C.objects[3].Id], TLSpot)
ProductionPriors = [SF.BuildBeta(ColorPriorParams[0], ColorPriorParams[1], "color")]
MyListener = ComplexListener(COL_D_C, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Color)
MyListener.Infer(COL_D_C_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "COL-D-3", False)

CGPrior = BF.Belief(4,[COL_D_D.objects[0].Id,COL_D_D.objects[1].Id,COL_D_D.objects[2].Id,COL_D_D.objects[3].Id], TRSpot)
ProductionPriors = [SF.BuildBeta(ColorPriorParams[0], ColorPriorParams[1], "color")]
MyListener = ComplexListener(COL_D_D, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Color)
MyListener.Infer(COL_D_D_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "COL-D-4", False)

# CATEGORY

CGPrior = BF.Belief(4,[CAT_A_A.objects[0].Id,CAT_A_A.objects[1].Id,CAT_A_A.objects[2].Id,CAT_A_A.objects[3].Id], TRSpot)
ProductionPriors = [SF.BuildBeta(CategoryPriorParams[0], CategoryPriorParams[1], "category")]
MyListener = ComplexListener(CAT_A_A, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Category)
MyListener.Infer(CAT_A_A_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "CAT-A-1", True)

CGPrior = BF.Belief(4,[CAT_A_B.objects[0].Id,CAT_A_B.objects[1].Id,CAT_A_B.objects[2].Id,CAT_A_B.objects[3].Id], BRSpot)
ProductionPriors = [SF.BuildBeta(CategoryPriorParams[0], CategoryPriorParams[1], "category")]
MyListener = ComplexListener(CAT_A_B, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Category)
MyListener.Infer(CAT_A_B_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "CAT-A-2", False)

CGPrior = BF.Belief(4,[CAT_A_C.objects[0].Id,CAT_A_C.objects[1].Id,CAT_A_C.objects[2].Id,CAT_A_C.objects[3].Id], BLSpot)
ProductionPriors = [SF.BuildBeta(CategoryPriorParams[0], CategoryPriorParams[1], "category")]
MyListener = ComplexListener(CAT_A_C, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Category)
MyListener.Infer(CAT_A_C_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "CAT-A-3", False)

CGPrior = BF.Belief(4,[CAT_A_D.objects[0].Id,CAT_A_D.objects[1].Id,CAT_A_D.objects[2].Id,CAT_A_D.objects[3].Id], TLSpot)
ProductionPriors = [SF.BuildBeta(CategoryPriorParams[0], CategoryPriorParams[1], "category")]
MyListener = ComplexListener(CAT_A_D, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Category)
MyListener.Infer(CAT_A_D_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "CAT-A-4", False)

CGPrior = BF.Belief(4,[CAT_B_A.objects[0].Id,CAT_B_A.objects[1].Id,CAT_B_A.objects[2].Id,CAT_B_A.objects[3].Id], BRSpot)
ProductionPriors = [SF.BuildBeta(CategoryPriorParams[0], CategoryPriorParams[1], "category")]
MyListener = ComplexListener(CAT_B_A, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Category)
MyListener.Infer(CAT_B_A_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "CAT-B-1", True)

CGPrior = BF.Belief(4,[CAT_B_B.objects[0].Id,CAT_B_B.objects[1].Id,CAT_B_B.objects[2].Id,CAT_B_B.objects[3].Id], BLSpot)
ProductionPriors = [SF.BuildBeta(CategoryPriorParams[0], CategoryPriorParams[1], "category")]
MyListener = ComplexListener(CAT_B_B, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Category)
MyListener.Infer(CAT_B_B_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "CAT-B-2", False)

CGPrior = BF.Belief(4,[CAT_B_C.objects[0].Id,CAT_B_C.objects[1].Id,CAT_B_C.objects[2].Id,CAT_B_C.objects[3].Id], TLSpot)
ProductionPriors = [SF.BuildBeta(CategoryPriorParams[0], CategoryPriorParams[1], "category")]
MyListener = ComplexListener(CAT_B_C, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Category)
MyListener.Infer(CAT_B_C_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "CAT-B-3", False)

CGPrior = BF.Belief(4,[CAT_B_D.objects[0].Id,CAT_B_D.objects[1].Id,CAT_B_D.objects[2].Id,CAT_B_D.objects[3].Id], TRSpot)
ProductionPriors = [SF.BuildBeta(CategoryPriorParams[0], CategoryPriorParams[1], "category")]
MyListener = ComplexListener(CAT_B_D, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Category)
MyListener.Infer(CAT_B_D_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "CAT-B-4", False)

CGPrior = BF.Belief(4,[CAT_C_A.objects[0].Id,CAT_C_A.objects[1].Id,CAT_C_A.objects[2].Id,CAT_C_A.objects[3].Id], BRSpot)
ProductionPriors = [SF.BuildBeta(CategoryPriorParams[0], CategoryPriorParams[1], "category")]
MyListener = ComplexListener(CAT_C_A, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Category)
MyListener.Infer(CAT_C_A_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "CAT-C-1", True)

CGPrior = BF.Belief(4,[CAT_C_B.objects[0].Id,CAT_C_B.objects[1].Id,CAT_C_B.objects[2].Id,CAT_C_B.objects[3].Id], BLSpot)
ProductionPriors = [SF.BuildBeta(CategoryPriorParams[0], CategoryPriorParams[1], "category")]
MyListener = ComplexListener(CAT_C_B, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Category)
MyListener.Infer(CAT_C_B_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "CAT-C-2", False)

CGPrior = BF.Belief(4,[CAT_C_C.objects[0].Id,CAT_C_C.objects[1].Id,CAT_C_C.objects[2].Id,CAT_C_C.objects[3].Id], TLSpot)
ProductionPriors = [SF.BuildBeta(CategoryPriorParams[0], CategoryPriorParams[1], "category")]
MyListener = ComplexListener(CAT_C_C, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Category)
MyListener.Infer(CAT_C_C_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "CAT-C-3", False)

CGPrior = BF.Belief(4,[CAT_C_D.objects[0].Id,CAT_C_D.objects[1].Id,CAT_C_D.objects[2].Id,CAT_C_D.objects[3].Id], TRSpot)
ProductionPriors = [SF.BuildBeta(CategoryPriorParams[0], CategoryPriorParams[1], "category")]
MyListener = ComplexListener(CAT_C_D, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Category)
MyListener.Infer(CAT_C_D_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "CAT-C-4", False)

CGPrior = BF.Belief(4,[CAT_D_A.objects[0].Id,CAT_D_A.objects[1].Id,CAT_D_A.objects[2].Id,CAT_D_A.objects[3].Id], BRSpot)
ProductionPriors = [SF.BuildBeta(CategoryPriorParams[0], CategoryPriorParams[1], "category")]
MyListener = ComplexListener(CAT_D_A, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Category)
MyListener.Infer(CAT_D_A_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "CAT-D-1", True)

CGPrior = BF.Belief(4,[CAT_D_B.objects[0].Id,CAT_D_B.objects[1].Id,CAT_D_B.objects[2].Id,CAT_D_B.objects[3].Id], BLSpot)
ProductionPriors = [SF.BuildBeta(CategoryPriorParams[0], CategoryPriorParams[1], "category")]
MyListener = ComplexListener(CAT_D_B, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Category)
MyListener.Infer(CAT_D_B_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "CAT-D-2", False)

CGPrior = BF.Belief(4,[CAT_D_C.objects[0].Id,CAT_D_C.objects[1].Id,CAT_D_C.objects[2].Id,CAT_D_C.objects[3].Id], TLSpot)
ProductionPriors = [SF.BuildBeta(CategoryPriorParams[0], CategoryPriorParams[1], "category")]
MyListener = ComplexListener(CAT_D_C, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Category)
MyListener.Infer(CAT_D_C_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "CAT-D-3", False)

CGPrior = BF.Belief(4,[CAT_D_D.objects[0].Id,CAT_D_D.objects[1].Id,CAT_D_D.objects[2].Id,CAT_D_D.objects[3].Id], TRSpot)
ProductionPriors = [SF.BuildBeta(CategoryPriorParams[0], CategoryPriorParams[1], "category")]
MyListener = ComplexListener(CAT_D_D, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Category)
MyListener.Infer(CAT_D_D_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "CAT-D-4", False)

# SIZE

CGPrior = BF.Belief(4,[SIZE_A_A.objects[0].Id,SIZE_A_A.objects[1].Id,SIZE_A_A.objects[2].Id,SIZE_A_A.objects[3].Id], TRSpot)
ProductionPriors = [SF.BuildBeta(SizePriorParams[0], SizePriorParams[1], "size")]
MyListener = ComplexListener(SIZE_A_A, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Size)
MyListener.Infer(SIZE_A_A_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "SIZE-A-1", True)

CGPrior = BF.Belief(4,[SIZE_A_B.objects[0].Id,SIZE_A_B.objects[1].Id,SIZE_A_B.objects[2].Id,SIZE_A_B.objects[3].Id], BRSpot)
ProductionPriors = [SF.BuildBeta(SizePriorParams[0], SizePriorParams[1], "size")]
MyListener = ComplexListener(SIZE_A_B, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Size)
MyListener.Infer(SIZE_A_B_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "SIZE-A-2", False)

CGPrior = BF.Belief(4,[SIZE_A_C.objects[0].Id,SIZE_A_C.objects[1].Id,SIZE_A_C.objects[2].Id,SIZE_A_C.objects[3].Id], BLSpot)
ProductionPriors = [SF.BuildBeta(SizePriorParams[0], SizePriorParams[1], "size")]
MyListener = ComplexListener(SIZE_A_C, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Size)
MyListener.Infer(SIZE_A_C_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "SIZE-A-3", False)

CGPrior = BF.Belief(4,[SIZE_A_D.objects[0].Id,SIZE_A_D.objects[1].Id,SIZE_A_D.objects[2].Id,SIZE_A_D.objects[3].Id], TLSpot)
ProductionPriors = [SF.BuildBeta(SizePriorParams[0], SizePriorParams[1], "size")]
MyListener = ComplexListener(SIZE_A_D, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Size)
MyListener.Infer(SIZE_A_D_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "SIZE-A-4", False)

CGPrior = BF.Belief(4,[SIZE_B_A.objects[0].Id,SIZE_B_A.objects[1].Id,SIZE_B_A.objects[2].Id,SIZE_B_A.objects[3].Id], BRSpot)
ProductionPriors = [SF.BuildBeta(SizePriorParams[0], SizePriorParams[1], "size")]
MyListener = ComplexListener(SIZE_B_A, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Size)
MyListener.Infer(SIZE_B_A_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "SIZE-B-1", True)

CGPrior = BF.Belief(4,[SIZE_B_B.objects[0].Id,SIZE_B_B.objects[1].Id,SIZE_B_B.objects[2].Id,SIZE_B_B.objects[3].Id], BLSpot)
ProductionPriors = [SF.BuildBeta(SizePriorParams[0], SizePriorParams[1], "size")]
MyListener = ComplexListener(SIZE_B_B, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Size)
MyListener.Infer(SIZE_B_B_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "SIZE-B-2", False)

CGPrior = BF.Belief(4,[SIZE_B_C.objects[0].Id,SIZE_B_C.objects[1].Id,SIZE_B_C.objects[2].Id,SIZE_B_C.objects[3].Id], TLSpot)
ProductionPriors = [SF.BuildBeta(SizePriorParams[0], SizePriorParams[1], "size")]
MyListener = ComplexListener(SIZE_B_C, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Size)
MyListener.Infer(SIZE_B_C_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "SIZE-B-3", False)

CGPrior = BF.Belief(4,[SIZE_B_D.objects[0].Id,SIZE_B_D.objects[1].Id,SIZE_B_D.objects[2].Id,SIZE_B_D.objects[3].Id], TRSpot)
ProductionPriors = [SF.BuildBeta(SizePriorParams[0], SizePriorParams[1], "size")]
MyListener = ComplexListener(SIZE_B_D, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Size)
MyListener.Infer(SIZE_B_D_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "SIZE-B-4", False)

CGPrior = BF.Belief(4,[SIZE_C_A.objects[0].Id,SIZE_C_A.objects[1].Id,SIZE_C_A.objects[2].Id,SIZE_C_A.objects[3].Id], BRSpot)
ProductionPriors = [SF.BuildBeta(SizePriorParams[0], SizePriorParams[1], "size")]
MyListener = ComplexListener(SIZE_C_A, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Size)
MyListener.Infer(SIZE_C_A_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "SIZE-C-1", True)

CGPrior = BF.Belief(4,[SIZE_C_B.objects[0].Id,SIZE_C_B.objects[1].Id,SIZE_C_B.objects[2].Id,SIZE_C_B.objects[3].Id], BLSpot)
ProductionPriors = [SF.BuildBeta(SizePriorParams[0], SizePriorParams[1], "size")]
MyListener = ComplexListener(SIZE_C_B, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Size)
MyListener.Infer(SIZE_C_B_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "SIZE-C-2", False)

CGPrior = BF.Belief(4,[SIZE_C_C.objects[0].Id,SIZE_C_C.objects[1].Id,SIZE_C_C.objects[2].Id,SIZE_C_C.objects[3].Id], TLSpot)
ProductionPriors = [SF.BuildBeta(SizePriorParams[0], SizePriorParams[1], "size")]
MyListener = ComplexListener(SIZE_C_C, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Size)
MyListener.Infer(SIZE_C_C_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "SIZE-C-3", False)

CGPrior = BF.Belief(4,[SIZE_C_D.objects[0].Id,SIZE_C_D.objects[1].Id,SIZE_C_D.objects[2].Id,SIZE_C_D.objects[3].Id], TRSpot)
ProductionPriors = [SF.BuildBeta(SizePriorParams[0], SizePriorParams[1], "size")]
MyListener = ComplexListener(SIZE_C_D, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Size)
MyListener.Infer(SIZE_C_D_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "SIZE-C-4", False)

CGPrior = BF.Belief(4,[SIZE_D_A.objects[0].Id,SIZE_D_A.objects[1].Id,SIZE_D_A.objects[2].Id,SIZE_D_A.objects[3].Id], BRSpot)
ProductionPriors = [SF.BuildBeta(SizePriorParams[0], SizePriorParams[1], "size")]
MyListener = ComplexListener(SIZE_D_A, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Size)
MyListener.Infer(SIZE_D_A_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "SIZE-D-1", True)

CGPrior = BF.Belief(4,[SIZE_D_B.objects[0].Id,SIZE_D_B.objects[1].Id,SIZE_D_B.objects[2].Id,SIZE_D_B.objects[3].Id], BLSpot)
ProductionPriors = [SF.BuildBeta(SizePriorParams[0], SizePriorParams[1], "size")]
MyListener = ComplexListener(SIZE_D_B, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Size)
MyListener.Infer(SIZE_D_B_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "SIZE-D-2", False)

CGPrior = BF.Belief(4,[SIZE_D_C.objects[0].Id,SIZE_D_C.objects[1].Id,SIZE_D_C.objects[2].Id,SIZE_D_C.objects[3].Id], TLSpot)
ProductionPriors = [SF.BuildBeta(SizePriorParams[0], SizePriorParams[1], "size")]
MyListener = ComplexListener(SIZE_D_C, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Size)
MyListener.Infer(SIZE_D_C_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "SIZE-D-3", False)

CGPrior = BF.Belief(4,[SIZE_D_D.objects[0].Id,SIZE_D_D.objects[1].Id,SIZE_D_D.objects[2].Id,SIZE_D_D.objects[3].Id], TRSpot)
ProductionPriors = [SF.BuildBeta(SizePriorParams[0], SizePriorParams[1], "size")]
MyListener = ComplexListener(SIZE_D_D, CGPrior, ProductionPriors, MyFilter, SpeakerRationalityNoise_Size)
MyListener.Infer(SIZE_D_D_Utterance)
Result = MyListener.ComputePosterior()
SF.PrintCSV(Result, "SIZE-D-4", False)
