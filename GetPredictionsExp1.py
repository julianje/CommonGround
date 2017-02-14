import PhysicalObject
import VisualWorld
from SimpleListener import *
from Bias import *
from ComplexListener import *
from Speaker import *

# Supporting function that outputs things as a simple csv


def PrintCSV(trialname, results, header=False):
    """
    Take the results from a JointInference call
    on the ComplexListener class and print results as a short csv.
    """
    if header:
        sys.stdout.write("Trial,CommonGround,Referent\n")
    # recover the referent:
    BestGuessIndex = results[3].index(max(results[3]))
    # Translate it into the name in the experiment.
    # Hardcoded for now
    if BestGuessIndex == 0:
        BestGuess = "TL"
    if BestGuessIndex == 1:
        BestGuess = "TR"
    if BestGuessIndex == 2:
        BestGuess = "BL"
    if BestGuessIndex == 3:
        BestGuess = "BR"
    sys.stdout.write(str(trialname)+","+str(results[0])+","+BestGuess+"\n")

# Number of samples to estimate posteriors
Samples = 10000

# Build a set of biases
PersonBias = Bias(['color', 'size', 'category'], [0.5, 0.2, 0.1])

# Load objects, visual worlds, and utterances.
execfile('ExperimentLoader.py')

# For each experiment build a listener model,
# run the utterance, and print the csv
MyListener = ComplexListener(PersonBias, COL_A_A, COL_A_A_CGPrior)
res = MyListener.JointInference(COL_A_A_Utterance, Samples, False)
PrintCSV("COL_A_A", res, True)
MyListener = ComplexListener(PersonBias, COL_A_B, COL_A_B_CGPrior)
res = MyListener.JointInference(COL_A_B_Utterance, Samples, False)
PrintCSV("COL_A_B", res, False)
MyListener = ComplexListener(PersonBias, COL_A_C, COL_A_C_CGPrior)
res = MyListener.JointInference(COL_A_C_Utterance, Samples, False)
PrintCSV("COL_A_C", res, False)
MyListener = ComplexListener(PersonBias, COL_A_D, COL_A_D_CGPrior)
res = MyListener.JointInference(COL_A_D_Utterance, Samples, False)
PrintCSV("COL_A_D", res, False)
MyListener = ComplexListener(PersonBias, COL_B_A, COL_B_A_CGPrior)
res = MyListener.JointInference(COL_B_A_Utterance, Samples, False)
PrintCSV("COL_B_A", res, False)
MyListener = ComplexListener(PersonBias, COL_B_B, COL_B_B_CGPrior)
res = MyListener.JointInference(COL_B_B_Utterance, Samples, False)
PrintCSV("COL_B_B", res, False)
MyListener = ComplexListener(PersonBias, COL_B_C, COL_B_C_CGPrior)
res = MyListener.JointInference(COL_B_C_Utterance, Samples, False)
PrintCSV("COL_B_C", res, False)
MyListener = ComplexListener(PersonBias, COL_B_D, COL_B_D_CGPrior)
res = MyListener.JointInference(COL_B_D_Utterance, Samples, False)
PrintCSV("COL_B_D", res, False)
MyListener = ComplexListener(PersonBias, COL_C_A, COL_C_A_CGPrior)
res = MyListener.JointInference(COL_C_A_Utterance, Samples, False)
PrintCSV("COL_C_A", res, False)
MyListener = ComplexListener(PersonBias, COL_C_B, COL_C_B_CGPrior)
res = MyListener.JointInference(COL_C_B_Utterance, Samples, False)
PrintCSV("COL_C_B", res, False)
MyListener = ComplexListener(PersonBias, COL_C_C, COL_C_C_CGPrior)
res = MyListener.JointInference(COL_C_C_Utterance, Samples, False)
PrintCSV("COL_C_C", res, False)
MyListener = ComplexListener(PersonBias, COL_C_D, COL_C_D_CGPrior)
res = MyListener.JointInference(COL_C_D_Utterance, Samples, False)
PrintCSV("COL_C_D", res, False)
MyListener = ComplexListener(PersonBias, COL_D_A, COL_D_A_CGPrior)
res = MyListener.JointInference(COL_D_A_Utterance, Samples, False)
PrintCSV("COL_D_A", res, False)
MyListener = ComplexListener(PersonBias, COL_D_B, COL_D_B_CGPrior)
res = MyListener.JointInference(COL_D_B_Utterance, Samples, False)
PrintCSV("COL_D_B", res, False)
MyListener = ComplexListener(PersonBias, COL_D_C, COL_D_C_CGPrior)
res = MyListener.JointInference(COL_D_C_Utterance, Samples, False)
PrintCSV("COL_D_C", res, False)
MyListener = ComplexListener(PersonBias, COL_D_D, COL_D_D_CGPrior)
res = MyListener.JointInference(COL_D_D_Utterance, Samples, False)
PrintCSV("COL_D_D", res, False)
MyListener = ComplexListener(PersonBias, CAT_A_A, CAT_A_A_CGPrior)
res = MyListener.JointInference(CAT_A_A_Utterance, Samples, False)
PrintCSV("CAT_A_A", res, False)
MyListener = ComplexListener(PersonBias, CAT_A_B, CAT_A_B_CGPrior)
res = MyListener.JointInference(CAT_A_B_Utterance, Samples, False)
PrintCSV("CAT_A_B", res, False)
MyListener = ComplexListener(PersonBias, CAT_A_C, CAT_A_C_CGPrior)
res = MyListener.JointInference(CAT_A_C_Utterance, Samples, False)
PrintCSV("CAT_A_C", res, False)
MyListener = ComplexListener(PersonBias, CAT_A_D, CAT_A_D_CGPrior)
res = MyListener.JointInference(CAT_A_D_Utterance, Samples, False)
PrintCSV("CAT_A_D", res, False)
MyListener = ComplexListener(PersonBias, CAT_B_A, CAT_B_A_CGPrior)
res = MyListener.JointInference(CAT_B_A_Utterance, Samples, False)
PrintCSV("CAT_B_A", res, False)
MyListener = ComplexListener(PersonBias, CAT_B_B, CAT_B_B_CGPrior)
res = MyListener.JointInference(CAT_B_B_Utterance, Samples, False)
PrintCSV("CAT_B_B", res, False)
MyListener = ComplexListener(PersonBias, CAT_B_C, CAT_B_C_CGPrior)
res = MyListener.JointInference(CAT_B_C_Utterance, Samples, False)
PrintCSV("CAT_B_C", res, False)
MyListener = ComplexListener(PersonBias, CAT_B_D, CAT_B_D_CGPrior)
res = MyListener.JointInference(CAT_B_D_Utterance, Samples, False)
PrintCSV("CAT_B_D", res, False)
MyListener = ComplexListener(PersonBias, CAT_C_A, CAT_C_A_CGPrior)
res = MyListener.JointInference(CAT_C_A_Utterance, Samples, False)
PrintCSV("CAT_C_A", res, False)
MyListener = ComplexListener(PersonBias, CAT_C_B, CAT_C_B_CGPrior)
res = MyListener.JointInference(CAT_C_B_Utterance, Samples, False)
PrintCSV("CAT_C_B", res, False)
MyListener = ComplexListener(PersonBias, CAT_C_C, CAT_C_C_CGPrior)
res = MyListener.JointInference(CAT_C_C_Utterance, Samples, False)
PrintCSV("CAT_C_C", res, False)
MyListener = ComplexListener(PersonBias, CAT_C_D, CAT_C_D_CGPrior)
res = MyListener.JointInference(CAT_C_D_Utterance, Samples, False)
PrintCSV("CAT_C_D", res, False)
MyListener = ComplexListener(PersonBias, CAT_D_A, CAT_D_A_CGPrior)
res = MyListener.JointInference(CAT_D_A_Utterance, Samples, False)
PrintCSV("CAT_D_A", res, False)
MyListener = ComplexListener(PersonBias, CAT_D_B, CAT_D_B_CGPrior)
res = MyListener.JointInference(CAT_D_B_Utterance, Samples, False)
PrintCSV("CAT_D_B", res, False)
MyListener = ComplexListener(PersonBias, CAT_D_C, CAT_D_C_CGPrior)
res = MyListener.JointInference(CAT_D_C_Utterance, Samples, False)
PrintCSV("CAT_D_C", res, False)
MyListener = ComplexListener(PersonBias, CAT_D_D, CAT_D_D_CGPrior)
res = MyListener.JointInference(CAT_D_D_Utterance, Samples, False)
PrintCSV("CAT_D_D", res, False)
MyListener = ComplexListener(PersonBias, SIZE_A_A, SIZE_A_A_CGPrior)
res = MyListener.JointInference(SIZE_A_A_Utterance, Samples, False)
PrintCSV("SIZE_A_A", res, False)
MyListener = ComplexListener(PersonBias, SIZE_A_B, SIZE_A_B_CGPrior)
res = MyListener.JointInference(SIZE_A_B_Utterance, Samples, False)
PrintCSV("SIZE_A_B", res, False)
MyListener = ComplexListener(PersonBias, SIZE_A_C, SIZE_A_C_CGPrior)
res = MyListener.JointInference(SIZE_A_C_Utterance, Samples, False)
PrintCSV("SIZE_A_C", res, False)
MyListener = ComplexListener(PersonBias, SIZE_A_D, SIZE_A_D_CGPrior)
res = MyListener.JointInference(SIZE_A_D_Utterance, Samples, False)
PrintCSV("SIZE_A_D", res, False)
MyListener = ComplexListener(PersonBias, SIZE_B_A, SIZE_B_A_CGPrior)
res = MyListener.JointInference(SIZE_B_A_Utterance, Samples, False)
PrintCSV("SIZE_B_A", res, False)
MyListener = ComplexListener(PersonBias, SIZE_B_B, SIZE_B_B_CGPrior)
res = MyListener.JointInference(SIZE_B_B_Utterance, Samples, False)
PrintCSV("SIZE_B_B", res, False)
MyListener = ComplexListener(PersonBias, SIZE_B_C, SIZE_B_C_CGPrior)
res = MyListener.JointInference(SIZE_B_C_Utterance, Samples, False)
PrintCSV("SIZE_B_C", res, False)
MyListener = ComplexListener(PersonBias, SIZE_B_D, SIZE_B_D_CGPrior)
res = MyListener.JointInference(SIZE_B_D_Utterance, Samples, False)
PrintCSV("SIZE_B_D", res, False)
MyListener = ComplexListener(PersonBias, SIZE_C_A, SIZE_C_A_CGPrior)
res = MyListener.JointInference(SIZE_C_A_Utterance, Samples, False)
PrintCSV("SIZE_C_A", res, False)
MyListener = ComplexListener(PersonBias, SIZE_C_B, SIZE_C_B_CGPrior)
res = MyListener.JointInference(SIZE_C_B_Utterance, Samples, False)
PrintCSV("SIZE_C_B", res, False)
MyListener = ComplexListener(PersonBias, SIZE_C_C, SIZE_C_C_CGPrior)
res = MyListener.JointInference(SIZE_C_C_Utterance, Samples, False)
PrintCSV("SIZE_C_C", res, False)
MyListener = ComplexListener(PersonBias, SIZE_C_D, SIZE_C_D_CGPrior)
res = MyListener.JointInference(SIZE_C_D_Utterance, Samples, False)
PrintCSV("SIZE_C_D", res, False)
MyListener = ComplexListener(PersonBias, SIZE_D_A, SIZE_D_A_CGPrior)
res = MyListener.JointInference(SIZE_D_A_Utterance, Samples, False)
PrintCSV("SIZE_D_A", res, False)
MyListener = ComplexListener(PersonBias, SIZE_D_B, SIZE_D_B_CGPrior)
res = MyListener.JointInference(SIZE_D_B_Utterance, Samples, False)
PrintCSV("SIZE_D_B", res, False)
MyListener = ComplexListener(PersonBias, SIZE_D_C, SIZE_D_C_CGPrior)
res = MyListener.JointInference(SIZE_D_C_Utterance, Samples, False)
PrintCSV("SIZE_D_C", res, False)
MyListener = ComplexListener(PersonBias, SIZE_D_D, SIZE_D_D_CGPrior)
res = MyListener.JointInference(SIZE_D_D_Utterance, Samples, False)
PrintCSV("SIZE_D_D", res, False)
