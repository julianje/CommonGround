import copy
import numpy as np
import VisualWorld as VW
import Belief
from itertools import compress
from itertools import product
import scipy.stats as st
import sys


def BuildVWHypSpace(VisualWorld, CGprior, Filter=None):
    """
    Build a hypothesis space of visualworlds
    (by building all non-empty subsets of the listeners visual world; including the full set)
    together with priors.

    Args:
    VisualWorld: Object of type VisualWorld.
    CGprior: Object of class belief. Each entry in the belief object is a prior over a visual world.
    Filter: Object of class Filter. When different from none, hypothesis generator
    reduces the hypothesis filter using the Filter.Check() method.
    """
    # Build the first entry:
    size = CGprior.HypothesisSize
    SubsetIndex = [[False] * size]
    SubsetIndex[0][0] = True
    CurrentEntry = SubsetIndex[0]
    # First adjust the prior for the first entry
    SubsetPriors = []
    SubsetPriors.append(np.prod([(CGprior.probs[
                        x]*CurrentEntry[x])+((1-CGprior.probs[x])*np.abs(1-CurrentEntry[x])) for x in range(size)]))
    while(CurrentEntry != [True] * size):
        NextEntry = copy.deepcopy(CurrentEntry)
        Index = 0
        while(NextEntry[Index] is True):
            NextEntry[Index] = False
            Index += 1
        NextEntry[Index] = True
        SubsetIndex.append(NextEntry)
        CurrentEntry = NextEntry
        # Add the new prior
        SubsetPriors.append(np.prod([(CGprior.probs[
                            x]*CurrentEntry[x])+((1-CGprior.probs[x])*np.abs(1-CurrentEntry[x])) for x in range(size)]))
    # Since we are not considering the hypothesis where the speaker cannot see anything, we
    # need to renormalize the prior.
    Norm = sum(SubsetPriors)
    SubsetPriors = [x*1.0/Norm for x in SubsetPriors]
    # Now use the SubsetIndex to build a set of visual worlds.
    VWHypothesisSpace = [VW.VisualWorld(
        list(compress(VisualWorld.objects, Indices))) for Indices in SubsetIndex]
    if Filter is None:
        return [VWHypothesisSpace, SubsetPriors]
    else:
        return Filter.Check(VWHypothesisSpace, SubsetPriors)


def BuildBiasHypSpace(BiasPriors):
    """
    Build a hypothesis space of the speaker's biases over different features.

    Args:
    BiasPriors: a list of Belief objects.
    """
    # Compute the hypothesis space size.
    # Retrieve hypothesis space lists.
    BiasValues = [x.values for x in BiasPriors]
    result = product(*BiasValues)
    result = [list(x) for x in result]
    # Now get the probabilities
    BiasProbs = [x.probs for x in BiasPriors]
    probs = product(*BiasProbs)
    probs = [np.prod(list(x)) for x in probs]
    return [result, probs]


def BuildBeta(alpha, beta, name=None, granularity=0.1):
    """
    Pack a beta distribution into a Belief object
    """
    values = list(np.arange(granularity, 1, granularity))
    HypothesisSize = len(values)
    prior = st.beta.pdf(values, alpha, beta)
    distribution = Belief.Belief(HypothesisSize, values, prior, name)
    distribution.Normalize()
    return distribution


def PrintCSV(results, Trial=None, header=True):
    """
    Take a result list obtained from ComplexListener.ComputePosterior() and
    use it to print a csv file.
    """
    # Print header
    if header:
        sys.stdout.write("Trial,Variable,Type,Value\n")
    # Now start with common ground inferences:
    for i in range(len(results[0][0])):
        sys.stdout.write(
            str(Trial)+","+str(results[0][0][i])+",VW,"+str(results[0][1][i])+"\n")
    # Now start with common ground inferences:
    for i in range(len(results[1][0])):
        sys.stdout.write(
            str(Trial)+","+str(results[1][0][i])+",CG,"+str(results[1][1][i])+"\n")
    # Now print the referent beliefs
    for Referent in results[2]:
        sys.stdout.write(
            str(Trial)+","+str(Referent[0])+",ProductionBias,"+str(Referent[1])+"\n")
    # Now print expected value over production biases.
    for i in range(len(results[3][0])):
        sys.stdout.write(
            str(Trial)+","+str(results[3][0][i])+",Referent,"+str(results[3][1][i])+"\n")
