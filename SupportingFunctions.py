import copy
import numpy as np


def BuildVisualWorldHypothesisSpace(priors):
    """
    Build a list of lists where each list has boolean markers indicating whether
    the object should be there. The function omits the empty set.

    Args:
    prior: Object of class belief. Each entry in the belief object is a prior over a visual world.
    """
    # Build the first entry:
    size = priors.HypothesisSize
    SubsetIndex = [[0] * size]
    SubsetIndex[0][0] = 1
    CurrentEntry = SubsetIndex[0]
    # First adjust the prior for the first entry
    SubsetPriors = []
    SubsetPriors.append(np.prod([(priors.probs[x]*CurrentEntry[x])+((1-priors.probs[x])*np.abs(1-CurrentEntry[x])) for x in range(size)]))
    while(CurrentEntry != [1] * size):
        NextEntry = copy.deepcopy(CurrentEntry)
        Index = 0
        while(NextEntry[Index] == 1):
            NextEntry[Index] = 0
            Index += 1
        NextEntry[Index] = 1
        SubsetIndex.append(NextEntry)
        CurrentEntry = NextEntry
        # Add the new prior
        SubsetPriors.append(np.prod([(priors.probs[x]*CurrentEntry[x])+((1-priors.probs[x])*np.abs(1-CurrentEntry[x])) for x in range(size)]))
    # Since we are not considering the hypothesis where the speaker cannot see anything, we
    # need to renormalize the prior.
    Norm = sum(SubsetPriors)
    SubsetPriors = [x*1.0/Norm for x in SubsetPriors]
    return [SubsetIndex, SubsetPriors]
