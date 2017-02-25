import numpy as np


class Belief:

    """
    Simple class to store belief distributions
    """

    def __init__(self, HypothesisSize, names=None, prior=None):
        """
        Generate a belief distribution of size HypothesisSize.
        By default a uniform prior is built. But if a vector with priors is provided
        then that one is used.

        Args:
        HypothesisSize (int): Number of hypothesis
        names (list): Optional list of names of each hypothesis
        prior (list): optional list indicating the prior
        """
        self.HypothesisSize = HypothesisSize
        if prior is not None:
            if len(prior) != HypothesisSize:
                print "ERROR: Warning has a different size than hypothesis space size"
            else:
                self.probs = prior
        else:
            self.probs = [1.0/HypothesisSize] * HypothesisSize
        if names is not None:
            if len(names) != HypothesisSize:
                print "ERROR: List of names does not match hypothesis size."
            else:
                self.names = names
        else:
            self.names = ["Item "+str(x) for x in range(HypothesisSize)]

    def Normalize(self):
        Norm = sum(self.probs)
        self.probs = [x*1.0/Norm for x in self.probs]

    def Certain(self):
        """
        Return true if distribution has a 1 in one place and zero in all other places
        """
        if sum(self.probs) != 1:
            self.Normalize
        return True if sum([x == 1 for x in self.probs]) == 1 else False
