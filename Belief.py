import numpy as np


class Belief:

    """
    Simple class to store belief distributions
    """

    def __init__(self, HypothesisSize, prior=None):
        """
        Generate a belief distribution of size HypothesisSize.
        By default a uniform prior is built. But if a vector with priors is provided
        then that one is used.
        """
        if prior is not None:
            if len(prior) != HypothesisSize:
                print "ERROR: Warning has a different size than hypothesis space size"
            else:
                self.prior = prior
        else:
            self.prior = [1.0/HypothesisSize] * HypothesisSize

    def Normalize(self):
        Norm = sum(self.prior)
        self.prior = [x*1.0/Norm for x in self.prior]
