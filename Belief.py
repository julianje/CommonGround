

class Belief:

    """
    Simple class to store belief distributions
    """

    def __init__(self, HypothesisSize, values=None, prior=None, name=None):
        """
        Generate a belief distribution of size HypothesisSize.
        By default a uniform prior is built. But if a vector with priors is provided
        then that one is used.

        Args:
        HypothesisSize (int): Number of hypothesis
        values (list): Optional list of values of each hypothesis.
        prior (list): optional list indicating the prior
        name (str): name that can be used to identify its content.
        """
        self.HypothesisSize = HypothesisSize
        self.name = name
        if prior is not None:
            if len(prior) != HypothesisSize:
                print "ERROR: Warning, prior has a different size than hypothesis space size"
            else:
                self.probs = prior
        else:
            self.probs = [1.0/HypothesisSize] * HypothesisSize
        if values is not None:
            if len(values) != HypothesisSize:
                print "ERROR: List of values does not match hypothesis size."
            else:
                self.values = values
        else:
            self.values = range(HypothesisSize)

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
