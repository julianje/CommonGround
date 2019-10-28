class Filter:

    def __init__(self, minval=None, maxval=None):
        """
        Class to store logical constraints
        over the kinds of hypotheses that are allowed.

        This class can be expanded to handle arbitrarily complex
        constraints, as long as the rest of the code can interface
        using the Check method.
        """
        if minval is None:
            self.minval = -1
        else:
            self.minval = minval
        if maxval is None:
            # some arbitrarily large number
            # this will be a problem if the number of hypothese is larger
            self.maxval = 50000
        else:
            self.maxval = maxval

    def Test(self, hypothesis):
        """
        Test a hypothesis in the filter
        """
        if len(hypothesis.objects) >= self.minval and len(hypothesis.objects) <= self.maxval:
            return True
        else:
            return False

    def Check(self, HypothesisSpace, Priors):
        """
        Take two lists, one with a list of hypotheses about the visual world,
        and one with the priors.
        Return a subset depending on some constraint.
        """
        if self.minval is None and self.maxval is None:
            return [HypothesisSpace, Priors]
        FilteredSpace = []
        FilteredPriors = []
        for i in range(len(HypothesisSpace)):
            if self.Test(HypothesisSpace[i]):
                FilteredSpace.append(HypothesisSpace[i])
                FilteredPriors.append(Priors[i])
        # Re-normalize filtered priors
        Norm = sum(FilteredPriors)
        FilteredPriors = [x/Norm for x in FilteredPriors]
        return [FilteredSpace, FilteredPriors]
