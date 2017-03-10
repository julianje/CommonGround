class Filter:

    def __init__(self, observable=None):
        """
        Class to store logical constraints
        over the kinds of hypotheses that are allowed.

        This class can be expanded to handle arbitrarily complex
        constraints, as long as the rest of the code can interface
        using the Check method.
        """
        self.observable = observable

    def Check(self, HypothesisSpace, Priors):
        """
        Take two lists, one with a list of hypotheses about the visual world,
        and one with the priors.
        Return a subset depending on some constraint.
        """
        if self.observable is None:
            return [HypothesisSpace, Priors]
        FilteredSpace = []
        FilteredPriors = []
        for i in range(len(HypothesisSpace)):
            if len(HypothesisSpace[i].objects) == self.observable:
                FilteredSpace.append(HypothesisSpace[i])
                FilteredPriors.append(Priors[i])
        # Re-normalize filtered priors
        Norm = sum(FilteredPriors)
        FilteredPriors = [x/Norm for x in FilteredPriors]
        return [FilteredSpace, FilteredPriors]
