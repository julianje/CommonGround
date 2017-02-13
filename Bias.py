class Bias:

    def __init__(self, BiasType, BiasValue):
        """
        Store a list of features and the
        baseline bias to produce them.
        """
        self.BiasType = BiasType
        self.BiasValue = BiasValue

    def GetBias(self, Type):
        """
        Return the bias for a given feature.
        Returns 0 bias if feature does not exist.
        """
        if Type in self.BiasType:
            return self.BiasValue[self.BiasType.index(Type)]
        else:
            return 0
