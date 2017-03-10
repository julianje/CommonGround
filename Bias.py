class Bias:

    def __init__(self, BiasType, BiasValue):
        """
        Store a list of features and the
        baseline bias to produce them.
        """
        self.BiasType = BiasType
        self.BiasValue = BiasValue

    def GetBias(self, feature):
        """
        Return the bias for a given feature.
        Returns 0 bias if feature does not exist.
        Feature is an ObjectFeature object type or a string.
        """
        if isinstance(feature, str):
            if feature in self.BiasType:
                return self.BiasValue[self.BiasType.index(feature)]
            else:
                return 0
        else:
            if feature.type in self.BiasType:
                return self.BiasValue[self.BiasType.index(feature.type)]
            else:
                return 0

    def __eq__(self, other):
        """
        check if two objects are identical
        """
        return self.__dict__ == other.__dict__
