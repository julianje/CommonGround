class ObjectFeature:

    def __init__(self, featurename, featuretype):
        """
        Simple class to store a feature name and a feature type.

        Args:

        featurename (str): the name of the feature (e.g. 'red')
        featuretype (str): the type of feature (e.g. 'color')
        """
        self.feature = featurename
        self.type = featuretype

    def __eq__(self, other):
        """
        check if two objects are identical
        """
        return self.__dict__ == other.__dict__
