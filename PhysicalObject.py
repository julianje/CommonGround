import ObjectFeature as OF


class PhysicalObject:

    def __init__(self, name, features, Id=None):
        """
        Save an object's name together with a collection of features.

        Args:

        name (str): object's name at the basic level
        features (list): list of objects of type ObjectFeature
        Id (str): Id name for internal use.
        """
        self.name = name
        # Make sure you got a list of object features.
        if sum([isinstance(x, OF.ObjectFeature) for x in features]) != len(features):
            print "PhysicalObject error. second argument is not a list of ObjectFeature objects."
            self.features = None
        else:
            self.features = features
        self.Id = Id

    def Matches(self, feature):
        """
        Check if object has certain feature
        """
        return 1 if feature in self.features else 0

    def GetID(self, IDonly=False):
        """
        Return list of IDs. When IDonly is false, function returns names when Ids aren't available.
        """
        if self.Id is not None:
            return self.Id
        else:
            if IDonly:
                return None
            else:
                return self.name

    def __eq__(self, other):
        """
        check if two objects are identical
        """
        return self.__dict__ == other.__dict__
