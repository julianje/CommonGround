import ObjectFeature as OF


class Utterance:

    def __init__(self, name, features=[]):
        """
        Build a simple class to capture utterances.

        args:
        name (str): object name
        features (list): list of ObjectFeatures that were used.
        """
        self.name = name
        # Make sure you got a list of object features.
        if sum([isinstance(x, OF.ObjectFeature) for x in features]) != len(features):
            print "PhysicalObject error. second argument is not a list of ObjectFeature objects."
            self.features = []
        else:
            self.features = features

    def __eq__(self, other):
        """
        check if two objects are identical
        """
        return self.__dict__ == other.__dict__
