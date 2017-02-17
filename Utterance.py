import ObjectFeature as OF
import sys


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

    def InsertFeature(self, feature):
        """
        Add a new feature
        """
        self.features.append(feature)

    def Print(self):
        """
        Print the utterance
        """
        for feature in self.features:
            sys.stdout.write(str(feature.feature)+" ")
        sys.stdout.write(str(self.name)+"\n")

    def __eq__(self, other):
        """
        check if two objects are identical
        """
        return self.__dict__ == other.__dict__