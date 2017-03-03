import Belief


class VisualWorld:

    def __init__(self, objects):
        """
        Save the set of objects (from the object class) in the visual world.
        """
        self.objects = objects

    def Contains(self, target):
        # Check that the target has the features.
        return 1 if sum([x == target for x in self.objects]) > 0 else 0

    def GetIndex(self, target):
        """
        Get index of a target in vector list.
        This is helpful because sometimes you want to
        identify an object by using an isomoprhic object
        from a different visual world object. In those cases
        direct comparisons don't work
        """
        match = [(x.name == target.name and x.feature == target.feature) for x in self.objects]
        return match.index(1)

    def Delete(self, target):
        """
        remove target from list of objects
        """
        if self.Contains(target):
            self.objects.pop(self.GetIndex(target))
        else:
            print "Nothing in list."

    def GetIDs(self):
        """
        return a list of object Ids, or basic name when Ids not available.
        """
        return [x.GetID() for x in self.objects]

    def BuildPrior(self):
        """
        Build a probability distribution over the objects in the visual world and return Belief object
        """
        return Belief.Belief(len(self.objects), self.GetIDs())
