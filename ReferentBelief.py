import VisualWorld


class ReferentBelief:

    """
    Store a visual world with a probability distribution.
    """

    def __init__(self, VisualWorld):
        self.VisualWorld = VisualWorld
        self.objectcounters = [0]*len(VisualWorld.objects)
        self.featurecounters = [0]*len(VisualWorld.objects)
        self.prob = [1]*len(VisualWorld.objects)
        norm = sum(self.prob)
        self.prob = [x*1.0/norm for x in self.prob]

    def insert(self, target, utterance):
        """
        Add a counter on an observation.
        target is an object
        objectname is the object's label
        and feature determines if the feature was used.
        """
        objectname = utterance[0]
        feature = utterance[1]
        objectindex = self.VisualWorld.objects.index(target)
        # For now, we'll just assume that the speaker
        # does not make errors. So the target is always right.
        self.objectcounters[objectindex] += 1
        if feature is not None:
            self.featurecounters[objectindex] += 1

    def ComputeProbability(self, utterance):
        """
        Using the samples from the insert method,
        compute the probability of a given target, given the utterance
        """
        for i in range(len(self.VisualWorld.objects)):
            objecttest = self.VisualWorld.objects[i]
            if objecttest.name == utterance[0]:
                p = self.featurecounters[i]*1.0/self.objectcounters[i]
                if utterance[1] is None:
                    self.prob[i] = 1-p
                else:
                    if utterance[1] == objecttest.feature:
                        self.prob[i] = p
                    else:
                        self.prob[i] = 0
            else:
                self.prob[i] = 0
