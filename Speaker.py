import SimpleListener
import random
import Utterance


class Speaker:

    def __init__(self, VisualWorld, Bias, rationalitynoise=0.1):
        """
        Create a new speaker with a visual world, a set of biases, and some rationality noise.

        Args:
        VisualWorld: Visualworld object type
        Bias: object of Bias type
        rationalitynoise (float): parameter showing amount of noise in rationality.
        Rationality nosie introduced a small probability that the speaker will
        fail to be overly specific even when there is reason to be.
        """
        self.VisualWorld = VisualWorld
        self.Bias = Bias
        self.rationalitynoise = rationalitynoise

    def GetUtteranceProbability(self, utterance, target, samples=1000):
        """
        Get the probability of an utterance by drawing a set of samples.

        Args:
        utterance: object of type utterance.
        target: PhysicalObject object type.
        samples: number of samples to draw.
        """
        utterances = [self.Communicate(target) for x in range(samples)]
        hits = sum([x == utterance for x in utterances])
        return hits*1.0/samples

    def SampleUtterance(self, target):
        """
        Return a baseline utterance relying on the bias.

        Args:
        target: PhysicalObject type of object.
        """
        # build a basic utterance
        SampledUtterance = Utterance.Utterance(target.name)
        # Now iterate over each feature, and sample a probability of using it.
        for currentfeature in target.features:
            basebias = self.Bias.GetBias(currentfeature)
            if random.random() < basebias:
                SampledUtterance.InsertFeature(currentfeature)
        return SampledUtterance

    def Communicate(self, target, giveup = 500):
        """
        Return a probability distribution over possible utterances.
        In this simple case, the only thing at stake is whether
        the feature is used or not.
        """
        if not self.VisualWorld.Contains(target):
            print "Error: communicating something I do not see or that does not exist."
        # Create a model of the listener
        ImaginedListener = SimpleListener.SimpleListener(self.VisualWorld)
        # Try the simplest utterance first but with
        # a baseline bias for producing a more complex utterance.
        Utterance = self.SampleUtterance(target)
        if random.random() < self.rationalitynoise:
            return Utterance
        InferredBelief = ImaginedListener.InferReferent(Utterance)
        # Loop here until you find a suitable utterance
        tries = 0
        while(not InferredBelief.Certain()):
            tries = tries + 1
            if tries >= giveup:
                return Utterance
            Utterance = self.SampleUtterance(target)
            InferredBelief = ImaginedListener.InferReferent(Utterance)
        return Utterance

    def __eq__(self, other):
        """
        check if two objects are identical
        """
        return self.__dict__ == other.__dict__
