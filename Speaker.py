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

    def Communicate(self, target):
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
        Inferred = ImaginedListener.RecoverWord(Utterance)
        if len(Inferred) == 1:
            return Utterance
        else:
            if random.random() < self.rationalitynoise:
                return [target.name, None]
            else:
                return [target.name, target.feature]
