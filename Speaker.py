import SimpleListener
import random


class Speaker:

    def __init__(self, VisualWorld, Bias, rationalitynoise=0.1):
        self.VisualWorld = VisualWorld
        self.Bias = Bias
        # rationality noise
        # introduces a small probability
        # that the speaker will fail to
        # be overly specific even when she knows that she should.
        self.rationalitynoise = 0.1

    def SampleUtterance(self, target):
        """
        Return a baseline utterance relying on the bias.
        """
        basebias = self.Bias.BiasValue[
            self.Bias.BiasType.index(target.featuretype)]
        if random.random() < basebias:
            return [target.name, target.feature]
        else:
            return [target.name, None]

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
