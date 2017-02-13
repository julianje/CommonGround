import SimpleListener
import random


class Speaker:

    def __init__(self, VisualWorld, Bias):
        self.VisualWorld = VisualWorld
        self.Bias = Bias

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
            return [target.name, target.feature]
