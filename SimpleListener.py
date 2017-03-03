import Belief


class SimpleListener:

    def __init__(self, VisualWorld):
        """
        Build a model of a listener with a certain visual world.
        """
        self.VisualWorld = VisualWorld

    def InferReferent(self, Utterance):
        """
        Given an utterance return a set of referents.
        The utterance is a list with a name and a feature (which might be empty).
        """
        results = [self.TestObject(x, Utterance)
                   for x in self.VisualWorld.objects]
        # Create a belief object with the results
        results = Belief.Belief(
            len(results), self.VisualWorld.GetIDs(), results)
        results.Normalize()
        return results

    def TestObject(self, referent, utterance):
        """
        Test if an utterance is consistent with a hypothesized referent
        """
        # First check if the basic noun works
        if referent.name != utterance.name:
            return 0
        else:
            # Check if all the features the agent added match the object.
            res = [referent.Matches(x) for x in utterance.features]
            return 0 if sum(res) < len(utterance.features) else 1
