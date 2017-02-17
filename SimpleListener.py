class SimpleListener:

    def __init__(self, VisualWorld):
        """
        Build a model of a listener with a certain visual world.
        """
        self.VisualWorld = VisualWorld

    def InferReferent(self, Utterance, ReturnIDs=False):
        """
        Given an utterance return a set of referents.
        The utterance is a list with a name and a feature (which might be empty).
        When return IDs is true return a list with beli
        """
        results = [self.TestObject(x, Utterance) for x in self.VisualWorld.objects]
        Norm = sum(results)
        if not ReturnIDs:
            return [x*1.0/Norm for x in results]
        else:
            return [[x*1.0/Norm for x in results], self.VisualWorld.GetIDs()]

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
