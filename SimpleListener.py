class SimpleListener:

    def __init__(self, VisualWorld):
        """
        Build a model of a listener with a certain visual world.
        """
        self.VisualWorld = VisualWorld

    def RecoverWord(self, Utterance):
        """
        Given an utterance return a set of referents.
        The utterance is a list with a name and a feature (which might be empty).
        """
        Label = Utterance[0]
        Feature = Utterance[1]
        Results = []
        for CurrentObject in self.VisualWorld.objects:
            if Label == CurrentObject.name:
                if Feature is not None:
                    if Feature == CurrentObject.feature:
                        Results.append(CurrentObject)
                else:
                    Results.append(CurrentObject)
        return Results
