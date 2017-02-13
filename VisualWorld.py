import PhysicalObject


class VisualWorld:

    def __init__(self, objects):
        """
        Save the set of objects (from the object class) in the visual world.
        """
        self.objects = objects

    def Contains(self, target):
        res = 1 if sum([x == target for x in self.objects]) > 0 else 0
        return res
