import copy
import Speaker
import sys
import ReferentBelief
import Bias
import SupportingFunctions as SF


class ComplexListener:

    def __init__(self, VisualWorld, CommonGroundPrior, BiasPriors):
        """
        VisualWorld is a VisualWorld object.
        CommonGroundPrior is a belief object.
        BiasPriors is a list of Belief objects that represent belief distributions over each kind
        of production bias that the speaker may have.

        Args:
        VisualWorld: Object of class VisualWorld
        CommonGroundPrior: Object of class Belief
        BiasPriors: List of objects of type belief
        """
        self.VisualWorld = VisualWorld
        self.CommonGroundPrior = CommonGroundPrior
        # This code assumes that the order of the Common ground prior
        # matches the order of the objects in the visual world. So just run a
        # sanity check
        if CommonGroundPrior.values != [obj.Id for obj in VisualWorld.objects]:
            print "ERROR: CommonGround prior object doesn't match the VisualWorld objects. Perhaps you used PhysicalObject.name instead of PhysicalObject.Id in the common ground prior constructor? Is the order in the visual world different from the order in the common ground priors?"
        self.BiasPriors = BiasPriors
        self.HypothesisSpace = []

    def ChangeVisualWorld(self, NewVisualWorld):
        """
        Change the visual world, and adjust
        self.CommonGroundPrior and self.HypothesisSpace
        to align with the new objects.
        """
        if len(self.VisualWorld.objects) != len(NewVisualWorld.objects):
            print "ERROR (CL): New visual world should have the same number of objects as current visual world."
            return None
        self.VisualWorld = NewVisualWorld
        self.CommonGroundPrior.values = [x.Id for x in NewVisualWorld.objects]
        # Now align the hypothesis space.

    def Infer(self, utterance, samples=1000):
        """
        Build a hypothesis space, where each hypothesis consists of a Speaker object
        together with a referent, and a prior.
        And then compute the prior and posterior given an utterance.

        Args:
        utterance: Object from Utterance class
        samples: Number of samples to use to estimate the probability of the utterance in each hypothesis space

        Returns:
        List of lists. Each sublist contains:
        [VisualWorld object, Speaker bias vector, testedobject (PhysicalObject), visual world prior, speaker bias prior, likelihood.]
        """
        self.HypothesisSpace = []
        # Each speaker object needs a visual world, and a bias object.
        # Build space of possible visual worlds
        #sys.stdout.write("Building visual world space.\n")
        [VW_HypothesisSpace, VW_Priors] = SF.BuildVWHypSpace(
            self.VisualWorld, self.CommonGroundPrior)
        # Now build hypothesis spaces over speaker biases.
        #sys.stdout.write("Building speaker bias space.\n")
        [SpeakerBias_HypothesisSpace, SpeakerBias_Priors] = SF.BuildBiasHypSpace(
            self.BiasPriors)
        # Next we need to combine these two to build a massive hypothesis
        # space.
        #sys.stdout.write("Building full hypothesis space.\n")
        BiasNames = [x.name for x in self.BiasPriors]
        for SB_index in range(len(SpeakerBias_Priors)):
            # Pack things into a set of Bias objects
            #sys.stdout.write("\tbuilding speaker biases.\n")
            CurrentBias = Bias.Bias(
                BiasNames, SpeakerBias_HypothesisSpace[SB_index])
            for VW_index in range(len(VW_Priors)):
                # Now we also need to build the space of possible referents.
                # First build the speaker object.
                #sys.stdout.write("\tbuilding speaker object.\n")
                TestSpeaker = Speaker.Speaker(
                    VW_HypothesisSpace[VW_index], CurrentBias)
                # Now iterate over the space of referents and get
                # the probability of producing the utterance.
                #sys.stdout.write("\t\titerating over referents.\n")
                for testobject in TestSpeaker.VisualWorld.objects:
                    # sys.stdout.write(".")
                    p = TestSpeaker.GetUtteranceProbability(
                        utterance, testobject, samples)
                    self.HypothesisSpace.append([VW_HypothesisSpace[VW_index], SpeakerBias_HypothesisSpace[
                                                SB_index], testobject, VW_Priors[VW_index], SpeakerBias_Priors[SB_index], p])

    def ComputePosterior(self, update=True):
        """
        Given the hypothesis space, compute the belief in speaker biases,
        visual access, and referent.

        args:
        update (bool): When set to True, the call also Updates the object's
        posterior about what is common ground, and about the speaker's production biases.
        """
        # First compute the belief in each target.
        # Kind of inefficient, but works.
        # Build a normalizing vector
        Posteriors = [x[3]*x[4]*x[5] for x in self.HypothesisSpace]
        Norm = sum(Posteriors)
        Posteriors = [x*1.0/Norm for x in Posteriors]
        CGBeliefs = []
        for cgobject in self.VisualWorld.objects:
            p = 0
            for HypothesisIndex in range(len(self.HypothesisSpace)):
                if self.HypothesisSpace[HypothesisIndex][0].Contains(cgobject):
                    p += Posteriors[HypothesisIndex]
            CGBeliefs.append([cgobject.Id, p])
        # Next compute the belief in each referent
        ReferentBeliefs = []
        for refobject in self.VisualWorld.objects:
            p = 0
            for HypothesisIndex in range(len(self.HypothesisSpace)):
                if self.HypothesisSpace[HypothesisIndex][2] == refobject:
                    p += Posteriors[HypothesisIndex]
            ReferentBeliefs.append([refobject.Id, p])
        # Next compute the posterior of each bias.
        BiasPosteriors = []
        # For each source of bias
        for i in range(len(self.BiasPriors)):
            # Iterate over each value
            Name = self.BiasPriors[i].name
            Domain = self.BiasPriors[i].values
            probs = []
            for DomainIndex in range(len(Domain)):
                CurrDomain = Domain[DomainIndex]
                p = 0
                for HypIndex in range(len(self.HypothesisSpace)):
                    if self.HypothesisSpace[HypIndex][1][i] == CurrDomain:
                        p += Posteriors[HypIndex]
                probs.append(p)
            BiasPosteriors.append([Name, Domain, probs])
        # Check if we need to update representations
        if update:
            # Update speaker biases.
            for BiasIndex in range(len(BiasPosteriors)):
                BiasUpdate = BiasPosteriors[BiasIndex]
                # BiasUpdate is a list where
                # BiasUpdate[0] is a string with the Id
                # BiasUpdate[1] is the domain and
                # BiasUpdate[2] is the posterior.
                # Check that the updater has the same name
                if BiasUpdate[0] != self.BiasPriors[BiasIndex].name:
                    print "WARNING: Bias names do not match. You probably have an error in the model specification."
                self.BiasPriors[BiasIndex].probs = BiasUpdate[2]
            # Update common ground beliefs.
            for ObjectCGIndex in range(len(CGBeliefs)):
                if self.CommonGroundPrior.values[ObjectCGIndex] != CGBeliefs[ObjectCGIndex][0]:
                    print "WARNING: Commong ground prior names do not match. You probably have an error in your model specification."
                self.CommonGroundPrior.probs[ObjectCGIndex] = CGBeliefs[ObjectCGIndex][1]
        return [CGBeliefs, ReferentBeliefs, BiasPosteriors]
