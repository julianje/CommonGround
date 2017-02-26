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
        if CommonGroundPrior.values != [obj.name for obj in VisualWorld.objects]:
            print "ERROR: CommonGround prior object doesn't match the VisualWorld objects (order matters)."
        self.BiasPriors = BiasPriors
        self.HypothesisSpace = []

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
        [VW_HypothesisSpace, VW_Priors] = SF.BuildVWHypSpace(
            self.VisualWorld, self.CommonGroundPrior)
        # Now build hypothesis spaces over speaker biases.
        [SpeakerBias_HypothesisSpace, SpeakerBias_Priors] = SF.BuildBiasHypSpace(
            self.BiasPriors)
        # Next we need to combine these two to build a massive hypothesis
        # space.
        BiasNames = [x.name for x in self.BiasPriors]
        for SB_index in range(len(SpeakerBias_Priors)):
            # Pack things into a set of Bias objects
            CurrentBias = Bias.Bias(
                BiasNames, SpeakerBias_HypothesisSpace[SB_index])
            for VW_index in range(len(VW_Priors)):
                # Now we also need to build the space of possible referents.
                # First build the speaker object.
                TestSpeaker = Speaker.Speaker(
                    VW_HypothesisSpace[VW_index], CurrentBias)
                # Now iterate over the space of referents and get
                # the probability of producing the utterance.
                for testobject in TestSpeaker.VisualWorld.objects:
                    p = TestSpeaker.GetUtteranceProbability(
                        utterance, testobject, samples)

                    self.HypothesisSpace.append([VW_HypothesisSpace[VW_index], SpeakerBias_HypothesisSpace[
                                                SB_index], testobject, VW_Priors[VW_index], SpeakerBias_Priors[SB_index], p])

    def ComputePosterior(self):
        """
        Given the hypothesis space, compute the belief in speaker biases,
        visual access, and referent.
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
        return [CGBeliefs, ReferentBeliefs, BiasPosteriors]
