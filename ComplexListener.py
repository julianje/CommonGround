import copy
import Speaker
import sys
import Bias
import SupportingFunctions as SF
import Hypothesis
import HypothesisSpace as HS


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
        self.HypothesisSpace = HS.HypothesisSpace()

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
        return self.Infer_New(utterance, samples) if self.HypothesisSpace.isEmpty() else self.Infer_Repeat(utterance, samples)
        # return self.Infer_New(utterance, samples)

    def Infer_Repeat(self, utterance, samples=1000):
        """
        Take an existing hypothesis space and compute the posterior
        """
        [VW_HypothesisSpace, VW_Priors] = SF.BuildVWHypSpace(
            self.VisualWorld, self.CommonGroundPrior)
        [SpeakerBias_HypothesisSpace, SpeakerBias_Priors] = SF.BuildBiasHypSpace(
            self.BiasPriors)
        BiasNames = [x.name for x in self.BiasPriors]
        index = 0
        for SB_index in range(len(SpeakerBias_Priors)):
            CurrentBias = Bias.Bias(
                BiasNames, SpeakerBias_HypothesisSpace[SB_index])
            for VW_index in range(len(VW_Priors)):
                TestSpeaker = Speaker.Speaker(
                    VW_HypothesisSpace[VW_index], CurrentBias)
                for testobject in TestSpeaker.VisualWorld.objects:
                    p = TestSpeaker.GetUtteranceProbability(
                        utterance, testobject, samples)
                    self.HypothesisSpace.AddTrial(index, utterance, p)
                    index += 1
        # for HypothesisIndex in range(len(self.HypothesisSpace)):
        #    CurrentHypothesis = self.HypothesisSpace[HypothesisIndex]
        #    VW = CurrentHypothesis[0]  # Used
        #    SB = CurrentHypothesis[1]  # Used
        #    Ref = CurrentHypothesis[2]  # Used
        #    past_likelihood = CurrentHypothesis[5]
        #    if past_likelihood > 0:
        #        # build the model using the hypothesis
        #        BiasNames = [x.name for x in self.BiasPriors]
        #        CurrentBias = Bias.Bias(BiasNames, SB)
        #        TestSpeaker = Speaker.Speaker(VW, CurrentBias)
        #        p = TestSpeaker.GetUtteranceProbability(utterance, Ref, samples)
        #        sys.stdout.write(str(past_likelihood)+"*"+str(p)+" ("+str(CurrentHypothesis[2].Id)+")\n")
        #        self.HypothesisSpace[HypothesisIndex][5] = past_likelihood*p

    def Infer_New(self, utterance, samples=1000):
        """
        Build a hypothesis space and compute the posterior
        """
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
                    self.HypothesisSpace.InsertHypothesis(Hypothesis.Hypothesis(VW_HypothesisSpace[VW_index], CurrentBias, testobject, utterance, VW_Priors[VW_index], SpeakerBias_Priors[SB_index], p))

    def ComputePosterior(self, update=False):
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
        [VW_HypothesisSpace, VW_Priors] = SF.BuildVWHypSpace(
            self.VisualWorld, self.CommonGroundPrior)
        return self.HypothesisSpace.ComputeVWPosterior(VW_HypothesisSpace)
