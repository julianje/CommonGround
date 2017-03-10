import itertools
import copy
import sys
import numpy as np


class HypothesisSpace:

    def __init__(self, hypotheses=None):
        self.hypotheses = hypotheses if hypotheses is not None else []
        self.expandedhypotheses = []

    def MoveForward(self):
        """
        Set the expanded hypothesis space as the core one and reset.
        """
        if self.expandedhypotheses != []:
            self.hypotheses = copy.deepcopy(self.expandedhypotheses)
            self.expandedhypotheses = []

    def Grow(self, VWIDs, SB, referentID, utterance, probability):
        """
        Expand the hypothesis space.
        This function finds all hypotheses with a given visual world and speaker bias
        and expands them to add a new entry with a new referent, utterance, and probability.
        It also deletes the old hypothesis.

        Args:
        VWIDs (List of VisualWorldIDs. Should be obtained from VisualWorld.GetIDs())
        SB an object of type Bias
        referentID
        """
        indices = self.LocateHypotheses(VWIDs, SB)
        # Extract those hypotheses from the hypothesis space.
        ExpansionHypotheses = [self.hypotheses[i] for i in indices]
        # Delete them from the hypothesis space
        # Move backwards so the indices stay the same as you remove the rightmost objects.
        # for i in range(len(indices)-1, -1, -1):
        #	del self.hypotheses[i]
        # Now expand each hypothesis and add it to the hypothesis range.
        for hypothesis in ExpansionHypotheses:
            ExpandedHypothesis = self.GrowHypothesis(
                hypothesis, referentID, utterance, probability)
            self.expandedhypotheses.append(ExpandedHypothesis)

    def LocateHypotheses(self, VW, SB):
        """
        Return the indices of all hypotheses with a given visual world and speaker bias.
        """
        indices = []
        for i in range(len(self.hypotheses)):
            if self.hypotheses[i].VisualWorldIDs == VW and self.hypotheses[i].SpeakerBiases == SB:
                indices.append(i)
        return indices

    def isEmpty(self):
        """
        Supporting function for ComplexListener
        """
        return True if self.hypotheses == [] else False

    def InsertHypothesis(self, hypothesis):
        """
        Add a new hypothesis.
        """
        self.hypotheses.append(hypothesis)

    def GrowHypothesis(self, hypothesis, referent, utterance, probability):
        """
        Take a hypothesis, duplicate it, and expand its referent, utterance, and probability.
        """
        newhypothesis = copy.deepcopy(hypothesis)
        newhypothesis.AddTrial(referent, utterance, probability)
        return newhypothesis

    def ComputeVWPosterior(self):
        VisualWorlds = []
        Beliefs = []
        for hypothesis in self.hypotheses:
            if hypothesis.VisualWorldIDs not in VisualWorlds:
                VisualWorlds.append(hypothesis.VisualWorldIDs)
                Beliefs.append(
                    np.prod(hypothesis.Likelihood)*hypothesis.SBprior*hypothesis.VWprior)
            else:
                index = VisualWorlds.index(hypothesis.VisualWorldIDs)
                Beliefs[
                    index] += np.prod(hypothesis.Likelihood)*hypothesis.SBprior*hypothesis.VWprior
        Norm = sum(Beliefs)
        return [VisualWorlds, [x/Norm for x in Beliefs]]

    def ComputeBiasPosterior(self):
        """
        Compute the posterior over speaker biases.
        """
        # First compute the probability of each bias value combination
        # and next compute the expected value.
        Biases = []
        Beliefs = []
        for hypothesis in self.hypotheses:
            if hypothesis.SpeakerBiases.BiasValue not in Biases:
                Biases.append(hypothesis.SpeakerBiases.BiasValue)
                Beliefs.append(
                    np.prod(hypothesis.Likelihood)*hypothesis.SBprior*hypothesis.VWprior)
            else:
                index = Biases.index(hypothesis.SpeakerBiases.BiasValue)
                Beliefs[
                    index] += np.prod(hypothesis.Likelihood)*hypothesis.SBprior*hypothesis.VWprior
        Norm = sum(Beliefs)
        Beliefs = [x/Norm for x in Beliefs]
        # Now compute the expected value.
        ExpVals = []
        # All hypotheses have the same dimensions over the number of speaker
        # biases the agent could have, so just take any.
        BiasTypes = self.hypotheses[0].SpeakerBiases.BiasType
        for BiasIndex in range(len(BiasTypes)):
            p = 0
            for HypothesisIndex in range(len(Biases)):
                p += Biases[HypothesisIndex][BiasIndex] * \
                    Beliefs[HypothesisIndex]
            ExpVals.append([BiasTypes[BiasIndex], p])
        return ExpVals

    def ComputeReferentPosterior(self, Refs):
        """
        Compute the posterior over referent vectors.
        Args:
        Refs (list of referent IDs)
        """
        # The hypothesis space consists of all vectors of referents.

        # for Referent in Refs:
        # Extract hypotheses that matter
        #    relevant = self.PullRefs(Referent)
