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
        	if hypothesis.

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

    def PullRefs(self, Referent):
        """
        Retrieve all hypotheses that have the target referent,
        split by
        Referent (target referent ID)
        """
        Returnset = []
        for hypothesis in self.hypotheses:
            if hypothesis.ReferentID == Referent:
                Returnset.append(hypothesis)
        return Returnset

    def PullVW(self, VW):
        """
        Retrieve all hypotheses with a given visual world, split by speaker bias objects.
        VW (List of object names; should be retrieved from VisualWorld.GetIDs())
        """
        Returnset = []
        for hypothesis in self.hypotheses:
            if hypothesis.VisualWorldIDs == VW:
                Returnset.append(hypothesis)
        # Now split Returnset based on the SpeakerBiases.
        # First create the space of speaker biases in the return set.
        SpeakerBiasSet = []
        for hypothesis in Returnset:
            if hypothesis.SpeakerBiases not in SpeakerBiasSet:
                SpeakerBiasSet.append(hypothesis.SpeakerBiases)
        # Now create the final return set.
        FinalReturnSet = []
        for SpeakerBias in SpeakerBiasSet:
            Hypothesissubset = []
            for hypothesis in Returnset:
                if hypothesis.SpeakerBiases == SpeakerBias:
                    Hypothesissubset.append(hypothesis)
            FinalReturnSet.append(Hypothesissubset)
        return FinalReturnSet
