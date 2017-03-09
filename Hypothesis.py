import sys
import numpy as np


class Hypothesis:

    def __init__(self, VisualWorld, SpeakerBiases, Referent, Utterance, VWprior, SBprior, Likelihood):
        """
        Class to store a single hypothesis about a situation.
        The hypothesis only contains some summary information and
        not the full objects that are used to build it.

        Args:
        VisualWorld (VisualWorld object)
        SpeakerBiases (Bias object)
        Referent (PhysicalObject object)
        VWprior (float): prior over visual world.
        SBprior (float): prior over speaker bias.
        Likelihood (list of floats): likelihoods given the utterances.
        Utterance (list of Utterance objects): List of heard utterances.
        """
        self.VisualWorldIDs = VisualWorld.GetIDs()
        self.SpeakerBiases = SpeakerBiases
        self.ReferentID = Referent.Id
        self.VWprior = VWprior
        self.SBprior = SBprior
        self.Utterances = [Utterance]
        self.Likelihood = [Likelihood]

    def GetVWPosterior(self):
        """
        Return the posterior of the visual world.
        This function only returns a subcomponent of the posterior, as the
        actual posterior is obtained by adding all the hypotheses with the same visual world,
        integrating over possible referents.
        """
        return sum([self.VWprior*self.SBprior*x for x in self.Likelihood])

    def GetReferentPosterior(self):
        """
        Return the posterior of the referent using the last utterance.
        """
        return (self.VWprior*self.SBprior*self.Likelihood[-1])

    def GetBiasPosterior(self):
        """
        Return posterior of the bias value.
        """
        # This is the same as GetVWPosterior, but its meaning is different.
        return sum([self.VWprior*self.SBprior*x for x in self.Likelihood])

    def AddTrial(self, utterance, likelihood):
        """
        Integrate a new likelihood into the current likelihood
        and save the new utterance.
        """
        self.Utterances.append(utterance)
        self.Likelihood.append(likelihood)

    def PrintValues(self):
        """
        User friendly printout of internal structure.
        """
        sys.stdout.write(str(self.VisualWorldIDs)+","+str(self.ReferentID) +
                         ","+str(self.VWprior)+","+str(self.Likelihood)+"\n")
