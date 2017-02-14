import copy
import Speaker
import sys
import VisualWorld
import ReferentBelief


class ComplexListener:

    def __init__(self, Bias, VisualWorld, CommonGroundPrior):
        """
        Bias and VisualWorld are objects of that type.
        CommonGroundPrior is a list of the same length as
        the number of objects in the visual world and they
        indicate the probability that the object is
        common ground.
        """
        self.Bias = Bias
        self.VisualWorld = VisualWorld
        # Sanity check that CommonGroundPrior makes sense
        if len(CommonGroundPrior) != len(self.VisualWorld.objects):
            print "Error: Prior doesn't match size of visual world"
        else:
            self.CommonGroundPrior = CommonGroundPrior

    def JointInference(self, utterance, samples=100, verbose=True):
        """
        Given an utterance jointly infer what object the speaker
        was talking about, and their knowledge. To do this, integrate
        over all the possible visual worlds the agent could have, times
        the probability of the utterance, and weight by the prior.

        utterance should be a list of two items where the first entry determines
        the speaker's base noun and the second utterance the extra information (if any)

        samples indicated how many samples to draw from each utterance.
        """
        # For now, model only supports priors that are either 0 or 0.5.
        # Create a model of the speaker's visual world
        SpeakerVisualWorld = copy.deepcopy(self.VisualWorld)
        # Create a model of the speaker.
        ImaginedSpeaker = Speaker.Speaker(SpeakerVisualWorld, self.Bias)
        # Now, for each target in the visual world, get what the speaker would
        # have said.
        # print "Testing full visual world...\n"
        # Build a referent belief to store the probability of the referrent.
        FullKnowledgeBeliefs = ReferentBelief.ReferentBelief(
            SpeakerVisualWorld)
        for TestReferent in ImaginedSpeaker.VisualWorld.objects:
            # Run a large set of imagined speakers.
            for sample in range(samples):
                FullKnowledgeBeliefs.insert(
                    TestReferent, ImaginedSpeaker.Communicate(TestReferent))
        # Assuming that there is only one place where CommonGroundPrior
        # is different from 0. Delete it.
        ObjectIndex = self.CommonGroundPrior.index(0.5)
        SimpleVisualWorld = copy.deepcopy(SpeakerVisualWorld)
        # sys.stdout.write(
        #    "Deleting object "+str(SpeakerVisualWorld.objects[ObjectIndex]))
        # sys.stdout.write("\n"+str(SimpleVisualWorld.objects))
        # This deletion will return the deleted object
        # but we don't need it
        Discard = SimpleVisualWorld.Delete(
            SpeakerVisualWorld.objects[ObjectIndex])
        # print SimpleVisualWorld.objects
        ImaginedSpeaker = Speaker.Speaker(SimpleVisualWorld, self.Bias)
        # print "Testing simple visual world...\n"
        PartialKnowledgeBeliefs = ReferentBelief.ReferentBelief(
            SimpleVisualWorld)
        for TestReferent in ImaginedSpeaker.VisualWorld.objects:
            for sample in range(samples):
                PartialKnowledgeBeliefs.insert(
                    TestReferent, ImaginedSpeaker.Communicate(TestReferent))
        # Now we can compute the probability of each target jointly with
        # knowledge or ignorance.
        FullKnowledgeBeliefs.ComputeProbability(utterance)
        PartialKnowledgeBeliefs.ComputeProbability(utterance)
        # return [FullKnowledgeBeliefs, PartialKnowledgeBeliefs]
        # Compute probability that the speaker had full knowledge:
        ProbKnowledge = sum(FullKnowledgeBeliefs.prob)*1.0 / \
            (sum(FullKnowledgeBeliefs.prob)+sum(PartialKnowledgeBeliefs.prob))
        # Make a guess about which object the agent was talking about.
        referent = []
        feature = []
        probability = []
        for i in range(len(FullKnowledgeBeliefs.VisualWorld.objects)):
            referent.append(FullKnowledgeBeliefs.VisualWorld.objects[i].name)
            feature.append(FullKnowledgeBeliefs.VisualWorld.objects[i].feature)
            probability.append(FullKnowledgeBeliefs.prob[i])
            if PartialKnowledgeBeliefs.VisualWorld.Contains(FullKnowledgeBeliefs.VisualWorld.objects[i]):
                tempindex = PartialKnowledgeBeliefs.VisualWorld.GetIndex(
                    FullKnowledgeBeliefs.VisualWorld.objects[i])
                probability[i] += PartialKnowledgeBeliefs.prob[tempindex]
        norm = sum(probability)
        if norm == 0:
            print "Error! All probabilities are zero :("
        probability = [x*1.0/norm for x in probability]
        if verbose:
            sys.stdout.write(
                "\nProbability of common ground: "+str(ProbKnowledge)+"\n")
            sys.stdout.write("Inferred referent:\n")
            sys.stdout.write(
                str(referent)+"\n"+str(feature)+"\n"+str(probability))
        return [ProbKnowledge, referent, feature, probability, FullKnowledgeBeliefs, PartialKnowledgeBeliefs]
