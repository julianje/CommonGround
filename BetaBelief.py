import numpy as np
import scipy.stats as st


class BetaBelief:

    def __init__(self, alpha, beta):
        """
        Store information about a beta distribution
        """
        self.alpha = alpha
        self.beta = beta

    def Update(self, outcome):
        """
        Update the prior relying on conjugates.
        """
        if outcome == 1:
            self.alpha += 1
        else:
            self.beta += 1

    def Sample(self):
        """
        Draw a sample.
        """
        return np.random.beta(self.alpha, self.beta)

    def Prob(self, value):
        """
        Return probability of value.
        """
        return st.beta.pdf(value, self.alpha, self.beta)

