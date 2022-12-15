import numpy as np
from scipy.stats import bernoulli

class ConfidenceBound:

    def __init__(self, cb_name, params):
        self.name = cb_name

        self.epsilon = params[0]
        self.delta = params[1]

        if cb_name == "Hoeffding" or cb_name == "HoeffdingAbs":
            # Pr (|x - xt| >= sqrt((b - a)^2/(2n) * ln (2/delta)) <= delta
            self.a = params[2]
            self.b = params[3]
            self.Delta_bar = params[4]
        elif cb_name == "Laplace":
            # Assumption: bounded in [0, 1]
            self.Delta_bar = params[2]

        # Dynamic filling Table
        self.maxim = 30000
        self.table = np.zeros([self.maxim])
        self.table[0] = np.infty


        for i in range(1, self.maxim):
            self.table[i] = self.get_dynamic_bound(i)
        return

    def get_bound(self, n_observatios):
        return self.table[n_observatios]

    def get_dynamic_bound(self, n_observations, time=-1):
        if self.name == "Hoeffding":
            return self.get_Hoeffding_bound(n_observations)
        elif self.name == "HoeffdingAbs":
            return self.get_HoeffdingAbs_bound(n_observations)
        elif self.name == "Laplace":
            return self.get_Laplace_bound(n_observations)
        elif self.name == "lemma1":
            return self.get_lemma1_bound(n_observations)

    def get_lemma1_bound(self, n_observations):
        return 0.5*np.sqrt(2.0/n_observations*(1+1.0/n_observations)*np.log(np.sqrt(n_observations+1)/self.delta))

    def get_Hoeffding_bound(self, n_observations):
        return np.sqrt((self.b - self.a) ** 2 / (2 * n_observations) * np.log(1 / self.delta(n_observations)))

    def get_HoeffdingAbs_bound(self, n_observations):
        return np.sqrt((self.b - self.a) ** 2 / (2 * n_observations) * np.log(2 / self.delta(n_observations)))

    def get_Laplace_bound(self, n_observation):
        return np.sqrt((1 + 1.0/n_observation)*np.log(np.sqrt(n_observation+1)/self.delta(n_observation))/(2*n_observation))

    def get_Laplace_bound_old(self, n_observation):
        # Delta = 1/(time^3)
        n_observations = np.zeros([1, n_observation]) + range(1, (n_observation+1))
        return 2 * self.sigma * np.sum(np.sqrt((1 + 1.0 / n_observations) * np.log(2*np.sqrt(n_observations + 1)/self.delta(n_observations))))

# This is hard to find out for different bounds! Not always possible.
    # f^{-1}(Delta) as defined in the paper
    def get_number(self, epsilon=-1, delta=-1):
        if epsilon == -1:
            epsilon = self.epsilon
        if delta == -1:
            delta = self.delta

        if self.name == "Hoeffding":
            return self.get_Hoeffding_number(epsilon, delta)
        elif self.name == "HoeffdingAbs":
            return self.get_HoeffdingAbs_number(epsilon, delta)
        elif self.name == "Laplace":
            return self.get_Laplace_number(epsilon, delta)

    def get_Hoeffding_number(self, epsilon, delta):
        return int(np.round((self.b - self.a)**2 / (2*epsilon**2) * np.log(1.0/delta(1))))

    def get_HoeffdingAbs_number(self, epsilon, delta):
        return int(np.round((self.b - self.a)**2 / (2*epsilon**2) * np.log(2.0/delta(1))))

    def get_Laplace_number(self, epsilon, delta):
        return 1.0/(epsilon**2) - delta/4.0

    def test(self):
        self.epsilon = 0.9
        self.delta = 0.1

        self.a = 0
        self.b = 1

        print([0.199813375, self.get_HoeffdingAbs_bound(10), (0.199813375 - self.get_HoeffdingAbs_bound(10)) < 0.000001, " ------- HoeffdingAbs"])
        print([0.072581167, self.get_Hoeffding_bound(10), (0.072581167 - self.get_Hoeffding_bound(10)) < 0.000001, " --------- Hoeffding"])

        self.name = "HoeffdingAbs"
        mu = 0.3
        print([self.get_samples([1, 10], mu)[0], "mu:", mu])

        self.b = 10
        print([1.99813375, self.get_HoeffdingAbs_bound(10), (1.99813375 - self.get_HoeffdingAbs_bound(10)) < 0.000001, " -------- HoeffdingAbs(b)"])

        print([self.get_samples([1, 10], mu)[0], "mu:", mu])

        print([self.get_number(), "Time Horizon"])


# test = ConfidenceBound("", [1,2,3,4])
# test.test()
