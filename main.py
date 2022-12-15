import numpy as np
from scipy.stats import bernoulli
from ConfidenceBound import ConfidenceBound
from sympy import symbols, solve


def intersection(x_aa, cb_aa, x_al, cb_al):
    # Intersection of two confidence intervals:
    intersect_al = (np.minimum(x_al + cb_al, x_aa + cb_aa) - np.maximum(x_al - cb_al, x_aa - cb_aa)) * (
            np.maximum(x_al - cb_al, x_aa - cb_aa) < np.minimum(x_al + cb_al, x_aa + cb_aa))
    return intersect_al


def aggresive_intersect(x_aa, cb_aa, x_al, cb_al, intersect):
    # If a and l 's intersection is more than min of a and l 's confidence bounds
    return np.minimum(cb_aa, cb_al) <= intersect


def union(x_aa, cb_aa, x_al, cb_al):
    # union of a and l's confidence intervals
    union_al = np.minimum(np.maximum(x_al + cb_al, x_aa + cb_aa) - np.minimum(x_al - cb_al, x_aa - cb_aa),
                          2 * cb_al + 2 * cb_aa)
    return union_al


def test_union_intersect_vectorized():
    x_aa = 1
    x_al = np.array([1, 2, 3, 4, 5, 4, 2, 1.9])
    cb_aa = 2
    cb_al = np.array([1, 1, 2, 3, 1, 2, 3, 1])
    intersect = intersection(x_aa, cb_aa, x_al, cb_al)
    print(intersect == [2., 2., 2., 2., -0., 1., 4., 2.])
    unionn = union(x_aa, cb_aa, x_al, cb_al)
    print(unionn == [4., 4., 6., 8., 6., 7., 6., 4.])
    aggresive_intersection = aggresive_intersect(x_aa, cb_aa, x_al, cb_al, intersect)
    print(aggresive_intersection == [1, 1, 1, 1, 0, 0, 1, 1])


def choose(communication_strategy, round_robin_counter=-1, Xbar_t_=None, C_a_t_=None, N_t_=None, curr_a=-1, n_agents_=-1,
           mu_ids_=None, CB_=""):
    # Finds who to query
    if communication_strategy == "random":
        return np.random.randint(0, n_agents_, 1)[0]
    elif communication_strategy == "round-robin":
        round_robin_counter = (round_robin_counter + 1) % n_agents_
        if curr_a == round_robin_counter:
            round_robin_counter = (round_robin_counter + 1) % n_agents_
        return round_robin_counter
    elif communication_strategy.endswith("restricted-round-robin"):
        # tie_breaking_choices = np.where(np.min(N_t_[curr_a, C_a_t_]) == N_t_[curr_a, C_a_t_])[0]
        # return C_a_t_[tie_breaking_choices[np.random.randint(0, len(tie_breaking_choices), 1)[0]]]
        round_robin_counter = (round_robin_counter + 1) % n_agents_
        while round_robin_counter not in C_a_t_ or round_robin_counter == curr_a:
            round_robin_counter = (round_robin_counter + 1) % n_agents_
        # return C_a_t_[np.argmin(N_t_[curr_a, C_a_t_])]
        return round_robin_counter
    elif communication_strategy == "oracle":
        C_a_t_ = np.argwhere(mu_ids_ == mu_ids_[curr_a])
        return C_a_t_[np.argmin(N_t_[curr_a, C_a_t_])]
    elif communication_strategy == "exploit":
        d_a_t = np.abs(Xbar_t_[curr_a, :] - Xbar_t_[curr_a, curr_a]) - CB_.table[N_t_[curr_a, :]] - CB_.table[
            N_t_[curr_a, curr_a]]
        tie_breaking_choices = \
            np.where(np.min(d_a_t[C_a_t_] * (N_t_[curr_a, C_a_t_] + 1)) == d_a_t[C_a_t_] * (N_t_[curr_a, C_a_t_] + 1))[
                0]
        return C_a_t_[tie_breaking_choices[np.random.randint(0, len(tie_breaking_choices), 1)[0]]]


def set_weight(algorithm_name="", Xbar_a_t=None, CB_="", curr_a=0, N_a_t_=None):
    # sets the weighting mechanism
    if algorithm_name == "soft-restricted-round-robin":
        intersect = intersection(Xbar_a_t, CB_.table[N_a_t_], Xbar_a_t[curr_a], CB_.table[N_a_t_[curr_a]])
        unionn = union(Xbar_a_t, CB_.table[N_a_t_], Xbar_a_t[curr_a], CB_.table[N_a_t_[curr_a]])
        return intersect * 1.0 / unionn
    elif algorithm_name == "aggressive-restricted-round-robin":
        intersect = intersection(Xbar_a_t, CB_.table[N_a_t_], Xbar_a_t[curr_a], CB_.table[N_a_t_[curr_a]])
        unionn = union(Xbar_a_t, CB_.table[N_a_t_], Xbar_a_t[curr_a], CB_.table[N_a_t_[curr_a]])
        return intersect * aggresive_intersect(Xbar_a_t, CB_.table[N_a_t_], Xbar_a_t[curr_a], CB_.table[N_a_t_[curr_a]],
                                               intersect) * 1.0 / unionn
    else:
        return np.ones(Xbar_a_t.shape)


def get_class(Xbar_t_, N_t_, a_, CB, mus_ids_, algorithm):
    if algorithm == "oracle":
        C_a_t = np.argwhere(mus_ids_ == mus_ids_[a_])
    elif algorithm == "local":
        C_a_t = np.zeros([1, 1])
        C_a_t[0, 0] = a_
    else:
        d_a_t = np.abs(Xbar_t_[a_, :] - Xbar_t_[a_, a_]) - CB.table[N_t_[a_, :]] - CB.table[N_t_[a_, a_]]
        C_a_t = np.argwhere(d_a_t <= 0)
    return C_a_t.reshape(len(C_a_t), ).copy()


def get_epsilon(delta, mus, mu_ids, CB, which_index):
    for i in range(len(mu_ids)):
        if mu_ids[i] == mu_ids[which_index]:
            continue
        nal_star = symbols('nal_star')
        expr = CB.table[0.5*(2*1.0/nal_star * (1 + 1/nal_star)*np.log((nal_star+1)**0.5*8*len(mu_ids)/delta))**0.5] - 0.5/4*(mus[mu_ids[which_index]] - mus[mu_ids[i]])
        sol = solve(expr)
        print(sol)
    # np.max(CB.table[len(mus)/2.0],CB.table[tmp])
    return


def compute_betainv(point, CB):
    left = 1
    right = 1000000

    while right - left > 1:
        mid = (left+right)/2.0
        eval_mid = CB.get_lemma1_bound(n_observations=mid)
        if point < eval_mid:
            left = mid
        else:
            right = mid

    return mid

def compute_class_identification_times(CB, mus, n_agents):
    times = []

    for i in range(len(mus)):
        Delta = 2
        for j in range(len(mus)):
            if i == j :
                continue
            absd =abs(mus[j] - mus[i])
            if absd < Delta :
                Delta = absd

        ctime = np.ceil(compute_betainv(Delta / 4.0, CB)) + n_agents - 1
        times = times + [ctime]

    return times

def compute_restrictedRR_class_identification_times(ctimes, true_means, n_agents):
    rrr_times = ctimes

    mus = np.unique(true_means)

    for i in range(len(mus)):
        eliminted_count = 0
        for j in range(len(mus)):
            if i == j:
                continue
            if ctimes[i] - n_agents + 1 > ctimes[j]:
                eliminted_count += len(np.where(mus[j]) == true_means)
        rrr_times[i] -= eliminted_count

    return rrr_times


def compute_mean_estimation_times(CB, delta, epsilon, n_class, ctimes):
    etime = compute_betainv(epsilon, CB) / n_class + (n_class - 1) / 2.0
    etimes = ctimes
    for i in range(len(etimes)):
        etimes[i] = np.maximum(etimes[i], etime)
    return etimes

def compute_average_theoretical_estimation_time(true_means, etimes):
    mus = np.unique(true_means)
    avg_etime = 0
    for i in range(len(mus)):
        inds = np.where(true_means == mus[i])
        avg_etime += len(inds[0]) * etimes[i]
    avg_etime /= len(true_means)

    return avg_etime


def __main___():
    folds = 20  # Number of runs
    n_agents = 200  # Number of agents
    horizon = 2500  # T
    # ----------- Bernoulli distributions Setup ------ #
    a = 0
    b = 1
    # mus = np.array([0.2, 0.1, 0.4, 0.6, 0.8])
    # mus = np.array([0.01, 0.33, 0.66, 0.99])
    mus = np.array([0.2, 0.4, 0.8])
    mu_ids = np.random.randint(0, len(mus), n_agents)
    # ------------ Generating bernoulli samples ----- #
    bernoullichoice = False  # False --> Gaussian
    X = np.zeros([folds, n_agents, horizon])
    for i in range(n_agents):
        if bernoullichoice:  # Bernoulli
            X[:, i, :] = bernoulli.rvs(mus[mu_ids][i], size=[folds, horizon])
        else:  # Gaussian
            X[:, i, :] = np.random.normal(mus[mu_ids][i], 0.5, size=[folds, horizon])

    print("Samples Generated.")
    # ----------- Confidence Bound ---------------- #
    CB_name = "lemma1"
    epsilon = 0.01
    delta = 0.001
    delta_func = delta / (8 * n_agents)
    Delta_bar = mus[1] - mus[0] - 0.001
    # params = [epsilon, delta_func, a, b, Delta_bar]  # HoeffdingAbs, Hoeffding
    # params = [epsilon, delta_func, Delta_bar]     # Laplace
    params = [epsilon, delta_func]  # lemma1
    # sigma = 1
    CB = ConfidenceBound(cb_name=CB_name, params=params)
    print("Confidence Bounds Done")
    print(CB.get_number(epsilon, delta_func))
    print(CB.table[500])

    ctimes = compute_class_identification_times(CB, mus, n_agents)
    print("Class identification Theoretical times:")
    print(ctimes)

    rrr_ctimes = compute_restrictedRR_class_identification_times(ctimes, mus[mu_ids], n_agents)

    etimes = compute_mean_estimation_times(CB, delta, epsilon, len(mus), rrr_ctimes)
    print("Mean Estimation Theoretical times(restricted-round-robin):")
    print(etimes)

    print("avg Mean Estimation Theoretical Time:")
    print(compute_average_theoretical_estimation_time(mus[mu_ids], etimes))

    print("Local Mean Estimation Theoretical Time:")
    print(compute_betainv(epsilon, CB))

    # ---------- Algorithm Variables --------------- #

    algorithm_set = ["soft-restricted-round-robin", "aggressive-restricted-round-robin", "restricted-round-robin", "round-robin", "oracle",
                     "local"]  # "exploit", "random"]

    # algorithm_set = ["restricted-round-robin", "round-robin", "local"]
    # algorithm_set = ["soft-restricted-round-robin", "local"]

    # Precision and Recall variables
    TP = np.zeros([len(algorithm_set), n_agents, folds, horizon])
    FP = np.zeros([len(algorithm_set), n_agents, folds, horizon])
    TN = np.zeros([len(algorithm_set), n_agents, folds, horizon])
    FN = np.zeros([len(algorithm_set), n_agents, folds, horizon])
    precision = np.zeros([len(algorithm_set), n_agents, folds])
    precision_is_set = np.zeros([len(algorithm_set), n_agents, folds]) == 1

    # Precision and recall pair class varibales
    n_class = len(mus)
    TP_class = np.zeros([int(n_class*(n_class - 1)/2),len(algorithm_set), n_agents, folds, horizon])
    FP_class = np.zeros([int(n_class*(n_class - 1)/2),len(algorithm_set), n_agents, folds, horizon])
    TN_class = np.zeros([int(n_class*(n_class - 1)/2),len(algorithm_set), n_agents, folds, horizon])
    FN_class = np.zeros([int(n_class*(n_class - 1)/2),len(algorithm_set), n_agents, folds, horizon])
    precision_class = np.zeros([n_class*(n_class - 1),len(algorithm_set), n_agents])
    precision_is_set_class = np.zeros([n_class*(n_class - 1),len(algorithm_set), n_agents, folds]) == 1

    Xbar_Ca_all = np.zeros([len(algorithm_set), n_agents, folds, horizon])
    local_std = np.zeros([len(algorithm_set), n_agents, horizon])
    local_mean = np.zeros([len(algorithm_set), n_agents, horizon])
    # get_epsilon(delta, mus, mu_ids, CB, 0)

    for fold in range(folds):
        for algorithm_index in range(len(algorithm_set)):
            Xbar_t = np.zeros([n_agents,
                               n_agents])  # last avg value of samples for each neighbour from the perspective of a particular agent, for all agents
            N_t = np.zeros([n_agents, n_agents],
                           dtype=np.int64)  # number of samples from the perspective of an agent, for a particular neighbour
            Xbar_Ca = np.zeros([n_agents, horizon])
            round_robin_counter = np.zeros([n_agents, ], dtype=int) - 1
            tmp_precision = np.zeros([len(algorithm_set), n_agents, folds])
            for t in range(horizon):
                # Perceive
                for a in range(n_agents):
                    Xbar_t[a, a] = np.mean(X[fold, a, 0:t + 1])
                    N_t[a, a] = t + 1
                    Xbar_Ca[a, t] = Xbar_t[a, a]

                if algorithm_set[algorithm_index] == "local":

                    continue

                # Communicate
                for a in range(n_agents):

                    C_a_t = get_class(Xbar_t_=Xbar_t, N_t_=N_t, a_=a, CB=CB, mus_ids_=mu_ids,
                                      algorithm=algorithm_set[algorithm_index]).copy()
                    # q_ind = choose("random")  # Random choice
                    q_ind = choose(communication_strategy=algorithm_set[algorithm_index],
                                   round_robin_counter=round_robin_counter[a], Xbar_t_=Xbar_t, C_a_t_=C_a_t, N_t_=N_t,
                                   curr_a=a, n_agents_=n_agents, mu_ids_=mu_ids, CB_=CB)  # Round-robin choice
                    if algorithm_set[algorithm_index] == "round-robin" or algorithm_set[algorithm_index].endswith("restricted-round-robin"):
                        round_robin_counter[a] = q_ind
                    #       - Update neighbouring mean values
                    Xbar_t[a, q_ind] = Xbar_t[q_ind, q_ind]
                    N_t[a, q_ind] = N_t[q_ind, q_ind]
                    #       - Class estimation
                    C_a_t = get_class(Xbar_t_=Xbar_t, N_t_=N_t, a_=a, CB=CB, mus_ids_=mu_ids,
                                      algorithm=algorithm_set[algorithm_index]).copy()

                    # Setting Variables for Precision and Recall
                    Positives = np.zeros([n_agents])
                    Positives[C_a_t] = 1.0
                    Trues = np.zeros([n_agents])
                    Trues[np.where(mu_ids == mu_ids[a])] = 1.0
                    TP[algorithm_index, a, fold, t] = np.dot(Trues, Positives)
                    FP[algorithm_index, a, fold, t] = np.dot(1 - Trues, Positives)
                    TN[algorithm_index, a, fold, t] = np.dot(1 - Trues, 1 - Positives)
                    FN[algorithm_index, a, fold, t] = np.dot(Trues, 1 - Positives)

                    # Setting variables for pair-class Precision and recall
                    cls_counter = 0
                    for cl1 in range(n_class):
                        for cl2 in range(cl1+1,n_class):
                            cls_ind1 = np.where(mu_ids == cl1)
                            cls_ind2 = np.where(mu_ids == cl2)
                            cls_ind = np.concatenate((cls_ind1[0], cls_ind2[0]))
                            TP_class[cls_counter, algorithm_index, a, fold, t] = np.dot(Trues[cls_ind], Positives[cls_ind])
                            FP_class[cls_counter, algorithm_index, a, fold, t] = np.dot(1 - Trues[cls_ind], Positives[cls_ind])
                            TN_class[cls_counter, algorithm_index, a, fold, t] = np.dot(1 - Trues[cls_ind], 1 - Positives[cls_ind])
                            FN_class[cls_counter, algorithm_index, a, fold, t] = np.dot(Trues[cls_ind], 1 - Positives[cls_ind])
                            cls_counter += 1


                    # Finding the time that precision becomes 1(recall is 1; thus, class is found)
                    if not precision_is_set[algorithm_index, a, fold]:
                        if TP[algorithm_index, a, fold, t] / (
                                TP[algorithm_index, a, fold, t] + FP[algorithm_index, a, fold, t]) == 1:
                            # precision[algorithm_index, a] += t * 1.0 / folds
                            precision[algorithm_index, a, fold] = t
                            #tmp_precision[algorithm_index, a, fold] = t * 1.0 / folds
                            precision_is_set[algorithm_index, a, fold] = True
                    else:
                        if TP[algorithm_index, a, fold, t] / (
                                TP[algorithm_index, a, fold, t] + FP[algorithm_index, a, fold, t]) != 1:
                            precision_is_set[algorithm_index, a, fold] = False
                            #precision[algorithm_index, a] -= tmp_precision[algorithm_index, a, fold]

                    #       - Aggregated Mean estimation
                    alpha_ai = set_weight(algorithm_set[algorithm_index], Xbar_a_t=Xbar_t[a, :], CB_=CB, curr_a=a,
                                          N_a_t_=N_t[a, :])
                    Xbar_Ca[a, t] = np.dot(Xbar_t[a, C_a_t], N_t[a, C_a_t] * alpha_ai[C_a_t]) / np.sum(N_t[a, C_a_t] * alpha_ai[C_a_t])

            Xbar_Ca_all[algorithm_index, :, fold, :] = Xbar_Ca.copy()
            print("{}: {}".format(fold, algorithm_set[algorithm_index]))

    # ----------------------- Save ---------------------------
    np.savez('npFiles', TP=TP, FP=FP, TN=TN, FN=FN, precision=precision, Xbar_Ca_all=Xbar_Ca_all, mus=mus,
             mu_ids=mu_ids, algorithm_set=algorithm_set, epsilon=epsilon, folds=folds, horizon=horizon, n_agents=n_agents,
             delta=delta, bernoullichoice=bernoullichoice, ctimes=ctimes, etimes=etimes, local_mean=local_mean, local_std=local_std, TP_class=TP_class, TN_class=TN_class, FP_class=FP_class, FN_class=FN_class)

    print("End")


__main___()
