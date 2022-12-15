import matplotlib.pyplot as painter
from matplotlib.lines import Line2D
import numpy as np
import os


class Analyze:

    def __init__(self):
        self.markers = ['.', 'v', '^', '<', '>', 's' , 'p', 'P', '*', 'h' , 'H' , '+', 'x',  'X', 'D',  'd' ,  '|' , '1', '2', '3', '4' , '8', '_']
        self.colors = ['tab:blue', 'tab:orange','tab:green', 'tab:red', 'tab:purple', 'tab:olive','tab:cyan', 'tab:brown', 'tab:pink', 'tab:gray', 'm', 'k']
        path = os.getcwd()+'/Results/'
        self.number = len(os.listdir(path))+1
        os.mkdir(path+'/'+str(self.number))
        return

    def plot_class_estimation_all(self, Xbar_a_all, true_mean, algorithms, precision_, epsilon):
        # Xbar_Ca_all = np.zeros([len(algorithm_set), n_agents, folds, horizon])
        horizon = Xbar_a_all.shape[3]
        n_agents = Xbar_a_all.shape[1]
        unique_means = np.unique(true_mean)

        mean_all = np.zeros([len(algorithms), n_agents, horizon])
        std_all = np.zeros([len(algorithms), n_agents, horizon])
        for alg_index in range(len(algorithms)):
            for h in range(horizon):
                for a in range(n_agents):
                    mean_all[alg_index, a, h] = np.mean(np.abs(Xbar_a_all[alg_index, a, :, h] - true_mean[a]))
                    std_all[alg_index, a, h] = np.std(np.abs(Xbar_a_all[alg_index, a, :, h] - true_mean[a]))

        for c_id in range(len(unique_means)):
            class_ids = np.argwhere(true_mean == unique_means[c_id])

            t = np.array(range(0, horizon, 20))
            painter.figure(figsize=[16, 10])
            painter.rcParams.update({'font.size': 27})
            painter.ylim([0, 0.4])
            if c_id == 2:
                painter.xlim([0, 1000])
            for alg_index in range(len(algorithms)):
                painter.plot(t, np.mean(mean_all[alg_index,class_ids, t], axis=0), self.markers[alg_index]+'-', label=algorithms[alg_index] )
                painter.fill_between(t , np.mean(mean_all[alg_index,class_ids, t], axis=0)
                                     - np.mean(std_all[alg_index, class_ids, t], axis=0), np.mean(mean_all[alg_index,class_ids, t], axis=0)
                                     + np.mean(std_all[alg_index, class_ids, t], axis=0),
                              alpha=0.5, linestyle=":")

            for alg_index in range(len(algorithms)):
                indexx = range(0, horizon, 20).index(int(np.ceil(np.mean(precision_[alg_index, class_ids]))/20)*20)
                if indexx != 0:
                    painter.plot(t ,  np.mean(mean_all[alg_index,class_ids, t], axis=0), marker="o", markevery=[indexx], ls="", color="yellow")


            # yerr = np.linspace(0.1, 0.2, 1)
            # painter.errorbar(n_agents - 3, 0, yerr=yerr)
            # painter.text(n_agents, -0.05,'t={}'.format(n_agents), style='italic', fontsize=20, color="black")
                # --- marked !
                # axes = painter.axes()
                # axes.set_yscale("log")
            painter.legend(loc="upper right")
            painter.xlabel("Time")
            painter.ylabel("Error in mean estimation")
            painter.savefig("Results/{}/Class{}-view-error".format(self.number, c_id)+".pdf")

            painter.show()
            painter.close()

        # ----------------------------- Over the whole network
        painter.figure(figsize=[16, 10])
        painter.rcParams.update({'font.size': 27})
        painter.ylim(0, 2*epsilon)
        # painter.xlim([0, 500])
        for alg_index in range(len(algorithms)):
            painter.plot(t , np.mean(mean_all[alg_index, :, t], axis=1), self.markers[alg_index]+'-', label=algorithms[alg_index] )
            painter.fill_between(t , np.mean(mean_all[alg_index, :, t], axis=1)
                                     - np.mean(std_all[alg_index, :, t], axis=1), np.mean(mean_all[alg_index,:, t], axis=1)
                                     + np.mean(std_all[alg_index, :, t], axis=1),
                              alpha=0.5, linestyle=":")


        # yerr = np.linspace(0.1, 0.2, 1)
        # painter.errorbar(n_agents - 3, 0, yerr=yerr)
        # painter.text(n_agents, -0.05,'t={}'.format(n_agents), style='italic', fontsize=20, color="black")
                # --- marked !
                # axes = painter.axes()
                # axes.set_yscale("log")
        # painter.legend(title="class perspective(mean={})".format(unique_means[c_id]), loc="upper right")
        painter.legend(loc="upper right")
        painter.xlabel("Time")
        painter.ylabel("Error in mean estimation")
        painter.savefig("Results/{}/network-view-error".format(self.number)+".pdf")

        painter.show()
        painter.close()

        return


    def plot_precision_recall_network_new(self, TP, FP, TN, FN, algorithms):
        horizon = TP.shape[3]
        t = np.array(range(0, horizon, 10))
        painter.figure(figsize=[16, 10])
        painter.rcParams.update({'font.size': 27})
        # painter.xlim([0, 500])
        for algorithm_index in range(len(algorithms)):
            if not (algorithm_index == 2 or algorithm_index == 3):
                    continue
            precision = TP[algorithm_index, :, :, t]*1.0/(TP[algorithm_index, :, :, t] + FP[algorithm_index, :, :, t])
            p_mean = np.mean(precision, axis=(1,2))
            p_std = np.std(precision, axis=(1,2))
            painter.plot(t , p_mean, self.markers[algorithm_index]+'-', label="{}".format(algorithms[algorithm_index]), color=self.colors[algorithm_index])
            painter.fill_between(t , p_mean - p_std, p_mean + p_std, alpha=0.5, linestyle=":", color=self.colors[algorithm_index])

        painter.legend(title="")
        painter.xlabel("Time")
        painter.ylabel("Precision in class estimation")
        painter.savefig("Results/{}/Network-view-precision".format(self.number)+".pdf")

        painter.show()
        painter.close()

    def plot_precision_recall_classes_new(self, TP_class, FP_class, algorithms, mus, mu_ids):

        horizon = TP_class.shape[4]
        print(horizon)
        print(TP_class.shape)
        t = np.array(range(0, horizon, 10))
        class_counter = 0
        for i in range(len(mus)):
            for j in range(i+1, len(mus)):
                cls_ind1 = np.where(mu_ids == i)
                cls_ind2 = np.where(mu_ids == j)
                cls_ind = np.concatenate((cls_ind1[0], cls_ind2[0]))
                painter.figure(figsize=[16, 10])
                painter.rcParams.update({'font.size': 27})
                # painter.xlim([0, 500])
                for algorithm_index in range(len(algorithms)):
                    if not (algorithm_index == 2 or algorithm_index == 3):
                            continue
                    tp = TP_class[class_counter, algorithm_index, :, :, t]
                    tp = tp[:, cls_ind, :]
                    fp = FP_class[class_counter, algorithm_index, :, :, t]
                    fp = fp[:, cls_ind, :]
                    precision = tp * 1.0 / np.maximum(1, tp+fp)
                    # precision = TP_class[class_counter, algorithm_index, cls_ind, :, t]*1.0/np.maximum(1, TP_class[class_counter,algorithm_index, cls_ind, :, t] + FP_class[class_counter,algorithm_index, cls_ind, :, t])
                    p_mean = np.mean(precision, axis=(1,2))
                    p_std = np.std(precision, axis=(1,2))
                    painter.plot(t , p_mean, self.markers[algorithm_index]+'-', label="{}".format(algorithms[algorithm_index]), color=self.colors[algorithm_index])
                    painter.fill_between(t , p_mean - p_std, p_mean + p_std, alpha=0.5, linestyle=":", color=self.colors[algorithm_index])

                painter.legend(title="")
                painter.xlabel("Time")
                painter.ylabel("Precision in class estimation("+str(mus[i])+"-"+str(mus[j])+")")
                painter.savefig("Results/{}/pairclass-view-precision{}".format(self.number, class_counter)+".pdf")

                painter.show()
                painter.close()
                class_counter += 1

    def compute_avg_pairclass_mean_estimation_time(self, emp_mean, true_mean, epsilon, algorithm_names, a_, precision_, mus):

        writer = open("Results/{}/avg_mean_estimation_times.txt".format(self.number), "w")

        n_agent = emp_mean.shape[1]
        n_folds = emp_mean.shape[2]
        horizon = emp_mean.shape[3]
        n_algorithm = len(algorithm_names)
        # unique_means = np.unique(true_mean)
        # unique_mean_len = len(unique_means)

        for g in range(n_algorithm):
            convergence_time = np.zeros([n_agent, n_folds, 1])
            for a in range(n_agent):
                for f in range(n_folds):
                    for h in range(horizon):

                        inds = np.where(np.abs(emp_mean[g, a, f, h:] - true_mean[a]) > epsilon)
                        if len(inds[0]) == 0 :
                            convergence_time[a, f] = h
                            break
                        else:
                            convergence_time[a, f] = horizon

                # writer.write(algorithm_names[g] + ", agent: " + str(a) + " (" + str(true_mean[a])+")")
                # writer.write(" time:" + str(np.mean(convergence_time[a, :])) + " +- " + str(np.std(convergence_time[a, :])))
                #
                # writer.write("\n")

            writer.write(algorithm_names[g] + ", Network time:"+ str(np.mean(convergence_time[:, :]))+ " +- " + str(np.std(convergence_time[:, :])))
            writer.write("\n")

            for mu_ind in range(len(mus)):
                inds = np.where(true_mean == mus[mu_ind])
                writer.write(algorithm_names[g] + ", Network class "+str(mu_ind)+" time:"+ str(np.mean(convergence_time[inds, :]))+ " +- " + str(np.std(convergence_time[inds, :])))
                writer.write("\n")

        writer.close()
        print("Done.")

    def compute_max_pairclass_mean_estimation_time(self, emp_mean, true_mean, epsilon, algorithm_names, a_, precision_, mus):

        writer = open("Results/{}/max_mean_estimation_times.txt".format(self.number), "w")

        n_agent = emp_mean.shape[1]
        n_folds = emp_mean.shape[2]
        horizon = emp_mean.shape[3]
        n_algorithm = len(algorithm_names)
        # unique_means = np.unique(true_mean)
        # unique_mean_len = len(unique_means)

        for g in range(n_algorithm):
            convergence_time = np.zeros([n_agent, n_folds, 1])
            for a in range(n_agent):
                for f in range(n_folds):
                    for h in range(horizon):
                        inds = np.where(np.abs(emp_mean[g, a, f, h:] - true_mean[a]) > epsilon)
                        if len(inds[0]) == 0 :
                            convergence_time[a, f] = h
                            break
                        else:
                            convergence_time[a, f] = horizon

                # writer.write(algorithm_names[g] + ", agent: " + str(a) + " (" + str(true_mean[a])+")")
                # writer.write(" time:" + str(np.mean(convergence_time[a, :])) + " +- " + str(np.std(convergence_time[a, :])))
                #
                # writer.write("\n")

            writer.write(algorithm_names[g] + ", Network time:"+ str(np.max(convergence_time[:, :])))
            writer.write("\n")
            writer.write("Max count:")
            maxx = np.max(convergence_time[:, :])
            inds = np.where( maxx == convergence_time[:, :])
            writer.write(str(len(inds[0])))
            writer.write("\n")

            for mu_ind in range(len(mus)):
                inds = np.where(true_mean == mus[mu_ind])
                writer.write(algorithm_names[g] + ", Network class "+str(mu_ind)+" time:"+ str(np.max(convergence_time[inds, :])))
                writer.write("\n")

        writer.close()
        print("Done.")

    def compute_avg_class_identification_time(self, true_mean, precision_, algorithm_names):

        n_algorithm = len(algorithm_names)

        writer = open("Results/{}/avg_class_identification_times.txt".format(self.number), "w")

        writer.write("Class estimation time complexity:")
        writer.write("\n")

        uniques = np.unique(true_mean)

        for g in range(n_algorithm):
            for a in range(len(uniques)):
                inds = np.where(true_mean == uniques[a])
                writer.write(algorithm_names[g]+","+str(a)+": " + str(np.mean(precision_[g, inds, :])) +" +- " + str(np.std(precision_[g, inds, :])))
                writer.write("\n")
        writer.close()
        print("Done.")


    def compute_max_class_identification_time(self, true_mean, precision_, algorithm_names):

        n_algorithm = len(algorithm_names)

        writer = open("Results/{}/max_class_identification_times.txt".format(self.number), "w")

        writer.write("Class estimation time complexity:")
        writer.write("\n")

        uniques = np.unique(true_mean)

        for g in range(n_algorithm):
            for a in range(len(uniques)):
                inds = np.where(true_mean == uniques[a])
                writer.write(algorithm_names[g]+","+str(a)+": " + str(np.max(precision_[g, inds, :])) )
                writer.write("\n")
        writer.close()
        print("Done.")
