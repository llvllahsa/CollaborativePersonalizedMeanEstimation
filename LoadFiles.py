import numpy as np
import Analyze as Analyze
import os

# ---------------------- Writing the setting ----------------------------
def write_setting():
    path = os.getcwd()+'/Results/'
    number = len(os.listdir(path))

    settingwriter = open("Results/{}/setting.txt".format(number), "w")
    if bernoullichoice:
        settingwriter.write("Bernoulli \n")
    else:
        settingwriter.write("Gaussian \n")
    settingwriter.write("means:"+str(mus))
    settingwriter.write("\nepsilon:"+str(epsilon))
    settingwriter.write("\ndelta:"+ str(delta))
    settingwriter.write("\nnumber of agents:"+str(n_agents))
    settingwriter.write("\nhorizon:"+str(horizon))
    settingwriter.write("\nruns:"+str(folds))
    settingwriter.write("\n(class identification theoretical time complexity):"+str(ctimes))
    settingwriter.write("\n(mean estimation theoretical time complexity):"+str(etimes))
    settingwriter.close()
# ------------------------------------------------------------------------


# -------------------------- Loading data -------------------------------------
npzData = np.load("npFiles.npz")

TP = npzData['TP']
FP = npzData['FP']
TN = npzData['TN']
FN = npzData['FN']

TP_class = npzData['TP_class']
FP_class = npzData['FP_class']
TN_class = npzData['TN_class']
FN_class = npzData['FN_class']


precision = npzData['precision']
Xbar_Ca_all = npzData['Xbar_Ca_all']
mus = npzData['mus']
mu_ids = npzData['mu_ids']
algorithm_set = npzData['algorithm_set']
print(algorithm_set)
epsilon = npzData['epsilon']
print(epsilon)
folds=npzData['folds']
horizon=npzData['horizon']
n_agents=npzData['n_agents']
delta = npzData['delta']
bernoullichoice = npzData['bernoullichoice']
ctimes=npzData['ctimes']
etimes=npzData['etimes']
local_mean  = npzData['local_mean']
local_std = npzData['local_std']

# Index of local algorithm
local_alg_index = np.where(algorithm_set == "local")[0][0]

Xbar_Ca_avg = np.mean(Xbar_Ca_all, axis=2)
Xbar_Ca_std = np.std(Xbar_Ca_all, axis=2)

print("Data loaded.")
# -------------------------------- Data Loaded ---------------------------------

analyzer = Analyze.Analyze()

analyzer.plot_class_estimation_all(Xbar_Ca_all, mus[mu_ids], algorithm_set, precision, epsilon)

analyzer.plot_precision_recall_network_new(TP, FP, TN, FN, algorithm_set[0:local_alg_index])
analyzer.plot_precision_recall_classes_new(TP_class, FP_class, algorithm_set[0:local_alg_index],mus, mu_ids)

analyzer.compute_avg_pairclass_mean_estimation_time(Xbar_Ca_all, mus[mu_ids], epsilon, algorithm_set, 0, precision, mus)
analyzer.compute_avg_class_identification_time(mus[mu_ids], precision, algorithm_set)

analyzer.compute_max_pairclass_mean_estimation_time(Xbar_Ca_all, mus[mu_ids], epsilon, algorithm_set, 0, precision, mus)
analyzer.compute_max_class_identification_time(mus[mu_ids], precision, algorithm_set)

write_setting()


print("End!")
