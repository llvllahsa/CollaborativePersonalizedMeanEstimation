# Collaborative Algorithms for Online Personalized Mean Estimation

This project is the implementation of the paper "Collaborative Algorithms for Online Personalized Mean Estimation":
https://arxiv.org/abs/2208.11530

Please email llvllahsa @ gmail for questions and suggestions revolving around this code repository.

## How to Run?

The code is written in python and you can access it via the folder named "CollaborativeCode" and to run the implementation, there are two steps:

1. Running the main file to run different algorithms and save the results in npFiles.npz:
    > python3 main.py

2. Loading npFiles.npz and plotting the results using: 

    > python3 LoadFiles.py


## How to modify and run different experiments?

You can go to the main.py file and main() function.  To change the means, you can change "mus" variable. For instance:

> mus = np.array([0.2, 0.3, 0.5, 0.7])

To change the maximum number of time steps for running each algorithm, you can modify:

> horizon = 1000

To change the number of agents:

> n_agents = 100

The list of algorithms implemented are: "soft-restricted-round-robin", "aggressive-restricted-round-robin", "restricted-round-robin", "round-robin", "oracle", "local" , "random" and you can add each of them to the list containing the algorithms to be run:

> algorithm_set = ["soft-restricted-round-robin", "aggressive-restricted-round-robin"]

To change the distribution functions from Gaussian to Bernoulli, do:

> bernoullichoice = True

To set PAC framework's epsilon and delta:

> epsilon = 0.01
> 
> delta = 0.001

## ConfidenceBound.py

Different concentration inequality bounds are implemented. However, they should be checked based on the analysis. We have used Lemma1 bound for our approach.
