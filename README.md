This Python code was used to generate and analyze the data presented in the manuscript "Quantum 
Gates Between Distant Atoms Mediated by Rydberg Excitation Antiferromagnet" by Georgios Doultsinos 
and David Petrosyan (https://arxiv.org/abs/2408.11542).

The core of the code for producing the data is contained in the files QMFunctions.py, QMClasses.py, 
and QMOptimize.py. The first two files contain functions and classes for simulating different systems 
of interacting atoms together with control pulses, while the third implements the Gradient-Ascent Pulse 
Engineering (GRAPE) method for optimal control, combined with the Broyden–Fletcher–Goldfarb–Shanno (BFGS) 
optimization method. Documentation is included within the code.

Each folder (named accordingly) contains the data and executable scripts necessary to produce and reproduce 
each figure in the manuscript. Scripts named plot___.py generate the exact figures using the included data 
(saved as .npy files). The remaining scripts reproduce and overwrite the included data.

Requirements: Python 3.11, NumPy 1.23.5, SciPy 1.10.1, Matplotlib 3.7.1.
Contact: gdoultsinos@iesl.forth.gr

