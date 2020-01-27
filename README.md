# StochasticSynthesis

The code found in this repository reproduces the numerical examples from the case study section of our paper titled "Abstraction-based Synthesis for Stochastic Systems with Omega-Regular Objectives" [ARXIV LINK]. 

Executing this code requires Python 2.7. Please download all files in this repository. Policy synthesis is performed by running the main file "Synthesis.py", which contains the following features:

- To select a given specification, comment/uncomment the appropriate L_mapping variable found between line 162 and line 175, and comment/uncomment the appropriate Automata and Automata_Accepting variables between line 182 and line 202,

- To choose between finite-mode and continuous set synthesis, comment/uncomment the appropriate sections. The finite-mode section starts on line 236 and ends on line 519. The continuous set section starts on line 528 and ends on line 1279.


To perform a fixed number of refinement steps, change the variables 'Num_Ref' and 'Num_Ref_Dis' in the "Synthesis_Functions.py" file, where 'Num_Ref' corresponds to continuous set synthesis and 'Num_Ref_Dis' to finite-mode synthesis. 

The files required to generate the plots are saved in the current directory after each refinement step.


