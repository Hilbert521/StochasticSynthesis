# StochasticSynthesis

The code found in this repository reproduces the numerical examples from the case study section of our paper titled "Abstraction-based Synthesis for Stochastic Systems with Omega-Regular Objectives" [ARXIV LINK]. 

Executing this code requires Python 2.7. Please download all files in this repository. Policy synthesis is performed by running the main file "Synthesis.py", which contains the following features:

- To select a given specification, comment/uncomment the appropriate 'L_mapping' variable found between line 162 and line 175, and comment/uncomment the appropriate 'Automata' and 'Automata_Accepting' variables between line 182 and line 202,

- To choose between finite-mode and continuous set synthesis, comment/uncomment the appropriate sections. The finite-mode section starts on line 236 and ends on line 519. The continuous set section starts on line 528 and ends on line 1279.


To perform a fixed number of refinement steps, change the variables 'Num_Ref' and 'Num_Ref_Dis' on line 85 and 86 of the "Synthesis_Functions.py" file, where 'Num_Ref' corresponds to continuous set synthesis and 'Num_Ref_Dis' to finite-mode synthesis. 

The files required to generate the plots are saved in the current directory after each refinement step. To generate the plots for the finite-mode examples, run the "Plot_File_Discrete.py" file; for the continuous mode examples, run the "Plot_File_Continuous.py" file. 

Note that plotting the final input sets found in the paper for the continuous set case first necessitates finding the corresponding product state number, and set the variable 'S_Plot' on line 176 of "Plot_File_Continuous.py" to that number. To do this, loop through the 'State_Space' variable to find the partition state number for the state to be plotted, then multiply that number by the size of the 'Automata' variable for the specification of interest, and add the automaton state number. For example, to plot the input set for partition state 530 with automaton state s2 for specification phi_1, S_Plot = 530 * 5 + 2. 


If you encounter any issue while running the code or are unable to produce the same plots as in the paper, please feel free to email me at maxdutreix@gmail.com.


