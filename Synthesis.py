#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:13:17 2019

@author: maxencedutreix
"""
import matplotlib as mpl #COMMENT TWO LINES IF WANT TO SEE PLOTS
mpl.use('pdf')
import numpy as np
import Synthesis_Functions as func
import matplotlib.pyplot as plt
from matplotlib import rc
import timeit
import sys
import random
import copy
import itertools
from itertools import combinations
from shapely.geometry import Polygon
import shapely
import scipy.sparse as sparse
import scipy.sparse.csgraph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
sys.setrecursionlimit(10000)
from math import sqrt
from math import erf
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds
from scipy.optimize import minimize
from matplotlib.patches import Rectangle
import os
import igraph
from execjs import get
from scipy.optimize import basinhopping
import matplotlib.patches as patches
import matplotlib.lines as mlines
import pickle
from scipy.sparse import lil_matrix


start = timeit.default_timer()
LOW_1 = 0.0 # These variable are the lower bounds and upper bounds of your domain
UP_1 = 4.0
LOW_2 = 0.0
UP_2 = 4.0
U_MIN_1 = -0.05 #Bounds on the input
U_MAX_1 = 0.05 
U_MIN_2 = -0.05
U_MAX_2 = 0.05
sigma1 = sqrt(0.1)
sigma2 = sqrt(0.1)
mu1 = -0.3
mu2 = -0.3
mus = [mu1, mu2]
Gaussian_Width_1 = 0.2
Gaussian_Width_2 = 0.2
Semi_Width_1 = Gaussian_Width_1/2.0
Semi_Width_2 = Gaussian_Width_2/2.0
w_list = [[mus[0] - Semi_Width_1, mus[0] + Semi_Width_1], [mus[1] - Semi_Width_2, mus[1] + Semi_Width_2]]
w1_up = mu1 + Semi_Width_1
w1_low = mu1 - Semi_Width_1
w2_up = mu2 + Semi_Width_2
w2_low = mu2 - Semi_Width_2
Threshold_Uncertainty = 0.30
Running_Times = []
Fraction_Above = []
 


#np.set_printoptions(threshold='NaN')
start = timeit.default_timer()
plt.close("all")
Space_Tag = 0
I_d = 0.01 #Refinement Threshold
V_Stop = 0.04
First_Verif = 1

def norm_u(u):
    return sqrt((u[0]**2) + (u[1]**2))

def Psi(xx):
    return (0.5)*(1.0+erf(xx/sqrt(2.0)))

def dummy_function(yy):
    return 0.0

def dummy_jac(hhh):
    return [0.0,0.0]

def dummy_hess(lll):
    return [[0.0,0.0],[0.0,0.0]]

def Upper_Bound_Func(inpx,inpy, a1,b1,a2,b2,l1,u1,l2,u2):
    
    #Computing the upper bound probability of transition for a given input
    
    m1 = mu1
    m2 = mu2
    s1 = sigma1
    s2 = sigma2
    sqr = sqrt(2.0)
    
    x_max_s = float(max(l1+inpx, min(u1+inpx, ( ((a1+b1)/2.0) - m1) ) )) #Computing the maximizing shifts
    y_max_s = float(max(l2+inpy, min(u2+inpy, ( ((a2+b2)/2.0) - m2) ) ))
    
       
    alpha_x = (w1_low - m1)/s1
    beta_x = (w1_up - m1)/s1
    alpha_y = (w2_low - m2)/s2
    beta_y = (w2_up - m2)/s2
    
    if (b1 - x_max_s) < w1_low:
        x_up = alpha_x        
    elif(b1 - x_max_s) > w1_up:
        x_up = beta_x
    else:
        x_up = ((b1 - x_max_s) - m1)/s1
        
    if (a1 - x_max_s) < w1_low:
        x_low = alpha_x        
    elif(a1 - x_max_s) > w1_up:
        x_low = beta_x
    else:
        x_low = ((a1 - x_max_s) - m1)/s1

    if (b2 - y_max_s) < w2_low:
        y_up = alpha_y       
    elif(b2 - y_max_s) > w2_up:
        y_up = beta_y
    else:
        y_up = ((b2 - y_max_s)- m2)/s2
        
    if (a2 - y_max_s) < w2_low:
        y_low = alpha_y        
    elif(a2 - y_max_s) > w2_up:
        y_low = beta_y
    else:
        y_low = ((a2 - y_max_s) - m2 )/s2       
    
    return (((0.5)*(1.0+erf(x_up/sqr)) - ((0.5)*(1.0+erf(alpha_x/sqr))))/((0.5)*(1.0+erf(beta_x/sqr)) - (0.5)*(1.0+erf(alpha_x/sqr)))) - (((0.5)*(1.0+erf(x_low/sqr)) - (0.5)*(1.0+erf(alpha_x/sqr)))/((0.5)*(1.0+erf(beta_x/sqr)) - (0.5)*(1.0+erf(alpha_x/sqr))))* (((0.5)*(1.0+erf(y_up/sqr)) - ((0.5)*(1.0+erf(alpha_y/sqr))))/((0.5)*(1.0+erf(beta_y/sqr)) - (0.5)*(1.0+erf(alpha_y/sqr)))) - (((0.5)*(1.0+erf(y_low/sqr)) - ((0.5)*(1.0+erf(alpha_y/sqr))))/((0.5)*(1.0+erf(beta_y/sqr)) - (0.5)*(1.0+erf(alpha_y/sqr))))


State_Space = np.array( [
                [[0.0,0.0],[1.0,1.0]],
                [[1.0,0.0],[2.0,1.0]],
                [[2.0,0.0],[3.0,1.0]],
                [[3.0,0.0],[4.0,1.0]], 
                [[0.0,1.0],[1.0,2.0]],
                [[1.0,1.0],[2.0,2.0]],
                [[2.0,1.0],[3.0,2.0]],
                [[3.0,1.0],[4.0,2.0]],
                [[0.0,2.0],[1.0,3.0]],
                [[1.0,2.0],[2.0,3.0]],
                [[2.0,2.0],[3.0,3.0]],
                [[3.0,2.0],[4.0,3.0]],
                [[0.0,3.0],[1.0,4.0]],
                [[1.0,3.0],[2.0,4.0]],
                [[2.0,3.0],[3.0,4.0]],
                [[3.0,3.0],[4.0,4.0]],      
               ] )
    
    
L_mapping = ['', '', '', '',
             '','A','A', '',
             '', '','A','',
             '', '', '', '']
    

#L_mapping = ['A', '', '', 'B',
#             '','B','', 'B',
#             '', '','','',
#             'B', '', 'C', '']    


#Labeling states in the partitioned State-Space.  # Order is from (0,0) and moving horizontal. "mirror image"

###AUTOMATA REPRESENTING THE SPECIFICATION PHI

Automata = [[['A'], [''],[], [], []], 
             [[], [''],['A'], [], []],
             [[], [],[], ['A'], ['']],
             [['A'], [],[], [], ['']],
             [[], [],[], [], ['', 'A']]]

Automata_Accepting = [[[],[0,1,2,3]]]


#Automata = [[[''], ['A'], ['C'], [],[],['B'],[]], 
#             [[''], ['A'], ['C'], [],[],['B'],[]],
#             [[], [], ['C',''], ['A'],['B'],[],[]],
#               [[], [], ['C',''], ['A'],['B'],[],[]],
#                [[], [], [], [],['A','B','C',''],[],[]],
#                [[], [], [], [],[],['A','B',''],['C']],
#                [[], [], [], [],[],[],['', 'A', 'B', 'C']]]
#
#Automata_Accepting = [[[1],[0]], [[3],[2]], [[],[5]]]


#Automata_Accepting contains the Rabin Pairs (Ei, Fi) of Automata, with Fi being the 'good' states


tag = 0;
func.Initial_Partition_Plot(State_Space); # calling function to plot the initial partitions based on state space variables. visual aid

    
Input_Space = list([[-0.05, -0.05], [0.05, 0.05]]) # Initial State of available inputs // CHANGE BOUNDS IN TAC_Synthesis file if modified
Input_Quantitative = list([])
Input_Qualitative = list([])

Step = 3 #Sample Step
List_A = np.linspace(Input_Space[0][0], Input_Space[1][0], Step)    # Sampling the Input Space to generate the discrete-input problem
List_B = np.linspace(Input_Space[0][1], Input_Space[1][1], Step)

Discrete_Input_Space = []

     
Discrete_Input_Space.append([0.0,0.0])  
Discrete_Input_Space.append([Input_Space[0][0],0.0])      
Discrete_Input_Space.append([Input_Space[1][0],0.0]) 
Discrete_Input_Space.append([0.0,Input_Space[0][1]])      
Discrete_Input_Space.append([0.0,Input_Space[1][1]])    
      
for i in range(State_Space.shape[0]):
    Input_Quantitative.append(list([list(Input_Space)])) #Set of inputs for the quantitative problem for all states, which is a list of square inputs
    Input_Qualitative.append(list([list(Input_Space)])) #Set of inputs for the qualitative problem for all states
    
      
            

''' DISCRETE-INPUT CASE STUDY '''    

Reachable_Sets = func.Reachable_Sets_Computation_Finite(State_Space, Discrete_Input_Space);


(Lower_Bound_Matrix, Upper_Bound_Matrix, Reachable_States, Is_Bridge_State, Bridge_Transitions, Pre_States) = func.BMDP_Computation(Reachable_Sets, State_Space, Discrete_Input_Space) 
(IA1_l, IA1_u, Is_Accepting, Is_Non_Accepting, Which_Accepting_Pair, Which_Non_Accepting_Pair, Product_Reachable_States, Product_Is_Bridge_State, Product_Bridge_Transitions, Init, Product_Pre_States) = func.Build_Product_BMDP(Lower_Bound_Matrix, Upper_Bound_Matrix, Automata, L_mapping, Automata_Accepting, Reachable_States, Is_Bridge_State, Bridge_Transitions)

Allowable_Actions = []
for i in range(len(State_Space)*len(Automata)):
    Allowable_Actions.append(range(len(Discrete_Input_Space)))



first = 1;
Allowable_Action_Potential = list([]) #Actions that could make the state a potential BSCC
Allowable_Action_Permanent = list([]) #Actions that could make the state a permanent BSCC
Is_In_Permanent_Comp = np.zeros(IA1_l['Num_S']) #Has a 1 if the state is in a permanent 
Is_In_Permanent_Comp = Is_In_Permanent_Comp.astype(int)
List_Permanent_Accepting_BSCC = [] #Lists which will keeps track of all permanent accepting BSCCs
List_Permanent_Non_Accepting_BSCC = list([])
Previous_Accepting_BSCC = list([])
Previous_Non_Accepting_BSCC = list([])

Objective = 0

if Objective == 0:
       
   Optimal_Policy = np.zeros(IA1_l['Num_S'])
   Optimal_Policy = Optimal_Policy.astype(int)
   Potential_Policy = np.zeros(IA1_l['Num_S']) #Policy to generate the "best" best-case (maximize upper bound)
   Potential_Policy = Potential_Policy.astype(int)
   
   IA1_l_Array = np.zeros(((IA1_l['Num_Act']),IA1_l['Num_S'],IA1_l['Num_S']))
   IA1_u_Array = np.zeros(((IA1_l['Num_Act']),IA1_l['Num_S'],IA1_l['Num_S']))
   
   for k in IA1_l.keys():
       if isinstance(k,tuple):
           IA1_l_Array[k[0],k[1],k[2]] = IA1_l[k]

   for k in IA1_u.keys():
       if isinstance(k,tuple):
           IA1_u_Array[k[0],k[1],k[2]] = IA1_u[k]           
   
   (Greatest_Potential_Accepting_BSCCs, Greatest_Permanent_Accepting_BSCCs, Potential_Policy, Optimal_Policy, Allowable_Action_Potential, Allowable_Action_Permanent, first, Is_In_Permanent_Comp, List_Permanent_Accepting_BSCC, List_Potential_Accepting_BSCC, Which_Potential_Accepting_BSCC, Is_In_Potential_Accepting_BSCC, Bridge_Accepting_BSCC) = func.Find_Greatest_Accepting_BSCCs(IA1_l_Array, IA1_u_Array, Is_Accepting, Is_Non_Accepting, Which_Accepting_Pair, Which_Non_Accepting_Pair, Allowable_Action_Potential, Allowable_Action_Permanent, first, Product_Reachable_States, Product_Bridge_Transitions, Product_Is_Bridge_State, Automata_Accepting, Potential_Policy, Optimal_Policy, Is_In_Permanent_Comp, List_Permanent_Accepting_BSCC, Previous_Accepting_BSCC) # Will return greatest potential accepting bsccs

   Which_Potential_Accepting_BSCC = Which_Potential_Accepting_BSCC.astype(int)
   (Greatest_Permanent_Winning_Component, Optimal_Policy, Is_In_Permanent_Comp, List_Potential_Accepting_BSCC, Is_In_Potential_Winning_Component, Bridge_Accepting_BSCC) = func.Find_Greatest_Winning_Components(IA1_l_Array, IA1_u_Array, Is_Accepting, Is_Non_Accepting, Which_Accepting_Pair, Which_Non_Accepting_Pair, Allowable_Actions, Product_Reachable_States, Product_Bridge_Transitions, Product_Is_Bridge_State, Automata_Accepting, Optimal_Policy, Is_In_Permanent_Comp, List_Permanent_Accepting_BSCC, List_Potential_Accepting_BSCC, Greatest_Permanent_Accepting_BSCCs, Is_In_Potential_Accepting_BSCC, Which_Potential_Accepting_BSCC, Bridge_Accepting_BSCC, len(Automata), Init, Automata_Accepting) # Will return greatest potential and permanent winning components

   List_Potential_Accepting_BSCC.sort()
   List_Potential_Accepting_BSCC = list(List_Potential_Accepting_BSCC for List_Potential_Accepting_BSCC,_ in itertools.groupby(List_Potential_Accepting_BSCC))

#   Previous_Potential_Winning_Components = list(List_Potential_Accepting_BSCC)
   Allowable_Action_Permanent = copy.deepcopy(Allowable_Action_Potential) #Some potential actions could be permanent actions for certain states under refinement 
  
   Greatest_Potential_Winning_Component = list([])
   Potential_BSCCs = [item for sublist in List_Potential_Accepting_BSCC for item in sublist]
   Previous_Potential_Winning_Components = list(Potential_BSCCs)
   Previous_Potential_Winning_Components.sort()
      
   for i in range(len(Greatest_Permanent_Winning_Component)):
       Greatest_Potential_Winning_Component.append(Greatest_Permanent_Winning_Component[i])
       Allowable_Actions[Greatest_Permanent_Winning_Component[i]] = list([Optimal_Policy[Greatest_Permanent_Winning_Component[i]]])

   for i in range(len(Potential_BSCCs)):
       Greatest_Potential_Winning_Component.append(Potential_BSCCs[i])

   
     
   #Using this for now until I actually look for the sink states
   Is_In_Potential_Winning_Component = list(Is_In_Potential_Accepting_BSCC)
   List_Potential_Winning_Components = copy.deepcopy(List_Potential_Accepting_BSCC)
   Bridge_Winning_Components = []
   Which_Potential_Winning_Component = [[] for i in range(IA1_l['Num_S'])] 
   for i in range(len(List_Potential_Accepting_BSCC)):
       Bridge_Winning_Components.append([[]])
       for k in range(len(List_Potential_Accepting_BSCC[i])):
           Which_Potential_Winning_Component[List_Potential_Accepting_BSCC[i][k]].append(list([i,0]))           
           for j in range(len(Allowable_Actions[List_Potential_Accepting_BSCC[i][k]])):
               if Product_Is_Bridge_State[ Allowable_Actions[List_Potential_Accepting_BSCC[i][k]][j] ][List_Potential_Accepting_BSCC[i][k]] == 1:
                   Bridge_Winning_Components[-1][-1].append(List_Potential_Accepting_BSCC[i][k])
                   break
  
    
   Reach_Allowed_Actions = []
   for y in range(IA1_l['Num_S']): # For all the system states
       Reach_Allowed_Actions.append(Allowable_Actions[y])
       
       
   Permanent_Losing_Components = [] 
    
   (Low_Bound, Low_Bounds_Prod, Worst_Markov_Chain, Optimal_Policy, List_Values_Low) = func.Maximize_Lower_Bound_Reachability(IA1_l, IA1_u, Greatest_Permanent_Winning_Component, State_Space.shape[0], len(Automata), Product_Reachable_States, Init, Optimal_Policy, Reach_Allowed_Actions, Product_Pre_States, Permanent_Losing_Components) # Maximizes Lower Bound
   (Upp_Bound, Upp_Bounds_Prod, Best_Markov_Chain, Potential_Policy, List_Values_Up) = func.Maximize_Upper_Bound_Reachability(IA1_l, IA1_u, Greatest_Potential_Winning_Component, State_Space.shape[0], len(Automata), Product_Reachable_States, Init, Potential_Policy, Reach_Allowed_Actions, Product_Pre_States, Permanent_Losing_Components) # Maximizes for winning component



#### The code below takes care of finding the permanent and potential losing components as well as their bridge states for the purpose of refinement 
    
   Is_In_Potential_Losing_Component = np.zeros(IA1_l['Num_S']) #Contains a 1 if the state is in a potential losing component
   Is_In_Potential_Losing_Component = Is_In_Potential_Losing_Component.astype(int)
   Which_Losing_Component = [[] for i in range(IA1_l['Num_S'])] #Tells to you which losing component the state belongs to. For each state, there will be a list of lists [a,b], [a',b'] etc. where a is the BSCC number and b is the number of the component around that BSCC (number 0 is the BSCC by itself)
   Permanent_Losing_Components = []
   Greatest_Permanent_Winning_Component = list([]) 

   Potential_Opposite_Components = [] #Contains the potential losing components of the Worst Case Markov Chain for refinement
   Is_Pot_Comp_Bridge = []#List which tells you whether the component is a Bridge State or not
   for i in range(len(Low_Bounds_Prod)):
       if Low_Bounds_Prod[i] == 0:
           if Upp_Bounds_Prod[i] > 0:
               Potential_Opposite_Components.append(i)
#               Is_In_Potential_Losing_Component[i] = 1
               if Product_Is_Bridge_State[Optimal_Policy[i]][i] == 1:
                   Is_Pot_Comp_Bridge.append(1)
               else:
                   Is_Pot_Comp_Bridge.append(0)
           else:
               Is_In_Permanent_Comp[i] = 1
               Permanent_Losing_Components.append(i)
               Allowable_Actions[i] = list([Optimal_Policy[i]])
       elif Low_Bounds_Prod[i] == 1.0:
           Is_In_Permanent_Comp[i] = 1
           Greatest_Permanent_Winning_Component.append(i)
           Allowable_Actions[i] = list([Optimal_Policy[i]])
    
   Component_Graph = np.zeros((len(Potential_Opposite_Components),len(Potential_Opposite_Components))) #Sub-Graph which will be analyzed to find all distinct components and their corresponding bridge States
   Have_Connection = np.zeros(len(Potential_Opposite_Components))
   
   Unconnected_States = []
   for i in range(len(Potential_Opposite_Components)):
       for j in range(len(Potential_Opposite_Components)):
           if Worst_Markov_Chain[Potential_Opposite_Components[i],Potential_Opposite_Components[j]] > 0:
               Component_Graph[i,j] = 1
               Have_Connection[i] = 1
       if Have_Connection[i] == 0:
          Unconnected_States.append(i)               

   Absorbing_State = 0 #Tag that tells you whether an absorbing state (representing the permanent components) has been added to the graph
   if len(Unconnected_States) != 0:
       Absorbing_State = 1  
       b = np.zeros((Component_Graph.shape[0]+1,Component_Graph.shape[0]+1)) #Create new graph with added absorbing state
       b[:Component_Graph.shape[0], :Component_Graph.shape[0]] = Component_Graph
       Component_Graph = np.asarray(b)
       Component_Graph[-1,-1] = 1 
       for i in range(len(Unconnected_States)):
           Component_Graph[Unconnected_States[i],-1] = 1

   (List_Potential_Losing_Components, Which_Potential_Losing_Component, Bridge_Losing_Components) = func.Find_Components_And_Bridge_States(Component_Graph ,Which_Losing_Component, Potential_Opposite_Components, Is_Pot_Comp_Bridge, Absorbing_State)


Greatest_Suboptimality_Factor = 0
Suboptimality_Factors = []
States_Above_Threshold = []
Success_Intervals = [[] for n in range(len(State_Space))] #Contains the probability of satisfying the spec in the original system (under the computed policies)
Product_Intervals = [[] for n in range(IA1_l['Num_S'])] #Contains the probability of satisfying the spec in the product (Under the computed policies)


#Here optimality factor is the number of possible actions left for each state
Sum_Num_Actions = 0
for i in range(len(Allowable_Actions)):        
    Delete_Actions = []
    Optimal = Optimal_Policy[i]
    Index_Optimal = Allowable_Actions[i].index(Optimal)
    Max_Sub_Fac = 0.0 #Keeps track of the suboptimality factor of a given state
    for j in range(len(Allowable_Actions[i])):
        if Allowable_Actions[i][j] != Optimal:
           if List_Values_Up[i][j] <= List_Values_Low[i][Index_Optimal]: #Action is considered suboptimal if upper bound is less than lower bound of optimal action + some tolerance
               Delete_Actions.append(Allowable_Actions[i][j])
           else:
               Max_Sub_Fac = max(Max_Sub_Fac, float(List_Values_Up[i][j] - List_Values_Low[i][Index_Optimal]))
    if len(Delete_Actions) != 0:
        for j in range(len(Delete_Actions)):
            Allowable_Actions[i].remove(Delete_Actions[j])
    
    Sum_Num_Actions += len(Allowable_Actions[i])
    
    Suboptimality_Factors.append(Max_Sub_Fac)    
    Greatest_Suboptimality_Factor = max(Greatest_Suboptimality_Factor, Max_Sub_Fac)   
    
    if Max_Sub_Fac >= Threshold_Uncertainty: #If the suboptimality factor is greater than the threshold precision, we need to refine with respect to that state
        States_Above_Threshold.append(i)


List_Max_Opt = []
List_Avg_Opt = []
Fractions_Above = []
Avg_Num_Actions = []

List_Max_Opt.append(Greatest_Suboptimality_Factor)    
List_Avg_Opt.append(sum(Suboptimality_Factors)/float(len(Suboptimality_Factors)))
Fractions_Above.append(len(States_Above_Threshold)/float(len(Suboptimality_Factors))) 
Running_Times.append(timeit.default_timer() - start) 
Avg_Num_Actions.append(float(Sum_Num_Actions)/float(len(Suboptimality_Factors)))

        
for i in range(len(Upp_Bound)):    
    Success_Intervals[i].append(Low_Bound[i])
    Success_Intervals[i].append(Upp_Bound[i])


for i in range(len(Upp_Bounds_Prod)):   
    Product_Intervals[i].append(Low_Bounds_Prod[i])
    Product_Intervals[i].append(Upp_Bounds_Prod[i])


#Greatest_Suboptimality_Factor  = 0.0
   
if Greatest_Suboptimality_Factor >= Threshold_Uncertainty:
        (State_Space, Low_Bound, Upp_Bound, Low_Bounds_Prod, Upp_Bounds_Prod, Potential_Policy, Optimal_Policy, Suboptimality_Factors, Init, L_mapping, Allowable_Actions, List_Max_Opt, List_Avg_Opt, Fractions_Above, Running_Times, Avg_Num_Actions) = func.State_Space_Refinement_BMDP(State_Space, Threshold_Uncertainty, Greatest_Suboptimality_Factor, Objective, Potential_Policy, Optimal_Policy, Best_Markov_Chain, Worst_Markov_Chain, first, States_Above_Threshold, Success_Intervals, Product_Intervals, len(Automata), Lower_Bound_Matrix, Upper_Bound_Matrix, Automata, Automata_Accepting, L_mapping, IA1_u, IA1_l, Is_In_Permanent_Comp, Allowable_Action_Potential, Allowable_Action_Permanent, List_Potential_Winning_Components, Which_Potential_Winning_Component, Bridge_Winning_Components, Is_In_Potential_Winning_Component, List_Potential_Losing_Components, Which_Potential_Losing_Component, Bridge_Losing_Components, Is_In_Potential_Losing_Component, Reachable_States, Is_Bridge_State, Bridge_Transitions, Pre_States, Product_Reachable_States, List_Permanent_Accepting_BSCC, Previous_Accepting_BSCC, Previous_Non_Accepting_BSCC, Allowable_Actions, Discrete_Input_Space, List_Max_Opt, List_Avg_Opt, Fractions_Above, Running_Times, Greatest_Permanent_Winning_Component, Previous_Potential_Winning_Components, Permanent_Losing_Components, Avg_Num_Actions)

        print 'Final Suboptimality Factor'
        print List_Max_Opt[-1]
        func.State_Space_Plot(State_Space)
        
        fig1= plt.figure('Greatest and Average Suboptimality Factors')
        plt.title(r'Precision Metrics vs. Refinement Step ($\phi_{1}$ Synthesis, Finite Inputs)')
        plt.plot(List_Max_Opt, label = r'Greatest Suboptimality Factor')
        plt.plot(List_Avg_Opt, label = r'Average Suboptimality Factor')
        plt.plot(Fractions_Above, label = r'Fraction of States above $\epsilon_{max}$')
        ax = plt.gca()
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        ax.set_xlabel(r'Refinement Step', fontsize=17)
        ax.set_ylabel(r'Controller Precision', fontsize=17)
        ax.set(ylim=(0, 1.4))
        plt.legend()
        plt.savefig('Optimality_Factors.pdf', bbox_inches='tight') 

        
        fig2= plt.figure('Running Time')
        plt.title(r'Cumulative Execution Time for $\phi_{1}$ Synthesis (Finite Inputs)', fontsize = 14)
        plt.plot(np.cumsum(Running_Times), label = r'Cumulative Execution Time')
        ax = plt.gca()
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        ax.set_xlabel(r'Refinement Step', fontsize=17)
        ax.set_ylabel(r'Time (s)', fontsize=17)
        ax.set_xticks(ax.get_xticks()[::2])
        plt.savefig('Running_Times.pdf', bbox_inches='tight') 
        plt.legend()


        fig3= plt.figure('Average Num Actions')
        plt.title(r'Average Number of Actions Left vs. Refinement Step ($\phi_{1}$ Synthesis, Finite Inputs)', fontsize = 14)
        plt.plot(Avg_Num_Actions, label = r'Average Number of Actions')
        ax = plt.gca()
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        ax.set(ylim=(0, 5.0))
        ax.set_xlabel(r'Refinement Step', fontsize=17)
        ax.set_ylabel(r'Number of Actions', fontsize=17)
        ax.set_xticks(ax.get_xticks()[::2])
        plt.savefig('Average_Act.pdf', bbox_inches='tight') 
        plt.legend()    
        
        
        
        with open('State_Space.pkl','wb') as f:
            pickle.dump(State_Space, f)
        with open('Running_Times.pkl','wb') as f:           
            pickle.dump(Running_Times, f)
        with open('Suboptimality_Factors.pkl','wb') as f:             
            pickle.dump(Suboptimality_Factors, f)
        with open('Avg_Num_Actions.pkl','wb') as f:      
            pickle.dump(Avg_Num_Actions, f)
        with open('List_Max_Opt.pkl','wb') as f:     
            pickle.dump(List_Max_Opt, f)
        with open('List_Avg_Opt.pkl','wb') as f:     
            pickle.dump(List_Avg_Opt, f)
        with open('Fractions_Above.pkl','wb') as f:     
            pickle.dump(Fractions_Above, f)
        with open('Low_Bound.pkl','wb') as f:     
            pickle.dump(Low_Bound, f)
        with open('Upp_Bound.pkl','wb') as f:     
            pickle.dump(Upp_Bound, f)                    
        with open('Optimal_Policy.pkl','wb') as f:      
            pickle.dump(Optimal_Policy, f)
            
else:

       print 'Final Suboptimality Factor'
       print Greatest_Suboptimality_Factor

print 'TOTAL RUN TIME'
print timeit.default_timer() - start








#''' CONTINUOUS-INPUT CASE STUDY ''' UNCOMMENT BELOW TO RUN CONTINUOUS INPUT SYNTHESIS   




##Compute the rectangular reachable set for 0 input
#
#Inputs = np.zeros((1,State_Space.shape[0]))
#Reachable_Sets = func.Reachable_Sets_Computation_Finite(State_Space, Inputs)
#Reachable_Sets = list(Reachable_Sets[0])
#
##Below, we will compute the set of actions for the qualitative problem for each state in the state space
#
#Reachable_States_Cont = [[] for x in range(State_Space.shape[0])] #Contains the set of states which are reachable under some input for all states
#Discrete_Actions = list([]) #Contains all inputs for each state
#
#for i in range(State_Space.shape[0]):
#   
#        
#    List_Overlaps = list([])
#    Maybe_Count = list([]) #Counts how many maybe trigger regions are within one overlap
#    On_States_List = list([list([]),list([]), list([]) ]) #Keeps track of all On states in each overlap, we originally have 3 overlaps
#    Maybe_States_List = list([list([]),list([]), list([])]) #Keeps Track of all Maybe States in each overlap, we originally have 3 overlaps
#    Discrete_Actions.append(list([]))
#    
#    for j in range(State_Space.shape[0]):
#        Target = copy.deepcopy(State_Space[j])
#        New_Trigger_Regions = list(func.Compute_Trigger_Regions(Reachable_Sets[i], Input_Qualitative[i], Target,j))
#
#        if len(New_Trigger_Regions[0]) > 0 or len(New_Trigger_Regions[1]) > 0:
#            Reachable_States_Cont[i].append(j)
#        #We have to compute all the Overlaps below 
#        
#        if j > 0:
#                        
#            Original_Length = len(List_Overlaps)
#            
#            for k in range(Original_Length):
#
#                New_Overlaps = list([list([]),list([]),list([])]) #First Position: OVERLAP WITH ON TRIGGER REGION, #Second position: OVERLAP WITH MAYBE TRIGGER REGION, #Third position: OVERLAP WITH OFF TRIGGER REGION
#                New_Maybe_States_List = list(Maybe_States_List[0])
#                New_Maybe_States_List.append(j)
#
#                New_On_States_List = list(On_States_List[0])
#                New_On_States_List.append(j)
#                
#               
#                for x in range(len(List_Overlaps[0])):
#                    
#                    
#                    Overlap_Polygon = Polygon([(List_Overlaps[0][x][0][0], List_Overlaps[0][x][0][1]), (List_Overlaps[0][x][0][0], List_Overlaps[0][x][1][1]), (List_Overlaps[0][x][1][0], List_Overlaps[0][x][1][1]), (List_Overlaps[0][x][1][0], List_Overlaps[0][x][0][1])])
#                    Overlap_Polygon_Subtract = Polygon([(List_Overlaps[0][x][0][0], List_Overlaps[0][x][0][1]), (List_Overlaps[0][x][0][0], List_Overlaps[0][x][1][1]), (List_Overlaps[0][x][1][0], List_Overlaps[0][x][1][1]), (List_Overlaps[0][x][1][0], List_Overlaps[0][x][0][1])])             
#                        
#                    
#                    #BELOW, WE ASSUME THAT OVERLAPS WHICH ONLY SHARE A POINT OR A LINE DO NOT INTERSECT // NOT SURE ABOUT CORRECTNESS BUT SIMPLIFYING ASSUMPTION
#                    Has_Intersected = 0
#                    
#                    if Overlap_Polygon_Subtract.is_valid == False:
#                        continue
#                    
#                    for y in range(len(New_Trigger_Regions[0])): 
#                        
#                        Trigger_Poly = Polygon([(New_Trigger_Regions[0][y][0][0], New_Trigger_Regions[0][y][0][1]), (New_Trigger_Regions[0][y][0][0], New_Trigger_Regions[0][y][1][1]), (New_Trigger_Regions[0][y][1][0], New_Trigger_Regions[0][y][1][1]), (New_Trigger_Regions[0][y][1][0], New_Trigger_Regions[0][y][0][1])])
#                        Intersect = Trigger_Poly.intersection(Overlap_Polygon)
#                        if Intersect.is_empty != 1 and Intersect.geom_type == 'Polygon' and Overlap_Polygon_Subtract.is_valid == True:
#                            Has_Intersected = 1
#                            New_Overlaps[0].append(list([[Intersect.bounds[0], Intersect.bounds[1]],[Intersect.bounds[2], Intersect.bounds[3]]]))
#                            Overlap_Polygon_Subtract = Overlap_Polygon_Subtract.difference(Intersect)
#                        
#                    if Overlap_Polygon_Subtract.is_empty == 1 or Overlap_Polygon_Subtract.is_valid == False: #If it is empty, then the entire polygon has been overlapped already
#                        continue
#                                        
#                    for y in range(len(New_Trigger_Regions[1])):   
#  
#                        Trigger_Poly = Polygon([(New_Trigger_Regions[1][y][0][0], New_Trigger_Regions[1][y][0][1]), (New_Trigger_Regions[1][y][0][0], New_Trigger_Regions[1][y][1][1]), (New_Trigger_Regions[1][y][1][0], New_Trigger_Regions[1][y][1][1]), (New_Trigger_Regions[1][y][1][0], New_Trigger_Regions[1][y][0][1])])
#                        Intersect = Trigger_Poly.intersection(Overlap_Polygon)
#                        if Intersect.is_empty != 1 and Intersect.geom_type == 'Polygon' and Overlap_Polygon_Subtract.is_valid == True:
#                            Has_Intersected = 1
#                            New_Overlaps[1].append(list([[Intersect.bounds[0], Intersect.bounds[1]],[Intersect.bounds[2], Intersect.bounds[3]]]))
#                            Overlap_Polygon_Subtract = Overlap_Polygon_Subtract.difference(Intersect)
#
#                                                                  
#                    if Overlap_Polygon_Subtract.is_empty == 1 or Overlap_Polygon_Subtract.is_valid == False:
#                        continue
#               
#                    
#                    if Has_Intersected == 1:
#                    
#                        if Overlap_Polygon_Subtract.geom_type == 'MultiPolygon':
#                            
#                            for pol in Overlap_Polygon_Subtract:  # same for multipolygon.geoms
#                                
#                                list_r = list([[]])
#                                poly_r = shapely.geometry.polygon.orient(pol) 
#                                
#                                for y in range(len(poly_r.exterior.coords)):
#                                    list_r[0].append(list(poly_r.exterior.coords[y]))
#                                
#                                
#                                runtime = get()
#                                ctx = runtime.compile('''
#                                    module.paths.push('%s');
#                                    var decompose = require('rectangle-decomposition'); 
#                                    function decompose_region(region){
#                                    var rectangles = decompose(region)
#                                    return rectangles;
#                                    }
#                                    
#                                ''' % os.path.join(os.path.dirname(__file__),'node_modules'))
#                                part = ctx.call("decompose_region",list_r)
#    
#                                
#                                for x in range(len(part)):
#                                    New_Overlaps[2].append( list([ [ part[x][0][0], part[x][0][1] ] , [ part[x][1][0], part[x][1][1] ] ]) )
#    
#                            
#                            #PARSE RESPONSE AND STORE IN NEW_OVERLAPS
#                            
#                                
#                            
#                        elif Overlap_Polygon_Subtract.geom_type == 'Polygon':    
#                            
#                            start5 = timeit.default_timer() 
#    
#                            list_r = list([[]])
#                            
#                            Overlap_Polygon_Subtract = shapely.geometry.polygon.orient(Overlap_Polygon_Subtract)
#                                
#                            for y in range(len(Overlap_Polygon_Subtract.exterior.coords)):
#                                list_r[0].append(list(Overlap_Polygon_Subtract.exterior.coords[y])) #NEED TO STORE VERTICES IN COUNTERCLOCKWISE DIRECTION
#                            
#                            runtime = get()
#                            ctx = runtime.compile('''
#                                module.paths.push('%s');
#                                var decompose = require('rectangle-decomposition'); 
#                                function decompose_region(region){
#                                var rectangles = decompose(region)
#                                return rectangles;
#                                }
#                                
#                            ''' % os.path.join(os.path.dirname(__file__),'node_modules'))
#                            
#    
#                            part = ctx.call("decompose_region",list_r)
#                            
#                            
#                            for x in range(len(part)):
#                                New_Overlaps[2].append(list([[part[x][0][0], part[x][0][1]],[part[x][1][0], part[x][1][1]]]))
#    
#                            
#                    else:
#                        New_Overlaps[2].append(list(List_Overlaps[0][x]))
#                                       
#                for x in range(len(New_Overlaps)):
#                    if len(New_Overlaps[x]) != 0:
#                        List_Overlaps.append(New_Overlaps[x])
#                        if x == 0:                            
#                            On_States_List.append(New_On_States_List)
#                            Maybe_States_List.append(Maybe_States_List[0])
#                        elif x == 1:   
#                            On_States_List.append(On_States_List[0])
#                            Maybe_States_List.append(New_Maybe_States_List) 
#                        else:
#                            On_States_List.append(On_States_List[0])
#                            Maybe_States_List.append(Maybe_States_List[0])
#                
#                List_Overlaps.pop(0)
#                Maybe_States_List.pop(0)
#                On_States_List.pop(0)
#            
#
#                
#                                
#        else:
#            
#            for k in range(len(New_Trigger_Regions)):
#                List_Overlaps.append(New_Trigger_Regions[k])
#            
#            On_States_List[0].append(j)
#            Maybe_States_List[1].append(j)
#              
#    Reach_l_x = Reachable_Sets[i][0][0]
#    Reach_u_x = Reachable_Sets[i][1][0]
#    Reach_l_y = Reachable_Sets[i][0][1]
#    Reach_u_y = Reachable_Sets[i][1][1]
#    
#                 
#    for j in range(len(List_Overlaps)): #Going through all overlaps and picking the approriate input                    
#        
#        if len(Maybe_States_List[j]) <= 1: #The overlap contains at most 1 maybe trigger region
#                    
#            Best_Norm = float('inf')
#            
#            for k in range(len(List_Overlaps[j])):
#                
#                 Current_Input = list([])
#                 for n in range(len(List_Overlaps[j][k][0])):
#                     if List_Overlaps[j][k][0][n] > 0:
#                        Current_Input.append(List_Overlaps[j][k][0][n])
#                     elif List_Overlaps[j][k][1][n] < 0:
#                        Current_Input.append(List_Overlaps[j][k][1][n])
#                     else:
#                        Current_Input.append(0.0) 
#                     
#                 Cur_Norm = np.linalg.norm(np.asarray(Current_Input))
#                 
#                 if Cur_Norm < Best_Norm:
#                     Chosen_Input = list(Current_Input)
#                     Best_Norm = Cur_Norm
#                 
#                 # Keeping the Input with smallest norm           
#            
#            
#            Discrete_Actions[-1].append(Chosen_Input) 
#                        
#        else:
#                        
#            O = list(On_States_List[j])
#            Y = list(Maybe_States_List[j])
#
#            K = list(Maybe_States_List[j])
#            K.sort()
#
#            if len(O) == 0:
#                L = []
#                Combinations_List = list(combinations(Y, len(Y)-1))                                
#                for k in range(len(Combinations_List)):
#                    List = list(Combinations_List[k])
#                    List.sort()
#                    L.append(List)
#            else:        
#                L = list([list(K)])
#
#            
#            m = 0
#            tag_loop = 0
#            Found_Sets = [] #Keeps track of the combinations which have been found
#            Found_Actions_Overlap = []
#
#            
#            while(tag_loop == 0): 
#                
#                
#                S = list(L[m])   
#                
#                for k in range(len(Found_Sets)):
#                    if len(set(S) - Found_Sets[k]) == 0:
#                        m += 1    
#                        if m == len(L): tag_loop = 1 
#                        continue
#                    
#                S_dif = list(set(Y) - set(S))                
#                
#                #Instead of solving for minimum u, which is too computationally expensive, we try to solve for a u that satisfies the conditions
#                
#                #List_Functions contains all the upper bound functions
#                List_Functions = list([])
#                List_Jac = list([])
#                
#                for k in range(len(O)):
#                    
#                    a_x = State_Space[O[k]][0][0] #Lower x bound of state O[k]
#                    b_x = State_Space[O[k]][1][0] #Upper x bound of state O[k]
#                    a_y = State_Space[O[k]][0][1] #Upper y bound of state O[k]
#                    b_y = State_Space[O[k]][1][1] #Upper y bound of state O[k]
#                    
#                    #Modifying the size of states on the boundary to account for maximum and minimum capacity
#                    
#                    if a_x == LOW_1:
#                        a_x = min(LOW_1 + U_MIN_1 + mus[0] - Semi_Width_1, a_x)
#                    
#                    elif b_x == UP_1:
#                        b_x = max(UP_1 + U_MAX_1 + mus[0] + Semi_Width_1, b_x)
#                        
#                    if a_y == LOW_2:
#                        a_y = min(LOW_2 + U_MIN_2 + mus[1] - Semi_Width_2, a_y)
#                        
#                    elif b_y == UP_2:
#                        b_y = max(UP_2 + U_MAX_2 + mus[1] + Semi_Width_2, b_y)
#
#                    List_Functions.append(lambda x, a_x= a_x, b_x = b_x, a_y = a_y , b_y= b_y, Reach_l_x = Reach_l_x, Reach_u_x = Reach_u_x, Reach_l_y = Reach_l_y, Reach_u_y = Reach_u_y: Upper_Bound_Func(x[0], x[1], a_x, b_x, a_y, b_y, Reach_l_x, Reach_u_x,Reach_l_y, Reach_u_y))
#                    List_Jac.append(lambda x, a_x= a_x, b_x = b_x, a_y = a_y , b_y= b_y, Reach_l_x = Reach_l_x, Reach_u_x = Reach_u_x, Reach_l_y = Reach_l_y, Reach_u_y = Reach_u_y: list(func.Upper_Bound_Func_Jac(x[0], x[1], a_x, b_x, a_y, b_y, Reach_l_x, Reach_u_x,Reach_l_y, Reach_u_y)))
#
#                
#                for k in range(len(S_dif)):  
#
#                    a_x = State_Space[S_dif[k]][0][0] 
#                    b_x = State_Space[S_dif[k]][1][0] 
#                    a_y = State_Space[S_dif[k]][0][1] 
#                    b_y = State_Space[S_dif[k]][1][1] 
#                    
#                                        
#                    #Modifying the size of states on the boundary to account for maximum and minimum capacity
#                    
#                    if a_x == LOW_1:
#                        a_x = min(LOW_1 + U_MIN_1 + mus[0] - Semi_Width_1, a_x)
#                    
#                    elif b_x == UP_1:
#                        b_x = max(UP_1 + U_MAX_1 + mus[0] + Semi_Width_1, b_x)
#                        
#                    if a_y == LOW_2:
#                        a_y = min(LOW_2 + U_MIN_2 + mus[1] - Semi_Width_2, a_y)
#                        
#                    elif b_y == UP_2:
#                        b_y = max(UP_2 + U_MAX_2 + mus[1] + Semi_Width_2, b_y)                    
#
#                    List_Functions.append(lambda x, a_x= a_x, b_x = b_x, a_y = a_y , b_y= b_y, Reach_l_x = Reach_l_x, Reach_u_x = Reach_u_x, Reach_l_y = Reach_l_y, Reach_u_y = Reach_u_y: Upper_Bound_Func(x[0], x[1], a_x, b_x, a_y, b_y, Reach_l_x, Reach_u_x,Reach_l_y, Reach_u_y))
#                    List_Jac.append(lambda x, a_x= a_x, b_x = b_x, a_y = a_y , b_y= b_y, Reach_l_x = Reach_l_x, Reach_u_x = Reach_u_x, Reach_l_y = Reach_l_y, Reach_u_y = Reach_u_y: list(func.Upper_Bound_Func_Jac(x[0], x[1], a_x, b_x, a_y, b_y, Reach_l_x, Reach_u_x,Reach_l_y, Reach_u_y)))
#                                      
#                    
#                f = lambda x: -sum(phi(x) for phi in List_Functions) 
#                
#                Already_Action = 0
#                for n in range(len(Found_Actions_Overlap)):                    
#                    if f(Found_Actions_Overlap[n]) <= -1.0:
#                        Already_Action = 1
#                        break
#                
#                if Already_Action == 1:
#                    Found_Sets.append(set(S))
#                    m += 1    
#                    if m == len(L): tag_loop = 1 
#                    continue
#                                       
#                f_jac = lambda x: np.array(np.sum(np.array([phi(x) for phi in List_Jac]), 0))
#                
#                #Loop over all squares in the List_Overlaps[j]
#                
#                Feasability = 0
#                
#                for k in range(len(List_Overlaps[j])):
#                    
# 
#                    Num_Points = 10
#                    XX, YY = np.mgrid[List_Overlaps[j][k][0][0]:List_Overlaps[j][k][0][1]:complex(0,Num_Points), List_Overlaps[j][k][1][0]:List_Overlaps[j][k][1][1]:complex(0,Num_Points)]
#                    positions = np.vstack([XX.ravel(), YY.ravel()])      
#                    loc_best = 0
#                    Best_Act_Loc =[random.uniform(List_Overlaps[j][k][0][0], List_Overlaps[j][k][1][0]),random.uniform(List_Overlaps[j][k][0][1], List_Overlaps[j][k][1][1])] 
#
#                    for www in range(len(positions)):
#                        res = minimize(f, [positions[0][www],positions[1][www]], method='L-BFGS-B', jac= f_jac, bounds=[(List_Overlaps[j][k][0][0],List_Overlaps[j][k][1][0]),(List_Overlaps[j][k][0][1], List_Overlaps[j][k][1][1])] )
#                    
#                        if -res.fun > loc_best:
#                            loc_best= -res.fun
#                            Best_Act_Loc = list([res.x[0], res.x[1]])
#                                       
#                    if loc_best >= 1.0:
#                        Feasability = 1    
#                        Discrete_Actions[-1].append(list(Best_Act_Loc))
#                        Found_Actions_Overlap.append(list(Best_Act_Loc))
#                        Found_Sets.append(set(S))
#                        break
#                                       
#                if Feasability == 0:     
#                    
#                    if len(S) >= 1: #NO NEED TO HAVE MORE COMBINATIONS WHEN THERE IS ONLY ONE STATE IN S
#                        Combinations_List = list(combinations(S, len(S)-1))
#                        
#                        for k in range(len(Combinations_List)):
#                            List = list(Combinations_List[k])
#                            List.sort()
#                            if List not in L:
#                                L.append(List)
#                                            
#                m += 1    
#                if m == len(L): tag_loop = 1 
#                
#
#                
##Below, we construct the IMC and the BMDP with the available actions (REDUNDANT, COULD BE DONE IN PREVIOUS SECTION. WILL BE IMPLEMENTED LATER)
#      
#
#
#Max_Num_Actions = len(max(Discrete_Actions,key=len)) #HAVE TO DO THIS BECAUSE CURRENT IMPLEMENTATION ASSUMES ALL STATES HAVE SAME NUMBER OF ACTIONS
#Reachable_Sets_Quanti, List_States_Per_Action = func.Reachable_Sets_Computation_Continuous(State_Space, Discrete_Actions, Max_Num_Actions);          
#(Lower_Bound_Matrix, Upper_Bound_Matrix, Reachable_States, Is_Bridge_State, Bridge_Transitions, Pre_States) = func.BMDP_Computation_Continuous(Reachable_Sets_Quanti, State_Space, Max_Num_Actions, List_States_Per_Action) 
#(IA1_l, IA1_u, Is_Accepting, Is_Non_Accepting, Which_Accepting_Pair, Which_Non_Accepting_Pair, Product_Reachable_States, Product_Is_Bridge_State, Product_Bridge_Transitions, Init) = func.Build_Product_BMDP_Continuous(Lower_Bound_Matrix, Upper_Bound_Matrix, Automata, L_mapping, Automata_Accepting, Reachable_States, Is_Bridge_State, Bridge_Transitions, Discrete_Actions)
# 
#IA1_l_array = np.zeros(((IA1_l['Num_Act']),IA1_l['Num_S'],IA1_l['Num_S']))
#IA1_u_array = np.zeros(((IA1_u['Num_Act']),IA1_u['Num_S'],IA1_u['Num_S'])) 
#
#for k in IA1_l.keys():
#    if isinstance(k,tuple):
#        IA1_l_array[k[0],k[1],k[2]] = IA1_l[k]
#    
#for k in IA1_u.keys():
#    if isinstance(k,tuple):
#        IA1_u_array[k[0],k[1],k[2]] = IA1_u[k] 
#        
#
#IA1_l = np.array(IA1_l_array)
#IA1_u = np.array(IA1_u_array)        
# 
#Objective_Prob = 0
#
##Allowable_Actions = []
##for i in range(len(State_Space)*len(Automata)):
##    Allowable_Actions.append(range(len(Discrete_Actions[i])))
#
#
#
#first = 1;
#Allowable_Action_Potential = list([]) #Actions that could make the state a potential BSCC
#Allowable_Action_Permanent = list([]) #Actions that could make the state a permanent BSCC
#Is_In_Permanent_Comp = np.zeros(IA1_l.shape[1]) #Has a 1 if the state is in a permanent 
#Is_In_Permanent_Comp = Is_In_Permanent_Comp.astype(int)
#List_Permanent_Accepting_BSCC = [] #Lists which will keeps track of all permanent accepting BSCCs
#List_Permanent_Non_Accepting_BSCC = list([])
#Previous_Accepting_BSCC = list([])
#Previous_Non_Accepting_BSCC = list([])
#
#
#if Objective_Prob == 0:
#    
#   Optimal_Policy = np.zeros(IA1_l.shape[1])
#   Optimal_Policy = Optimal_Policy.astype(int)
#   Optimal_Policy_Continuous = [[] for x in range(IA1_l.shape[1])] #Continuous the optimal action from the continuous set of actions
#
#
#   Potential_Policy = np.zeros(IA1_l.shape[1]) #Policy to generate the "best" best-case (maximize upper bound)
#   Potential_Policy = Potential_Policy.astype(int)
#   Potential_Policy_Continuous = [[] for x in range(IA1_l.shape[1])]
#                   
##Below, we solve the finite-mode input problem
#
#   Permanent_Losing_Components = []
#   Allowable_Actions = []
#   for i in range(len(Discrete_Actions)):
#       for k in range(len(Automata)):
#           Allowable_Actions.append(range(len(Discrete_Actions[i])))
#
#   (Greatest_Potential_Accepting_BSCCs, Greatest_Permanent_Accepting_BSCCs, Potential_Policy, Optimal_Policy, Allowable_Action_Potential, Allowable_Action_Permanent, first, Is_In_Permanent_Comp, List_Permanent_Accepting_BSCC, List_Potential_Accepting_BSCC, Which_Potential_Accepting_BSCC, Is_In_Potential_Accepting_BSCC, Bridge_Accepting_BSCC) = func.Find_Greatest_Accepting_BSCCs(IA1_l, IA1_u, Is_Accepting, Is_Non_Accepting, Which_Accepting_Pair, Which_Non_Accepting_Pair, Allowable_Action_Potential, Allowable_Action_Permanent, first, Product_Reachable_States, Product_Bridge_Transitions, Product_Is_Bridge_State, Automata_Accepting, Potential_Policy, Optimal_Policy, Is_In_Permanent_Comp, List_Permanent_Accepting_BSCC, Previous_Accepting_BSCC) # Will return greatest potential and permanent accepting bsccs
#  
#   Which_Potential_Accepting_BSCC = Which_Potential_Accepting_BSCC.astype(int)
#   (Greatest_Permanent_Winning_Component, Optimal_Policy, Is_In_Permanent_Comp, List_Potential_Accepting_BSCC, Is_In_Potential_Winning_Component, Bridge_Accepting_BSCC) = func.Find_Greatest_Winning_Components(IA1_l, IA1_u, Is_Accepting, Is_Non_Accepting, Which_Accepting_Pair, Which_Non_Accepting_Pair, Allowable_Actions, Product_Reachable_States, Product_Bridge_Transitions, Product_Is_Bridge_State, Automata_Accepting, Optimal_Policy, Is_In_Permanent_Comp, List_Permanent_Accepting_BSCC, List_Potential_Accepting_BSCC, Greatest_Permanent_Accepting_BSCCs, Is_In_Potential_Accepting_BSCC, Which_Potential_Accepting_BSCC, Bridge_Accepting_BSCC, len(Automata), Init, Automata_Accepting) # Will return greatest potential and permanent winning components
# 
#   Previous_Accepting_BSCC = list(List_Potential_Accepting_BSCC)
#   Allowable_Action_Permanent = copy.deepcopy(Allowable_Action_Potential) #Some potential actions could be permanent actions for certain states under refinement 
#
#
#   Greatest_Potential_Winning_Component = list([])
#   Potential_BSCCs = [item for sublist in List_Potential_Accepting_BSCC for item in sublist]
#
#      
#   for i in range(len(Greatest_Permanent_Winning_Component)):
#       Greatest_Potential_Winning_Component.append(Greatest_Permanent_Winning_Component[i])
##       Allowable_Actions[Greatest_Permanent_Accepting_BSCCs[i]] = list([Optimal_Policy[Greatest_Permanent_Accepting_BSCCs[i]]])
#       Optimal_Policy_Continuous[Greatest_Permanent_Winning_Component[i]] = list(Discrete_Actions[Greatest_Permanent_Winning_Component[i]/len(Automata)][Optimal_Policy[Greatest_Permanent_Winning_Component[i]]])
#
#   for i in range(len(Potential_BSCCs)):
#       Greatest_Potential_Winning_Component.append(Potential_BSCCs[i])
#
##MAYBE SAVE POTENTIAL ACTIONS AS WELL
#
#
#   #Using this for now until I actually look for the sink states
#   Is_In_Potential_Winning_Component = np.asarray(Is_In_Potential_Winning_Component).astype(int)
#   List_Potential_Winning_Components = copy.deepcopy(List_Potential_Accepting_BSCC)
#   Bridge_Winning_Components = copy.deepcopy(Bridge_Accepting_BSCC)
#   Bridge_Winning_Components = [[Bridge_Winning_Components[i]] for i in range(len(Bridge_Winning_Components))]
#   Which_Potential_Winning_Component = [[] for i in range(IA1_l.shape[1])]   
#   for i in range(len(Which_Potential_Accepting_BSCC)):
#       if Is_In_Potential_Winning_Component[i] == 1:
#           Which_Potential_Winning_Component[i].append([Which_Potential_Accepting_BSCC[i],0]) 
#       
#
##Below, we solve the quantitative problem
#    
#
#   State_Space_Extended = [] #WE HAVE TO MAKE THE BORDER VIRTUALLY LARGER TO ACCOUNT FOR CONTAINEDNESS
#   Reachable_Quantitative_Product = [[] for x in range(State_Space.shape[0])*len(Automata)] #Contains the reachable states in the product automaton
#   Pre_Quantitative_Product = [set([]) for x in range(State_Space.shape[0])*len(Automata)] #Contains the pre states in the product automaton   
#   Input_Quantitative_Product  = [[] for x in range(State_Space.shape[0])*len(Automata)] #Contains the available actions for each states of the product automaton
#   
#   # We create the extended State Space to account for the border extension , and compute the reachable states in the product
#   
#   for i in range(State_Space.shape[0]):
#       
#       Copy_list_state = copy.deepcopy(State_Space[i])
#       State_Space_Extended.append(list(Copy_list_state))
#       
#       if State_Space_Extended[-1][0][0] == LOW_1:
#           State_Space_Extended[-1][0][0] = min(LOW_1 + U_MIN_1 + mus[0] - Semi_Width_1, State_Space_Extended[-1][0][0])
#       
#       elif State_Space_Extended[-1][1][0] == UP_1:
#           State_Space_Extended[-1][1][0] = max(UP_1 + U_MAX_1 + mus[0] + Semi_Width_1, State_Space_Extended[-1][1][0])
#           
#       if State_Space_Extended[-1][0][1] == LOW_2:
#           State_Space_Extended[-1][0][1] = min(LOW_2 + U_MIN_2 + mus[1] - Semi_Width_2, State_Space_Extended[-1][0][1])
#                 
#       elif State_Space_Extended[-1][1][1] == UP_2:  
#           State_Space_Extended[-1][1][1] = max(UP_2 + U_MAX_2 + mus[1] + Semi_Width_2, State_Space_Extended[-1][1][1]) 
#              
#           
#   for j in range(State_Space.shape[0]):
#       for k in range(len(Reachable_States_Cont[j])):
#           for x in range(len(Automata)):
#               if k == 0:
#                   Input_Quantitative_Product[j*len(Automata)+x] = list(Input_Quantitative[j])
#               for y in range(len(Automata)):
#                   if L_mapping[Reachable_States_Cont[j][k]] in Automata[x][y]:
#                       Reachable_Quantitative_Product[j*len(Automata)+x].append(Reachable_States_Cont[j][k]*len(Automata)+y)
#                       Pre_Quantitative_Product[Reachable_States_Cont[j][k]*len(Automata)+y].add(j*len(Automata)+x)
#                       break
#
#
#     
#(Low_Bound, Low_Bounds_Prod, Worst_Markov_Chain, Optimal_Policy_Continuous) = func.Maximize_Lower_Bound_Reachability_Continuous(Greatest_Permanent_Winning_Component, State_Space_Extended, State_Space.shape[0], len(Automata), Reachable_Quantitative_Product, Init, Optimal_Policy_Continuous, Input_Quantitative_Product, Reachable_Sets, Pre_Quantitative_Product, Permanent_Losing_Components) # Maximizes Lower Bound
#
#for i in range(len(Low_Bounds_Prod)):
#    if Low_Bounds_Prod[i] == 1: #This state is a sink state that is a part of the greatest permanent winning component
#        Is_In_Permanent_Comp[i] = 1
#        Greatest_Potential_Winning_Component.append(i)
#
##The function below takes care of computing the optimality factor as well as removing the suboptimal sets of inputs
#
#(Upp_Bound, Upp_Bounds_Prod, Best_Markov_Chain, Potential_Policy_Continuous, List_Optimality_Factors, Input_Quantitative_Product) = func.Maximize_Upper_Bound_Reachability_Continuous(Greatest_Potential_Winning_Component, State_Space_Extended, State_Space.shape[0], len(Automata), Reachable_Quantitative_Product, Init, Potential_Policy_Continuous, Input_Quantitative_Product, Reachable_Sets, Low_Bounds_Prod, Pre_Quantitative_Product, Permanent_Losing_Components) # Maximizes Lower Bound
#
#States_Above_Threshold = list([])
#Greatest_Optimality_Factor = 0.0
#for j in range(len(List_Optimality_Factors)):
#   Greatest_Optimality_Factor = max(Greatest_Optimality_Factor, List_Optimality_Factors[j])
#   if List_Optimality_Factors[j] > Threshold_Uncertainty:
#       States_Above_Threshold.append(j)
#
#Fraction_Above.append(float(len(States_Above_Threshold))/float(len(List_Optimality_Factors)))
#
#List_Max_Opt = []
#List_Avg_Opt = []
#if Greatest_Optimality_Factor > Threshold_Uncertainty:
#        
#        List_Max_Opt.append(Greatest_Optimality_Factor)
#        List_Avg_Opt.append(sum(List_Optimality_Factors)/(len(List_Optimality_Factors)))
#        #BELOW, WE FIRST COMPUTE ALL PERMANENT AND POTENTIAL LOSING COMPONENTS OF THE WORST CASE MARKOV CHAIN FOR THE PURPOSE OF REFINEMENT
#        Greatest_Permanent_Winning_Component = list([])
#        Potential_Losing = list([])
#        Permanent_Losing = list([])
#        States_To_Delete = list([])
#        Is_In_Potential_Losing_Component = np.zeros(len(List_Optimality_Factors)) #Contains a 1 if the state is in a potential losing component
#        Is_In_Potential_Losing_Component = Is_In_Potential_Losing_Component.astype(int)
#        Indices = list([]) #Keeps track of the indices of the states in the original Markov Chain
#        for i in range(len(Low_Bounds_Prod)):
#           if Low_Bounds_Prod[i] == 0:  #If the upper_bound of a state is non-zero and a lower bound zero, then the state belongs a potential losing component
#               if Upp_Bounds_Prod[i] > 0:   
#                   Potential_Losing.append(i)
#                   Indices.append(i)
##                   Is_In_Potential_Losing_Component[i] = 1
#               else:
#                   Permanent_Losing.append(i)
#                   Is_In_Permanent_Comp[i] = 1
#                   Permanent_Losing_Components.append(i)
#                   Indices.append(i)
##                   States_To_Delete.append(i)
#           else:
#               if Low_Bounds_Prod[i] == 1: #This state is a sink state that is a part of the greatest permanent winning component
#                   Is_In_Permanent_Comp[i] = 1
#                   Greatest_Permanent_Winning_Component.append(i)
#               States_To_Delete.append(i)
#        
#        
#        Worst_Reduced1 = np.delete(Worst_Markov_Chain.todense(),States_To_Delete,axis=0)
#        Worst_Reduced = np.delete(np.array(Worst_Reduced1),States_To_Delete,axis=1)
#        Worst_Reduced = (Worst_Reduced > 0).astype(int)
#        
#        C,n = func.SSCC(Worst_Reduced)
#        List_Potential_Losing_Components = list([])
#        Bridge_Losing_Components = list([])
#        Which_Potential_Losing_Component = list([[] for x in range(Worst_Markov_Chain.shape[0])])
#        
#        Number_BSCCs = 0
#        for j in range(len(C)):
#           BSCC = 1
#           All_Permanent = 1
#           for k in range(len(C[j])): 
#               if Is_In_Permanent_Comp[Indices[C[j][k]]] == 0:
#                   All_Permanent = 0               
#               if sum(Worst_Reduced[C[j][k],C[j]]) < sum(Worst_Reduced[C[j][k],:]): #This means, if the state in the SCC leaks
#                   BSCC = 0
#                   break
#        
#           if All_Permanent == 1:
#               continue
#        
#           if BSCC == 1:
#               if Is_In_Potential_Losing_Component[Indices[C[j][0]]] == 1: #We only consider the  potential losing BSCCs for now, not the sink states towards permanent BSCCs
#                  List_Potential_Losing_Components.append(list([]))
#               
#                  Bridge_Losing_Components.append(list([]))
#                  Bridge_Losing_Components[-1].append(list([]))
#        
#                  for i in range(len(C[j])):
#                      List_Potential_Losing_Components[-1].append(Indices[C[j][i]])
#                      Which_Potential_Losing_Component[Indices[C[j][i]]].append([Number_BSCCs,0])
#                      if Is_In_Potential_Winning_Component[Indices[C[j][i]]] == 1:
#                          Bridge_Losing_Components[-1][0].append(Indices[C[j][i]])
#                      else:    
#                          for k in range(len(Reachable_Quantitative_Product[Indices[C[j][i]]])):
#                              if Worst_Markov_Chain[Indices[C[j][i]]][Reachable_Quantitative_Product[Indices[C[j][i]]][k]] == 0 and Best_Markov_Chain[Indices[C[j][i]]][Reachable_Quantitative_Product[Indices[C[j][i]]][k]] > 0: #Comparing the worst and best case Markov Chains to figure out the bridge states
#                                  Bridge_Losing_Components[-1][0].append(Indices[C[j][i]])
#        
#        
#           
#                  Gr = igraph.Graph.Adjacency(Worst_Reduced.tolist())
#                  R = [] #Contains all sink states w.r.t current BSCC
#                  for q in range(len(C[j])):                 
#                     Res = Gr.subcomponent(C[j][q], mode="IN")
#                     R2 = [x for x in Res if x not in R]
#                     R.extend(R2)
#                  R = list(set(R) - set(C[j])) #Removing the BSCC states from the set of states which can reach the BSCC    
#                  R.sort()
#                  Check_Disjoint_Graph = Worst_Reduced[R,:] #Create graph to see if sink stakes are disjoint
#                  Check_Disjoint_Graph = Check_Disjoint_Graph[:,R]
#                  N, label= connected_components(csgraph=csr_matrix(Check_Disjoint_Graph), directed=False, return_labels=True)
#                  Dis_Comp = [[] for x in range(N)]
#                  for k in range(len(label)):
#                     Dis_Comp[label[k]].append(k)
#                             
#                  for k in range(len(Dis_Comp)):
#                      
#                      Bridge_Losing_Components[-1].append([])
#                      for l in range(len(Bridge_Losing_Components[-1][0])): #Adding the bridge states of the BSCC to the bridge states of the component (because those could destroy the component under refinement)
#                          Bridge_Losing_Components[-1][-1].append(Bridge_Losing_Components[-1][0][l])
#                      for l in range(len(Dis_Comp[k])):
#                          List_Potential_Losing_Components[-1].append(Indices[R[Dis_Comp[k][l]]])
#                          Which_Potential_Losing_Component[Indices[R[Dis_Comp[k][l]]]].append([Number_BSCCs, k+1])
#                          if Is_In_Potential_Winning_Component[Indices[R[Dis_Comp[k][l]]]] == 1:
#                              Bridge_Losing_Components[-1][-1].append(Indices[R[Dis_Comp[k][l]]])
#                          else:    
#                              for y in range(len(Reachable_Quantitative_Product[Indices[R[Dis_Comp[k][l]]]])):
#                                   if Worst_Markov_Chain[Indices[R[Dis_Comp[k][l]]]][Reachable_Quantitative_Product[Indices[R[Dis_Comp[k][l]]]][y]] == 0 and Best_Markov_Chain[Indices[R[Dis_Comp[k][l]]]][Reachable_Quantitative_Product[Indices[R[Dis_Comp[k][l]]]][y]] > 0: #Comparing the worst and best case Markov Chains to figure out the bridge states
#                                       Bridge_Losing_Components[-1][-1].append(Indices[R[Dis_Comp[k][l]]])
#                          
#                              
#        
#                  Number_BSCCs += 1
#        
#        Success_Intervals = [[] for x in range(len(Low_Bound))]
#        Product_Intervals = [[] for x in range(len(Low_Bounds_Prod))]
#        
#        for i in range(len(Upp_Bound)):
#            
#            Success_Intervals[i].append(Low_Bound[i])
#            Success_Intervals[i].append(Upp_Bound[i])
#        
#        
#        for i in range(len(Upp_Bounds_Prod)):
#            
#            Product_Intervals[i].append(Low_Bounds_Prod[i])
#            Product_Intervals[i].append(Upp_Bounds_Prod[i])        
#        
#
#        Pre_States = []
#        
#
#        Running_Times.append(timeit.default_timer() - start)
#        (State_Space, Low_Bound, Upp_Bound, Low_Bounds_Prod, Upp_Bounds_Prod, Potential_Policy_Continuous, Optimal_Policy_Continuous, List_Optimality_Factors, Init, L_mapping, Input_Quantitative_Product, Greatest_Permanent_Winning_Component, Permanent_Losing_Components, List_Max_Opt, List_Avg_Opt, Worst_Markov_Chain, Best_Markov_Chain, Running_Times, States_Above_Threshold, Fraction_Above) = func.State_Space_Refinement_Continuous(State_Space, Threshold_Uncertainty, Greatest_Optimality_Factor, Objective_Prob, Potential_Policy_Continuous, Optimal_Policy_Continuous, Best_Markov_Chain, Worst_Markov_Chain, States_Above_Threshold, Success_Intervals, Product_Intervals, len(Automata), Automata, Automata_Accepting, L_mapping, Is_In_Permanent_Comp, Input_Qualitative, List_Potential_Winning_Components, Which_Potential_Winning_Component, Bridge_Winning_Components, Is_In_Potential_Winning_Component, List_Potential_Losing_Components, Which_Potential_Losing_Component, Bridge_Losing_Components, Is_In_Potential_Losing_Component, list(Reachable_States_Cont), Pre_States, Reachable_Quantitative_Product, List_Permanent_Accepting_BSCC, Input_Quantitative_Product, Greatest_Permanent_Winning_Component, Permanent_Losing_Components, List_Max_Opt, List_Avg_Opt, Running_Times, Fraction_Above)
#
#        print 'Final Suboptimality Factor'
#        print List_Max_Opt[-1]
#        func.State_Space_Plot(State_Space)
#        
#        fig1= plt.figure('Greatest and Average Suboptimality Factors')
#        plt.title(r'Precision Metrics vs. Refinement Step ($\phi_{1}$ Synthesis, Continuous Input)')
#        plt.plot(List_Max_Opt, label = 'Greatest Suboptimality Factor')
#        plt.plot(List_Avg_Opt, label = 'Average Suboptimality Factor')
#        plt.plot(Fraction_Above, label = r'Fraction of States above $\epsilon_{max}$')
#        ax = plt.gca()
#        plt.rc('xtick', labelsize=16)
#        plt.rc('ytick', labelsize=16)
#        ax.set_xlabel(r'Refinement Step', fontsize=17)
#        ax.set_ylabel(r'Controller Precision', fontsize=17)
#        ax.set(ylim=(0, 1.4))
#        plt.legend()
#        plt.savefig('Optimality_Factors.pdf', bbox_inches='tight') 
#
#        
#        fig2= plt.figure('Running Time')
#        plt.title(r'Cumulative Execution Time for $\phi_{1}$ Synthesis (Continuous Inputs)', fontsize = 14)
#        plt.plot(np.cumsum(Running_Times), label = 'Cumulative Execution Time')
#        ax = plt.gca()
#        plt.rc('xtick', labelsize=16)
#        plt.rc('ytick', labelsize=16)
#        ax.set_xlabel(r'Refinement Step', fontsize=17)
#        ax.set_ylabel(r'Time (s)', fontsize=17)
#        ax.set_xticks(ax.get_xticks()[::2])
#        plt.savefig('Running_Times.pdf', bbox_inches='tight') 
#        plt.legend()
#        
#        fig3= plt.figure('Initial Input Space')
#        plt.title(r'Initial Input Space for All States', fontsize = 22)
#        ax = plt.gca()
#        rect = Rectangle((Input_Space[0][0],Input_Space[0][1]),Input_Space[1][0]-Input_Space[0][0],Input_Space[1][1]-Input_Space[0][1],linewidth=1,edgecolor='k',facecolor='lightcoral')
#        ax.add_patch(rect)
#        ax.set_xlabel(r'$u_1$', fontsize=20)
#        ax.set_ylabel(r'$u_2$', fontsize=20)
#        ax.set(xlim=(Input_Space[0][0] - (Input_Space[1][0]-Input_Space[0][0])/6.0, Input_Space[1][0] + (Input_Space[1][0]-Input_Space[0][0])/6.0 ))
#        ax.set(ylim=(Input_Space[0][1] - (Input_Space[1][1]-Input_Space[0][1])/6.0, Input_Space[1][1] + (Input_Space[1][0]-Input_Space[0][1])/6.0 ))
#        ax.set_xticks(ax.get_xticks()[::2])
#        ax.set_yticks(ax.get_yticks()[::2])
#        plt.savefig('Initial_Input_Space.pdf', bbox_inches='tight')
##        
##        S_plot = 1 #Index of the state for which we want to plot the final input space
##
##        fig4= plt.figure('Final Input Space') 
##        plt.title(r'Final Input Space of State $[2.40625, 1.40625] \times [2,2]$ \\  Automaton State $s_0$  ($\phi_{1}$ Synthesis)', multialignment = 'center')
##        ax = plt.gca()
##        for i in range(len(Input_Quantitative_Product[S_plot])):
##            Input = list(Input_Quantitative_Product[S_plot][i])
##            rect = Rectangle((Input[0][0],Input[0][1]),Input[1][0]-Input[0][0],Input[1][1]-Input[0][1],linewidth=1,edgecolor='k',facecolor='lightcoral')
##            ax.add_patch(rect)
##        plt.plot(Optimal_Policy_Continuous[S_plot][0], Optimal_Policy_Continuous[S_plot][1], marker='o', markersize=7, color="red", label="nolegend")    
##        ax.set(xlim=(Input_Space[0][0] - (Input_Space[1][0]-Input_Space[0][0])/6.0, Input_Space[1][0] + (Input_Space[1][0]-Input_Space[0][0])/6.0 ))
##        ax.set(ylim=(Input_Space[0][1] - (Input_Space[1][1]-Input_Space[0][1])/6.0, Input_Space[1][1] + (Input_Space[1][0]-Input_Space[0][1])/6.0 ))
##        ax.set_xticks(ax.get_xticks()[::2])
##        ax.set_yticks(ax.get_yticks()[::2])
##        ax.set_xlabel(r'$u_1$', fontsize=20)
##        ax.set_ylabel(r'$u_2$', fontsize=20)
##        red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
##                          markersize=7, label='Computed Near-Optimal Input')        
##        plt.legend(numpoints=1, handles=[red_dot])
##        plt.savefig('Final_Input_Space.pdf', bbox_inches='tight')
#        
#        
#        with open('State_Space.pkl','wb') as f:
#            pickle.dump(State_Space, f)
#        with open('Running_Times.pkl','wb') as f:           
#            pickle.dump(Running_Times, f)
#        with open('List_Optimality_Factors.pkl','wb') as f:             
#            pickle.dump(List_Optimality_Factors, f)
#        with open('Input_Quantitative_Product.pkl','wb') as f:      
#            pickle.dump(Input_Quantitative_Product, f)
#        with open('List_Max_Opt.pkl','wb') as f:     
#            pickle.dump(List_Max_Opt, f)
#        with open('List_Avg_Opt.pkl','wb') as f:     
#            pickle.dump(List_Avg_Opt, f)
#        with open('Fraction_Above.pkl','wb') as f:     
#            pickle.dump(Fraction_Above, f)
#        with open('Low_Bound.pkl','wb') as f:     
#            pickle.dump(Low_Bound, f)
#        with open('Upp_Bound.pkl','wb') as f:     
#            pickle.dump(Upp_Bound, f)             
#        with open('Optimal_Policy_Continuous.pkl','wb') as f:      
#            pickle.dump(Optimal_Policy_Continuous, f)
#
#else:  
#       print 'Final Suboptimality Factor'
#       print Greatest_Optimality_Factor
#       
#       
#
#print 'TOTAL RUN TIME'
#print timeit.default_timer() - start
