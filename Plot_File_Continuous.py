#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:57:09 2019

@author: maxencedutreix
"""

import matplotlib.pyplot as plt
from matplotlib import rc
import pickle
import matplotlib.patches as patches
import matplotlib.lines as mlines
import os
import numpy as np
from scipy.sparse import csr_matrix
from matplotlib.patches import Rectangle



plt.close('all')



    
def State_Space_Plot_Color(Space, Y, N, M):
    
    
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    
    fig = plt.figure('Final State Space')
    plt.title(r'Verifying $\phi_{2}$ under Continuous Input Set Control Policy', fontsize=18)
    

    plt.xlim([0,4])
    plt.ylim([0,4])

    ax=plt.gca()
    
    
    
    for i in range(Space.shape[0]):      
        if i in N:
            
            pol = plt.Rectangle((Space[i][0][0], Space[i][0][1]), Space[i][1][0] - Space[i][0][0], Space[i][1][1] - Space[i][0][1], facecolor='red', edgecolor='k', linewidth = 0.02)
            
        elif i in Y:            
            pol = plt.Rectangle((Space[i][0][0], Space[i][0][1]), Space[i][1][0] - Space[i][0][0], Space[i][1][1] - Space[i][0][1], facecolor='green', edgecolor='k', linewidth = 0.02)
        
        else:           
            pol = plt.Rectangle((Space[i][0][0], Space[i][0][1]), Space[i][1][0] - Space[i][0][0], Space[i][1][1] - Space[i][0][1], facecolor='yellow', edgecolor='k', linewidth = 0.02)
        
        ax.add_artist(pol)
    
    ax1 = plt.gca()
    ax1.set_xlabel('$x_1$', fontsize=20)
    ax1.set_ylabel('$x_2$', fontsize=20)
    plt.savefig('results_color.pdf', bbox_inches='tight')

          
    return 1    

def State_Space_Plot(Space):
    
    
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
        
    fig = plt.figure('Final Refined Partition')
    plt.title(r'Final Partition for $\phi_{2}$ Synthesis (Continuous Input Set)', fontsize=18)

    plt.xlim([0,4])
    plt.ylim([0,4])

    ax=plt.gca()
    
      
    for i in range(Space.shape[0]):      
           
        pol = plt.Rectangle((Space[i][0][0], Space[i][0][1]), Space[i][1][0] - Space[i][0][0], Space[i][1][1] - Space[i][0][1], facecolor = 'white', edgecolor='k', linewidth = 0.4)      
        ax.add_artist(pol)
    
    ax1 = plt.gca()
    ax1.set_xlabel(r'$x_1$', fontsize=20)
    ax1.set_ylabel(r'$x_2$', fontsize=20)
    
    plt.savefig('results.pdf', bbox_inches='tight')
      
    return 1

Thresh = 0.80

Input_Space = list([[-0.05, -0.05], [0.05, 0.05]]) # Initial State of available inputs // CHANGE BOUNDS IN TAC_Synthesis file if modified


    
file = open('State_Space.pkl', 'rb')
State_Space = pickle.load(file)
file.close()
file = open('Running_Times.pkl', 'rb')
Running_Times = pickle.load(file)
file.close()
file = open('List_Optimality_Factors.pkl', 'rb')
List_Optimality_Factors = pickle.load(file)
file.close()
file = open('Input_Quantitative_Product.pkl', 'rb')
Input_Quantitative_Product = pickle.load(file)
file.close()
file = open('List_Max_Opt.pkl', 'rb')
List_Max_Opt = pickle.load(file)
file.close()
file = open('List_Avg_Opt.pkl', 'rb')
List_Avg_Opt = pickle.load(file)
file.close()
file = open('Fraction_Above.pkl', 'rb')
Fraction_Above = pickle.load(file)
file.close()
file = open('Low_Bound.pkl', 'rb')
Low_Bound = pickle.load(file)
file.close()
file = open('Upp_Bound.pkl', 'rb')
Upp_Bound = pickle.load(file)
file.close()
file = open('Optimal_Policy_Continuous.pkl', 'rb')
Optimal_Policy_Continuous = pickle.load(file)
file.close()
    
    
State_Space_Plot(State_Space)


fig1= plt.figure('Greatest and Average Suboptimality Factors')
plt.title(r'Optimality Metrics vs. Refinement Step ($\phi_{2}$ Synthesis, Continuous Input Set)', fontsize = 13)
plt.plot(List_Max_Opt, label = r'Greatest Suboptimality Factor')
plt.plot(List_Avg_Opt, label = r'Average Suboptimality Factor')
plt.plot(Fraction_Above, label = r'Fraction of States above $\epsilon_{max}$')
ax = plt.gca()
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
ax.set_xlabel(r'Refinement Step', fontsize=17)
ax.set_ylabel(r'Suboptimality Factor \epsilon', fontsize=17)
ax.set(ylim=(0, 1.4))
plt.legend()
plt.savefig('Optimality_Factors.pdf', bbox_inches='tight') 


fig2= plt.figure('Running Time')
plt.title(r'Cumulative Execution Time for $\phi_{2}$ Synthesis (Continuous Input Set)', fontsize = 15)
plt.plot(np.cumsum(Running_Times), label = r'Cumulative Execution Time')
ax = plt.gca()
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
ax.set_xlabel(r'Refinement Step', fontsize=17)
ax.set_ylabel(r'Time (s)', fontsize=17)
plt.legend(loc=2)
plt.savefig('Running_Times.pdf', bbox_inches='tight') 



fig3= plt.figure('Initial Input Space')
plt.title(r'Initial Input Space for All States', fontsize = 22)
ax = plt.gca()
rect = Rectangle((Input_Space[0][0],Input_Space[0][1]),Input_Space[1][0]-Input_Space[0][0],Input_Space[1][1]-Input_Space[0][1],linewidth=1,edgecolor='k',facecolor='lightcoral')
ax.add_patch(rect)
ax.set_xlabel(r'$u_1$', fontsize=20)
ax.set_ylabel(r'$u_2$', fontsize=20)
ax.set(xlim=(Input_Space[0][0] - (Input_Space[1][0]-Input_Space[0][0])/6.0, Input_Space[1][0] + (Input_Space[1][0]-Input_Space[0][0])/6.0 ))
ax.set(ylim=(Input_Space[0][1] - (Input_Space[1][1]-Input_Space[0][1])/6.0, Input_Space[1][1] + (Input_Space[1][0]-Input_Space[0][1])/6.0 ))
ax.set_xticks(ax.get_xticks()[::2])
ax.set_yticks(ax.get_yticks()[::2])
plt.savefig('Initial_Input_Space.pdf', bbox_inches='tight')
            
            

S_plot = 17388 #Index of the state for which we want to plot the final input space

fig4= plt.figure('Final Input Space') 
plt.title(r'Final Input Space of State $[2.8125, 2.84375] \times [1.484375, 1.5],$ \\' '\n'  r'Automaton State $s_0$  ($\phi_{2}$ Synthesis)', multialignment = 'center')
ax = plt.gca()
for i in range(len(Input_Quantitative_Product[S_plot])):
    Input = list(Input_Quantitative_Product[S_plot][i])
    rect = Rectangle((Input[0][0],Input[0][1]),Input[1][0]-Input[0][0],Input[1][1]-Input[0][1],linewidth=1,edgecolor='k',facecolor='lightcoral')
    ax.add_patch(rect)
plt.plot(Optimal_Policy_Continuous[S_plot][0], Optimal_Policy_Continuous[S_plot][1], marker='o', markersize=7, color="red", label="nolegend")    
ax.set(xlim=(Input_Space[0][0] - (Input_Space[1][0]-Input_Space[0][0])/6.0, Input_Space[1][0] + (Input_Space[1][0]-Input_Space[0][0])/6.0 ))
ax.set(ylim=(Input_Space[0][1] - (Input_Space[1][1]-Input_Space[0][1])/6.0, Input_Space[1][1] + (Input_Space[1][0]-Input_Space[0][1])/6.0 ))
ax.set_xticks(ax.get_xticks()[::2])
ax.set_yticks(ax.get_yticks()[::2])
ax.set_xlabel(r'$u_1$', fontsize=20)
ax.set_ylabel(r'$u_2$', fontsize=20)
red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                  markersize=7, label='Selected Input')        
plt.legend(numpoints=1, handles=[red_dot])
plt.savefig('Final_Input_Space.pdf', bbox_inches='tight')  

Yes_States = []
No_States = []
Maybe_States = []

for i in range(len(Upp_Bound)): 
    if Upp_Bound[i] <= Thresh:     
        No_States.append(i)
        
    elif Low_Bound[i] > Thresh:       
        Yes_States.append(i)
        
    else:
        Maybe_States.append(i)
        
#State_Space_Plot_Color(State_Space,Yes_States, No_States, Maybe_States )  