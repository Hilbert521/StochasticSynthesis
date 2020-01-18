#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 22:06:12 2019

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


plt.close('all')



if os.path.isdir("/usr/local/texlive/2018/bin"):
    os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2018/bin/x86_64-darwin'
    rc('text', usetex='True')
    rc('font', family='serif', serif = 'Helvetica')
    
def State_Space_Plot_Color(Space, Y, N, M):
    
    
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    

    fig = plt.figure('Final State Space')
    plt.title(r'Verifying $\phi_{1}$ under Finite-mode Switching Policy', fontsize=20)
    
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
    plt.title(r'Final Partition for $\phi_{1}$ Synthesis (Finite-mode)', fontsize=18)

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

    
file = open('State_Space.pkl', 'rb')
State_Space = pickle.load(file)
file.close()
file = open('Running_Times.pkl', 'rb')
Running_Times = pickle.load(file)
file.close()
file = open('Suboptimality_Factors.pkl', 'rb')
Suboptimality_Factors = pickle.load(file)
file.close()
file = open('Avg_Num_Actions.pkl', 'rb')
Avg_Num_Actions = pickle.load(file)
file.close()
file = open('List_Max_Opt.pkl', 'rb')
List_Max_Opt = pickle.load(file)
file.close()
file = open('List_Avg_Opt.pkl', 'rb')
List_Avg_Opt = pickle.load(file)
file.close()
file = open('Fractions_Above.pkl', 'rb')
Fractions_Above = pickle.load(file)
file.close()
file = open('Low_Bound.pkl', 'rb')
Low_Bound = pickle.load(file)
file.close()
file = open('Upp_Bound.pkl', 'rb')
Upp_Bound = pickle.load(file)
file.close()
file = open('Optimal_Policy.pkl', 'rb')
Optimal_Policy = pickle.load(file)
file.close()
    
    
State_Space_Plot(State_Space)


fig1= plt.figure('Greatest and Average Suboptimality Factors')
plt.title(r'Optimality Metrics vs. Refinement Step ($\phi_{1}$ Synthesis, Finite-mode)')
plt.plot(List_Max_Opt, label = r'Greatest Suboptimality Factor')
plt.plot(List_Avg_Opt, label = r'Average Suboptimality Factor')
plt.plot(Fractions_Above, label = r'Fraction of States above $\epsilon_{max}$')
ax = plt.gca()
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
ax.set_xlabel(r'Refinement Step', fontsize=17)
ax.set_ylabel(r'Suboptimality Factor \epsilon', fontsize=17)
ax.set(ylim=(0, 1.4))
ax.set(xlim=(0, len(List_Max_Opt) - 1))
plt.legend()
plt.savefig('Optimality_Factors.pdf', bbox_inches='tight') 


fig2= plt.figure('Running Time')
plt.title(r'Cumulative Execution Time for $\phi_{1}$ Synthesis (Finite-mode)', fontsize = 15)
plt.plot(np.cumsum(Running_Times), label = r'Cumulative Execution Time')
ax = plt.gca()
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
ax.set_xlabel(r'Refinement Step', fontsize=17)
ax.set_ylabel(r'Time (s)', fontsize=17)
ax.set(xlim=(0, len(List_Max_Opt) - 1))
plt.legend(loc=2)
plt.savefig('Running_Times.pdf', bbox_inches='tight') 


fig3= plt.figure('Average Num Actions')
plt.title(r'Average Number of Actions Left vs. Refinement Step ($\phi_{1}$, Finite-mode)', fontsize = 14)
plt.plot(Avg_Num_Actions, label = r'Average Number of Actions')
ax = plt.gca()
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
ax.set(ylim=(0, 5.0))
ax.set(xlim=(0, len(List_Max_Opt) - 1))
ax.set_xlabel(r'Refinement Step', fontsize=17)
ax.set_ylabel(r'Number of Actions', fontsize=17)
plt.legend() 
plt.savefig('Average_Act.pdf', bbox_inches='tight')    

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
        
State_Space_Plot_Color(State_Space,Yes_States, No_States, Maybe_States )  
#
List_above = []
for i in range(len(Suboptimality_Factors)):
    if Suboptimality_Factors[i] > 0.30 :
        List_above.append(i)


        