#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:15:44 2019

@author: maxencedutreix
"""
import matplotlib as mpl
mpl.use('pdf')
import numpy as np
import matplotlib.pyplot as plt
import timeit
import sys
from shapely.geometry import Polygon #NEED TO INSTALL SHAPELY
from shapely import ops
from math import sqrt
from math import erf
from math import pi
from math import exp
from operator import add
from operator import sub
from itertools import combinations
import warnings
from networkx.algorithms import bipartite
import math
import scipy.sparse as sparse
import scipy.sparse.csgraph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from mpl_toolkits.mplot3d import Axes3D
import copy
import igraph
import bisect
from scipy.optimize import minimize
import random
from random import uniform
from shapely.ops import cascaded_union
import shapely
import os
from execjs import get
from joblib import Parallel, delayed
from matplotlib import rc
import matplotlib.lines as mlines
from scipy.sparse import lil_matrix
import pickle
from blist import blist
from numpy.linalg import inv
from numpy.linalg import solve
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity
from numpy.linalg import norm
from numpy import mgrid
from numpy import vstack



sys.setrecursionlimit(10000)

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
Partition_Parameter = 2 #DECIDES HOW THE INPUTS ARE PARTITIONED FOR THE UPPER BOUND REACHABILITY MAXIMIZATION FOR CONTINUOUS INPUTS. E.G, 2 means that the square is partitioned in 2 in both dimensions
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
Initial_Num_Points = 11
Min_Num_Points = 3
Count_Cut_Off = 40 #Value iteration algorithm for continuous set of inputs can get stuck in infinite loop due to numerical errors and possible non global solution to optimization problem. This variable prevents this
Num_Ref = 20 #number of refinement steps for continuous set of inputs
Num_Ref_Dis = 20 #number of refinement steps for continuous set of inputs
Lim_Qualitative_Ref = 5

Init_Volume = (U_MAX_1 - U_MIN_1)*(U_MAX_2 - U_MIN_2)



Time_Step = 0.05

def Initial_Partition_Plot(Space):
    
    #Plots the initial state space before verification/synthesis/refinement
       
    fig = plt.figure('Partition P')

    plt.title(r'Initial Partition for $\phi_{2}$ Synthesis', fontsize=22)  
    plt.plot([0, 4], [0, 0], color = 'k')
    plt.plot([0, 4], [1.0,1.0], color = 'k')
    plt.plot([0, 4], [2.0,2.0], color = 'k')
    plt.plot([0, 4], [3.0,3.0], color = 'k')
    plt.plot([0, 4], [4,4], color = 'k')
    plt.plot([0, 0], [0 ,4], color = 'k')
    plt.plot([1.0, 1.0], [0,4], color = 'k')
    plt.plot([2.0, 2.0], [0,4], color = 'k')
    plt.plot([3.0, 3.0], [0,4], color = 'k')
    plt.plot([4.0, 4.0], [0,4], color = 'k')
    
    ax = plt.gca()

    ax.text(0.42, 0.4, r'$A$', fontsize=27)
    ax.text(1.42, 1.4, r'$B$', fontsize=27)
    ax.text(3.42, 2.4, r'$B$', fontsize=27)    
    ax.text(3.42, 1.4, r'$B$', fontsize=27)
    ax.text(0.42, 3.4, r'$B$', fontsize=27)
    ax.text(2.42, 3.4, r'$C$', fontsize=27)      
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    ax.set_xlabel(r'$x_1$', fontsize=20)
    ax.set_ylabel(r'$x_2$', fontsize=20)
    
    return 1    



''' Computation of Reachable Sets for Discrete Mode Problem '''
def Reachable_Sets_Computation_Finite(State, u):
    
    #Compute the reachable sets for all states in a rectangular partition for mixed-monotone dynamics        
    R_sets = [[] for i in range(len(u))]
    a = 1.3
    b = 0.25
    low_1 = LOW_1
    low_2 = LOW_2
    up_1 = UP_1
    up_2 = UP_2
    Time_S = Time_Step
    
    for j, uj in enumerate(u):
        for i in State:
        
        
            R_set = [[0.0,0.0], [0.0,0.0]]
            x1 = i[0][0]
            y1 = i[0][1]
            x2 = i[1][0]
            y2 = i[1][1]
            
            R_set[0][0] = min(max(low_1,x1 + (-a*x1 + y1 )*Time_S + uj[0]), up_1)
            R_set[0][1] = min(max(low_2, y1 + (((x1**2)/((x1**2) + 1)) - b* y1 ) * Time_S + uj[1]), up_2)
            
            R_set[1][0] = min(max(low_1, x2 + (-a*x2 + y2 )*Time_S + uj[0]), up_1)
            R_set[1][1] = min(max(low_2, y2 + (((x2**2)/((x2**2) + 1)) - b* y2 ) * Time_S + uj[1]), up_2)
            
            R_sets[j].append(R_set)

    return R_sets



''' Computation of Reachable Sets for Continuous Input Problem '''
def Reachable_Sets_Computation_Continuous(State, u, Max_Num):
    
    
    
    #Compute the reachable sets for all states in a rectangular partition for mixed-monotone dynamics        
    
    #We make R_sets the length of the longest set of inputs, to accomodate the rest of the code unfortunately 

    R_sets = [[[] for k in range(len(State))] for i in range(Max_Num)]
    List_States_Per_Action = [[] for i in range(Max_Num)] #List the set of states which have a given action index
    a = 1.3
    b = 0.25
    
    for i, Sta in enumerate(State):
        for j, u_j in enumerate(u[i]):
        
            R_set = [[0,0],[0,0]]    
            R_set[0][0] = min(max(LOW_1,Sta[0][0] + (-a*Sta[0][0] + Sta[0][1] )*Time_Step + u_j[0]), UP_1)
            R_set[0][1] = min(max(LOW_1, Sta[0][1] + (((Sta[0][0]**2)/((Sta[0][0]**2) + 1)) - b* Sta[0][1] ) * Time_Step + u_j[1]), UP_1)
            
            R_set[1][0] = min(max(LOW_1, Sta[1][0] + (-a*Sta[1][0] + Sta[1][1] )*Time_Step + u_j[0]), UP_1)
            R_set[1][1] = min(max(LOW_1, Sta[1][1] + (((Sta[1][0]**2)/((Sta[1][0]**2) + 1)) - b* Sta[1][1] ) * Time_Step + u_j[1]), UP_1)
            
            R_sets[j][i].append(list(R_set))
            List_States_Per_Action[j].append(i)

    return R_sets, List_States_Per_Action





def Compute_Trigger_Regions(Reachable_Set, All_Inputs, Target, Tag):

    
    #We first compute the sets of inputs in R^2 which create the on and maybe trigger regions  
    
    #Here we compute the input that places the reachable set at the center of the box and the maximum deltas for the other states to generate the on trigger region
    
    centering_input = list([])
    delta_on = list([])
    Tag1 = 0
    
    #Adjusting the size of the Target Set if on the boundary of the domain
    
    if Target[0][0] == LOW_1:
        Target[0][0] = min(LOW_1 + U_MIN_1 + mus[0] - Semi_Width_1, Target[0][0])
     
    elif Target[1][0] == UP_1:
        Target[1][0] = max(UP_1 + U_MAX_1 + mus[0] + Semi_Width_1, Target[1][0])
        
        
    if Target[0][1] == LOW_2:
        Target[0][1] = min(LOW_2 + U_MIN_2 + mus[1] - Semi_Width_2, Target[0][1])
     
    elif Target[1][1] == UP_2: 
        Target[1][1] = max(UP_2 + U_MAX_2 + mus[1] + Semi_Width_2, Target[1][1])
    

    
    for i in range(Target[0].shape[0]):
        
        centering_input.append(((Target[0][i] + Target[1][i])/2.0) - mus[i] - ((Reachable_Set[0][i] + Reachable_Set[1][i])/2.0) ) # Input which puts the reachable set at the center of the target box state
        delta = Target[1][i] - (Reachable_Set[1][i] + centering_input[i] + w_list[i][0])
        if delta >= 0:
            delta_on.append(delta)
        else:
            Tag1 = 1
    # Tag is a function of the deltas
    
    if Tag1 == 0: 
        On_Trig = list([list([]), list([])])
        for i in range(len(centering_input)):
            On_Trig[0].append(centering_input[i] - delta_on[i])
            On_Trig[1].append(centering_input[i] + delta_on[i])
        

    else:
        On_Trig = [] #In this case, no input can create an always on transition to the Target_State
    
        
    # Now, we generate the Maybe Trigger Region
    
    Maybe_Trig = list([[list([]) , list([])]])
    

    #Case where there is no On Trigger Region, then the Maybe trigger region is a box
    if len(On_Trig) == 0:
        for i in range(len(centering_input)):
            delta = Target[1][i] - (Reachable_Set[0][i] + centering_input[i] + w_list[i][0])
            Maybe_Trig[0][0].append(centering_input[i] - delta)
            Maybe_Trig[0][1].append(centering_input[i] + delta)
       
       

        
    #When an On Trigger Region is present, then the Maybe trigger region is the substraction of two boxes, with the first one (On Region) contained inside the other    
    else:
        
        Orig_Maybe = list([list([]) , list([])])
        for i in range(len(centering_input)):
            delta = Target[1][i] - (Reachable_Set[0][i] + centering_input[i] + w_list[i][0])
            Orig_Maybe[0].append(centering_input[i] - delta)
            Orig_Maybe[1].append(centering_input[i] + delta)        
    
        #HARDCODE A PARTITION OF THE POLYGON INTO 4 RECTANGLES
        
        
        Maybe_Trig = list([])
        Maybe_Trig.append(list([list([Orig_Maybe[0][0], Orig_Maybe[0][1]]) , list([On_Trig[0][0], Orig_Maybe[1][1]])]))
        Maybe_Trig.append(list([list([On_Trig[0][0], On_Trig[1][1]]) , list([Orig_Maybe[1][0], Orig_Maybe[1][1]])]))
        Maybe_Trig.append(list([list([On_Trig[1][0], Orig_Maybe[0][1]]) , list([Orig_Maybe[1][0], On_Trig[1][1]])]))
        Maybe_Trig.append(list([list([On_Trig[0][0], Orig_Maybe[0][1]]) , list([On_Trig[1][0], On_Trig[0][1]])]))
        

        
    
    #HARD CODED FOR 2D
    
    # Below, we compute the intersections of the input space and the trigger regions in order to get the trigger regions with respect to available inputs
    
    Trigger_Regions = list([list([]) , list([]), list([])]) #First entry contains the ON trigger regions, the second contains the Maybe trigger region and the third contains the OFF trigger regions
         
    
    for i in range(len(All_Inputs)):  
         
        Input = list(All_Inputs[i])           
        Input_Polygon = Polygon([(Input[0][0], Input[0][1]), (Input[0][0], Input[1][1]), (Input[1][0], Input[1][1]), (Input[1][0], Input[0][1])])
        Input_Polygon_Subtract = Polygon([(Input[0][0], Input[0][1]), (Input[0][0], Input[1][1]), (Input[1][0], Input[1][1]), (Input[1][0], Input[0][1])]) #Used to keep track of the off trigger region
        
        #ON TRIGGER INTERSECTION
        
        

        if len(On_Trig) != 0:
            On_Trig_Poly = Polygon([(On_Trig[0][0], On_Trig[0][1]), (On_Trig[0][0], On_Trig[1][1]), (On_Trig[1][0], On_Trig[1][1]), (On_Trig[1][0], On_Trig[0][1])])
            
            Input_Polygon_Subtract = Input_Polygon_Subtract.difference(On_Trig_Poly) 
            Intersect = (Input_Polygon.intersection(On_Trig_Poly))


            if Intersect.is_empty != 1 and Intersect.geom_type == 'Polygon':
                Trigger_Regions[0].append(list([[Intersect.bounds[0], Intersect.bounds[1]],[Intersect.bounds[2], Intersect.bounds[3]]]))
        
        #MAYBE TRIGGER INTERSECTION
        
        for j in range(len(Maybe_Trig)):
            
            Maybe_Trig_Poly = Polygon([(Maybe_Trig[j][0][0], Maybe_Trig[j][0][1]), (Maybe_Trig[j][0][0], Maybe_Trig[j][1][1]), (Maybe_Trig[j][1][0], Maybe_Trig[j][1][1]), (Maybe_Trig[j][1][0], Maybe_Trig[j][0][1])])
            
            Input_Polygon_Subtract = Input_Polygon_Subtract.difference(Maybe_Trig_Poly) 
            Intersect = (Input_Polygon.intersection(Maybe_Trig_Poly))


            if Intersect.is_empty != 1 and Intersect.geom_type == 'Polygon':
                Trigger_Regions[1].append(list([[Intersect.bounds[0], Intersect.bounds[1]],[Intersect.bounds[2], Intersect.bounds[3]]]))        
        
        #OFF TRIGGER INTERSECTION/MAY NOT NEED TO EXPLICITLY COMPUTE IT
        

        

        
        if Tag == 0: ## If this is not the first state, we don't need to explicitely compute the OFF trigger region, since it is just the complement of the other 2

            if len(Trigger_Regions[0]) == 0 and len(Trigger_Regions[1]) == 0:
                Trigger_Regions[2].append(list([list([Input[0][0], Input[0][1]]) , list([Input[1][0], Input[1][1]])]))
            
           
            else:
                

                
                if Input_Polygon_Subtract.geom_type != 'MultiPolygon' :
                
                    
                    if Input_Polygon_Subtract.is_empty != 1:
                        
                        #NEED TO PARTITION THIS INTO RECTANGLES, HARDCODED FOR 2D
                        
                        
                        #IF THE POLYGON DOES NOT HAVE AN INTERIOR
                        if len(Input_Polygon_Subtract.interiors) == 0:
                            Interior_Line = Input_Polygon_Subtract.exterior.difference(Input_Polygon.exterior).simplify(0)

                            
                            
                            if Interior_Line.geom_type == 'MultiLineString':
                                Interior_Line = ops.linemerge(Interior_Line)
                           
#                            

                            
                            if len(Interior_Line.coords[:]) == 3: #INTERSECTION IS IN A CORNER
                                
                                diff = np.subtract(Interior_Line.bounds,Input_Polygon.bounds)
                                ind =  [index for index, value in enumerate(diff) if value == 0.0]
                                ind.sort()
                                
                                
                                
                                if ind[0] == 0 and ind[1] == 1:
                                    
                                    Trigger_Regions[2].append(list([list([Interior_Line.bounds[0], Interior_Line.bounds[3]]) , list([Interior_Line.bounds[2], Input[1][1]])]))
                                    Trigger_Regions[2].append(list([list([Interior_Line.bounds[2], Input[0][1]]) , list([Input[1][0], Input[1][1]])]))
        
                                elif ind[0] == 0 and ind[1] == 3:
        
                                    Trigger_Regions[2].append(list([list([Input[0][0], Input[0][1]]) , list([Interior_Line.bounds[2], Interior_Line.bounds[1]])]))
                                    Trigger_Regions[2].append(list([list([Interior_Line.bounds[2], Input[0][1]]) , list([Input[1][0], Input[1][1]])]))
                                                   
                                elif ind[0] == 1 and ind[1] == 2:
        
                                    Trigger_Regions[2].append(list([list([Input[0][0], Input[0][1]]) , list([Interior_Line.bounds[0], Input[1][1]])]))
                                    Trigger_Regions[2].append(list([list([Interior_Line.bounds[0], Interior_Line.bounds[3]]) , list([Input[1][0], Input[1][1]])]))
        
                                elif ind[0] == 2 and ind[1] == 3:
        
                                    Trigger_Regions[2].append(list([list([Input[0][0], Input[0][1]]) , list([Interior_Line.bounds[0], Input[1][1]])]))
                                    Trigger_Regions[2].append(list([list([Interior_Line.bounds[0], Input[0][0]]) , list([Interior_Line.bounds[2], Interior_Line.bounds[1]])]))
                                     
                                    
                                
                            else: #INTERSECTION IS ON A SIDE
                                
                                ind = tuple(np.subtract(Interior_Line.bounds,Input_Polygon.bounds)).index(0.0)
                                
                                if ind == 0:
                                    
                                    Trigger_Regions[2].append(list([list([Input[0][0], Input[0][1]]) , list([Interior_Line.bounds[2], Interior_Line.bounds[1]])]))
                                    Trigger_Regions[2].append(list([list([Input[0][0], Interior_Line.bounds[3]]) , list([Interior_Line.bounds[2], Input[1][1]])]))
                                    Trigger_Regions[2].append(list([list([Interior_Line.bounds[2], Input[0][1]]) , list([Input[1][0], Input[1][1]])]))
                                                              
                                elif ind == 1: 
                                    
                                    Trigger_Regions[2].append(list([list([Input[0][0], Input[0][1]]) , list([Interior_Line.bounds[0], Input[1][1]])]))
                                    Trigger_Regions[2].append(list([list([Interior_Line.bounds[0], Interior_Line.bounds[3]]) , list([Interior_Line.bounds[2], Input[1][1]])]))
                                    Trigger_Regions[2].append(list([list([Interior_Line.bounds[2], Input[0][1]]) , list([Input[1][0], Input[1][1]])]))
                                                             
                                elif ind == 2:
                                    
                                    Trigger_Regions[2].append(list([list([Input[0][0], Input[0][1]]) , list([Interior_Line.bounds[0], Input[1][1]])]))
                                    Trigger_Regions[2].append(list([list([Interior_Line.bounds[0], Interior_Line.bounds[3]]) , list([Interior_Line.bounds[2], Input[1][1]])]))
                                    Trigger_Regions[2].append(list([list([Interior_Line.bounds[0], Input[0][1]]) , list([Input[1][0], Interior_Line.bounds[1]])]))
                                                                     
                                    
                                else: 
                                    
                                    Trigger_Regions[2].append(list([list([Input[0][0], Input[0][1]]) , list([Interior_Line.bounds[0], Input[1][1]])]))
                                    Trigger_Regions[2].append(list([list([Interior_Line.bounds[0], Input[0][1]]) , list([Interior_Line.bounds[2], Interior_Line.bounds[1]])]))
                                    Trigger_Regions[2].append(list([list([Interior_Line.bounds[2], Input[0][1]]) , list([Input[1][0], Input[1][1]])]))
                                                               
                                    
                                    
                        else:
                            xmin = Orig_Maybe[0][0]
                            ymin = Orig_Maybe[0][1]
                            xmax = Orig_Maybe[1][0]
                            ymax = Orig_Maybe[1][1]
                                          
                        
                            Trigger_Regions[2].append(list([list([Input[0][0], Input[0][1]]) , list([xmin, Input[1][1]])]))
                            Trigger_Regions[2].append(list([list([xmin, ymax]) , list([Input[1][0], Input[1][1]])]))
                            Trigger_Regions[2].append(list([list([xmax, Input[0][1]]) , list([Input[1][0], ymax])]))
                            Trigger_Regions[2].append(list([list([xmin, Input[0][1]]) , list([xmax, ymin])]))
                

                
                else:
                    
                    Trigger_Regions[2].append([[Input_Polygon_Subtract[0].bounds[0], Input_Polygon_Subtract[0].bounds[1]], [Input_Polygon_Subtract[0].bounds[2],Input_Polygon_Subtract[0].bounds[3]]])
                    Trigger_Regions[2].append([[Input_Polygon_Subtract[1].bounds[0], Input_Polygon_Subtract[1].bounds[1]], [Input_Polygon_Subtract[1].bounds[2],Input_Polygon_Subtract[1].bounds[3]]])
            
    return Trigger_Regions







def BMDP_Computation(R_Set, Target_Set, u):
    
    #Computes the lower and upper bound probabilities of transition from state
    #to state using the reachable sets in R_set and the target sets in Target_Set
       
    
    Lower = blist([ blist([ blist([[0]*Target_Set.shape[0] for i in range(Target_Set.shape[0])]) for j in range(Target_Set.shape[0])]) for k in range(len(u))])
    Upper = blist([ blist([ blist([[0]*Target_Set.shape[0] for i in range(Target_Set.shape[0])]) for j in range(Target_Set.shape[0])]) for k in range(len(u))])
    
    Reachable_States = [[[] for x in range(Target_Set.shape[0])] for y in range(len(u))] # 
    Pre_States = [[[] for x in range(Target_Set.shape[0])] for y in range(len(u))]
    Is_Bridge_State = np.zeros((len(u), Target_Set.shape[0]))
    Bridge_Transitions = [[[] for x in range(Target_Set.shape[0])] for y in range(len(u))] # list is more dynamic.
    

    Z1 = (erf(Semi_Width_1/sigma1)/sqrt(2)) - (erf(-Semi_Width_1/sigma1)/sqrt(2))
    Z2 = (erf(Semi_Width_2/sigma2)/sqrt(2)) - (erf(-Semi_Width_2/sigma2)/sqrt(2))
    
    
           
    for z in range(len(R_Set)): # For number of modes
        
        for j in range(len(R_Set[z])): # For number of reachable sets
            
            
            
            r0 = R_Set[z][j][0][0]
            r1 = R_Set[z][j][1][0]
            r2 = R_Set[z][j][0][1]
            r3 = R_Set[z][j][1][1]

            
            for h in range(len(Target_Set)):
    
                         
                q0 = Target_Set[h][0][0]
                q1 = Target_Set[h][1][0]
                q2 = Target_Set[h][0][1]
                q3 = Target_Set[h][1][1]               
    
                if q0 == LOW_1 and r0 + mu1 - Semi_Width_1 < LOW_1:
                    q0 = r0 + mu1 - Semi_Width_1
                        
                if q1 == UP_1 and r1 + mu1 + Semi_Width_1 > UP_1:
                    q1 = r1 + mu1 + Semi_Width_1
                        
                if q2 == LOW_2 and r2 + mu2 - Semi_Width_2 < LOW_2:
                    q2 = r2 + mu2 - Semi_Width_2
                        
                if q3 == UP_2 and r3 + mu2 + Semi_Width_2 > UP_2:
                    q3 = r3 + mu2 + Semi_Width_2                          
                
# Virtually extend the state to take into account that goes outside of the domain. Attach it back. Part of our dynamics.                
                if (r0 >= q1 + Semi_Width_1 - mu1) or (r1 <= q0 - Semi_Width_1 - mu1) or (r2 >= q3 + Semi_Width_2 - mu2) or (r3 <= q2 - Semi_Width_2 - mu2):
                    Lower[z][j][h] = 0.0
                    Upper[z][j][h] = 0.0
                    continue
                
                Reachable_States[z][j].append(h)
                Pre_States[z][h].append(j)
                
                a1_Opt = ((q0 + q1)/2.0) - mu1
                a2_Opt = ((q2 + q3)/2.0) - mu2
                
                
                if (r1 < a1_Opt): 
                    a1_Max = r1
                    a1_Min = r0
                elif(r0 > a1_Opt): 
                    a1_Max = r0
                    a1_Min = r1
                else: 
                    a1_Max = a1_Opt       
                    if (a1_Opt <= (r1+r0)/2.0):
                        a1_Min = r1
                    else:
                        a1_Min = r0
                    
                    
                
                if (r2 > a2_Opt): 
                    a2_Max = r2
                    a2_Min = r3
                elif(r3 < a2_Opt): 
                    a2_Max = r3
                    a2_Min = r2
                else: 
                    a2_Max = a2_Opt
                    if (a2_Opt <= (r2+r3)/2.0):
                        a2_Min = r3
                    else:
                        a2_Min = r2
                                
                 
                    
  # If Gaussian is fully contained inside box, then upper bound is 1.          
                if a1_Max + mu1 - Semi_Width_1  > q0 and a1_Max + mu1 + Semi_Width_1 < q1 and a2_Max + mu2 - Semi_Width_2 > q2 and a2_Max + mu2 + Semi_Width_2 < q3:
                    H = 1.0
                else:
# Below, determining upper and lower bounds of INTEGRATION of the GAUSSIAN                    
                    if q0 < a1_Max + mu1 - Semi_Width_1:
                        b0 = a1_Max + mu1 - Semi_Width_1
                    else:
                        b0 = q0
                        
                    if q1 > a1_Max + mu1 + Semi_Width_1:
                        b1 = a1_Max + mu1 + Semi_Width_1
                    else:
                        b1 = q1
                        
                    if q2 < a2_Max + mu2 - Semi_Width_2:
                        b2 = a2_Max + mu2 - Semi_Width_2
                    else:
                        b2 = q2
                        
                    if q3 > a2_Max + mu2 + Semi_Width_2:
                        b3 = a2_Max + mu2 + Semi_Width_2
                    else:
                        b3 = q3    
                          
                        
                    # Perform integration
                    H = ( ( (erf((b1 - a1_Max - mu1)/sigma1)/sqrt(2)) - (erf((b0 - a1_Max - mu1)/sigma1)/sqrt(2)) ) / (Z1)) * (( (erf((b3 - a2_Max - mu2)/sigma2)/sqrt(2)) - (erf((b2 - a2_Max - mu2)/sigma2)/sqrt(2)) ) / ( Z2 ))
            # For numerical errors, if probability is greater than 1. For upper bound.
                    if H > 1:
                        H = 1.0
                        
                    
# Now we want to compute lower bound. If minimizer point is too far from the target state, based on if the edge of state is beyond than the edge of the Gaussian (mean + minimizer + Gaussian "half" width)                 
                if (a1_Min + mu1 + Semi_Width_1 <= q0) or (a1_Min + mu1 - Semi_Width_1 >= q1) or (a2_Min + mu2 + Semi_Width_2 <= q2) or (a2_Min + mu2 - Semi_Width_2 >= q3):                    

                    Is_Bridge_State[z][j] = 1 # Means TRUE
                    Bridge_Transitions[z][j].append(h)
                    Lower[z][j][h] = 0.0
                    Upper[z][j][h] = H
                    continue            # to next state    
                
                
                else:
                    
                    if q0 < a1_Min + mu1 - Semi_Width_1:
                        b0 = a1_Min + mu1 - Semi_Width_1
                    else:
                        b0 = q0
                        
                    if q1 > a1_Min + mu1 + Semi_Width_1:
                        b1 = a1_Min + mu1 + Semi_Width_1
                    else:
                        b1 = q1
                        
                    if q2 < a2_Min + mu2 - Semi_Width_2:
                        b2 = a2_Min + mu2 - Semi_Width_2
                    else:
                        b2 = q2
                        
                    if q3 > a2_Min + mu2 + Semi_Width_2:
                        b3 = a2_Min + mu2 + Semi_Width_2
                    else:
                        b3 = q3   
                                        
              # Perform integration for lower bound probability of transition, based on Gaussian overlap      
                    L = ( ( (erf((b1 - a1_Min - mu1)/sigma1)/sqrt(2)) - (erf((b0 - a1_Min - mu1)/sigma1)/sqrt(2)) ) / (Z1)) * (( (erf((b3 - a2_Min - mu2)/sigma2)/sqrt(2)) - (erf((b2 - a2_Min - mu2)/sigma2)/sqrt(2)) ) / ( Z2 ))
            
            
                    if L < 0: 
                        L = 0.0
    
                if L == 0.0: #This statement shouldn't be used either if everything was done correctly previously
                    Is_Bridge_State[z][j] = 1
                    Bridge_Transitions[z][j].append(h)
                
                
                Lower[z][j][h] = L
                Upper[z][j][h] = H
          
    Is_Bridge_State[z][:] = Is_Bridge_State[z][:].astype(int)        
            
    return (Lower,Upper, Reachable_States, Is_Bridge_State, Bridge_Transitions, Pre_States)









def BMDP_Computation_Continuous(R_Set, Target_Set, Max_Num, List_States_Per_Action):
    
    #Computes the lower and upper bound probabilities of transition from state
    #to state using the reachable sets in R_set and the target sets in Target_Set
       
    Lower = np.array(np.zeros((Max_Num, Target_Set.shape[0],Target_Set.shape[0]))) # i.e. 3D
    Upper = np.array(np.zeros((Max_Num, Target_Set.shape[0],Target_Set.shape[0]))) # Declare matrices for upper and lower boundaries
    Reachable_States = [[[] for x in range(Target_Set.shape[0])] for y in range(Max_Num)] # 
    Pre_States = [[[] for x in range(Target_Set.shape[0])] for y in range(Max_Num)]
    Is_Bridge_State = np.zeros((Max_Num, Target_Set.shape[0]))
    Bridge_Transitions = [[[] for x in range(Target_Set.shape[0])] for y in range(Max_Num)] # list is more dynamic.
    # We use this because we don't know how many bridge transitions are there)
    

    Z1 = (erf(Semi_Width_1/sigma1)/sqrt(2)) - (erf(-Semi_Width_1/sigma1)/sqrt(2))
    Z2 = (erf(Semi_Width_2/sigma2)/sqrt(2)) - (erf(-Semi_Width_2/sigma2)/sqrt(2))
    
    
           
    for z in range(len(R_Set)): # For number of modes
        
        for x in range(len(List_States_Per_Action[z])): # For number of reachable sets
            
            j = List_States_Per_Action[z][x]
            
            
            r0 = R_Set[z][j][0][0][0]
            r1 = R_Set[z][j][0][1][0]
            r2 = R_Set[z][j][0][0][1]
            r3 = R_Set[z][j][0][1][1]

            
            for h in range(len(Target_Set)):
    
                         
                q0 = Target_Set[h][0][0]
                q1 = Target_Set[h][1][0]
                q2 = Target_Set[h][0][1]
                q3 = Target_Set[h][1][1]               
    
                if q0 == LOW_1 and r0 + mu1 - Semi_Width_1 < LOW_1:
                    q0 = r0 + mu1 - Semi_Width_1
                        
                if q1 == UP_1 and r1 + mu1 + Semi_Width_1 > UP_1:
                    q1 = r1 + mu1 + Semi_Width_1
                        
                if q2 == LOW_2 and r2 + mu2 - Semi_Width_2 < LOW_2:
                    q2 = r2 + mu2 - Semi_Width_2
                        
                if q3 == UP_2 and r3 + mu2 + Semi_Width_2 > UP_2:
                    q3 = r3 + mu2 + Semi_Width_2                          
                
# Virtually extend the state to take into account that goes outside of the domain. Attach it back. Part of our dynamics.                
                if (r0 >= q1 + Semi_Width_1 - mu1) or (r1 <= q0 - Semi_Width_1 - mu1) or (r2 >= q3 + Semi_Width_2 - mu2) or (r3 <= q2 - Semi_Width_2 - mu2):
                    Lower[z][j][h] = 0.0
                    Upper[z][j][h] = 0.0
                    continue
# If the Target state is too far from a given reachable set                    
                
                Reachable_States[z][j].append(h)
                Pre_States[z][h].append(j)
                
                a1_Opt = ((q0 + q1)/2.0) - mu1
                a2_Opt = ((q2 + q3)/2.0) - mu2
                # To maximize Gaussian integration, place at center of Target state. Because Gaussian is symmetric.
                # Therefore, shift Gaussian as close as possible to optimizer point, for upper bound.
                # Therefore, shift Gaussian as far as possible to optimizer point, for lower bound.
                
                
                if (r1 < a1_Opt): 
                    a1_Max = r1
                    a1_Min = r0
                elif(r0 > a1_Opt): 
                    a1_Max = r0
                    a1_Min = r1
                else: 
                    a1_Max = a1_Opt       
                    if (a1_Opt <= (r1+r0)/2.0):
                        a1_Min = r1
                    else:
                        a1_Min = r0
                    
                    
                
                if (r2 > a2_Opt): 
                    a2_Max = r2
                    a2_Min = r3
                elif(r3 < a2_Opt): 
                    a2_Max = r3
                    a2_Min = r2
                else: 
                    a2_Max = a2_Opt
                    if (a2_Opt <= (r2+r3)/2.0):
                        a2_Min = r3
                    else:
                        a2_Min = r2
                                
                 
                    
  # If Gaussian is fully contained inside box, then upper bound is 1.          
                if a1_Max + mu1 - Semi_Width_1  > q0 and a1_Max + mu1 + Semi_Width_1 < q1 and a2_Max + mu2 - Semi_Width_2 > q2 and a2_Max + mu2 + Semi_Width_2 < q3:
                    H = 1.0
                else:
# Below, determining upper and lower bounds of INTEGRATION of the GAUSSIAN                    
                    if q0 < a1_Max + mu1 - Semi_Width_1:
                        b0 = a1_Max + mu1 - Semi_Width_1
                    else:
                        b0 = q0
                        
                    if q1 > a1_Max + mu1 + Semi_Width_1:
                        b1 = a1_Max + mu1 + Semi_Width_1
                    else:
                        b1 = q1
                        
                    if q2 < a2_Max + mu2 - Semi_Width_2:
                        b2 = a2_Max + mu2 - Semi_Width_2
                    else:
                        b2 = q2
                        
                    if q3 > a2_Max + mu2 + Semi_Width_2:
                        b3 = a2_Max + mu2 + Semi_Width_2
                    else:
                        b3 = q3    
                          
                        
                    # Perform integration
                    H = ( ( (erf((b1 - a1_Max - mu1)/sigma1)/sqrt(2)) - (erf((b0 - a1_Max - mu1)/sigma1)/sqrt(2)) ) / (Z1)) * (( (erf((b3 - a2_Max - mu2)/sigma2)/sqrt(2)) - (erf((b2 - a2_Max - mu2)/sigma2)/sqrt(2)) ) / ( Z2 ))
            # For numerical errors, if probability is greater than 1. For upper bound.
                    if H > 1:
                        H = 1.0
                        
                    
# Now we want to compute lower bound. If minimizer point is too far from the target state, based on if the edge of state is beyond than the edge of the Gaussian (mean + minimizer + Gaussian "half" width)                 
                if (a1_Min + mu1 + Semi_Width_1 <= q0) or (a1_Min + mu1 - Semi_Width_1 >= q1) or (a2_Min + mu2 + Semi_Width_2 <= q2) or (a2_Min + mu2 - Semi_Width_2 >= q3):
                    if H < 1e-14: #To account for numerical errors in the computation of overlaps which may create transitions that do not actually exist// Could be done better by saving which states are reachable per action when searching for the overlaps and only computing the transitions for those states 
                        H = 0.0
                        Reachable_States[z][j].remove(h)
                        Pre_States[z][h].remove(j)
                    else:   
                        Is_Bridge_State[z][j] = 1 # Means TRUE
                        Bridge_Transitions[z][j].append(h)
                    Lower[z][j][h] = 0.0
                    Upper[z][j][h] = H
                    continue            # to next state    
                
                
                else:
                    
                    if q0 < a1_Min + mu1 - Semi_Width_1:
                        b0 = a1_Min + mu1 - Semi_Width_1
                    else:
                        b0 = q0
                        
                    if q1 > a1_Min + mu1 + Semi_Width_1:
                        b1 = a1_Min + mu1 + Semi_Width_1
                    else:
                        b1 = q1
                        
                    if q2 < a2_Min + mu2 - Semi_Width_2:
                        b2 = a2_Min + mu2 - Semi_Width_2
                    else:
                        b2 = q2
                        
                    if q3 > a2_Min + mu2 + Semi_Width_2:
                        b3 = a2_Min + mu2 + Semi_Width_2
                    else:
                        b3 = q3   
                                        
              # Perform integration for lower bound probability of transition, based on Gaussian overlap      
                    L = ( ( (erf((b1 - a1_Min - mu1)/sigma1)/sqrt(2)) - (erf((b0 - a1_Min - mu1)/sigma1)/sqrt(2)) ) / (Z1)) * (( (erf((b3 - a2_Min - mu2)/sigma2)/sqrt(2)) - (erf((b2 - a2_Min - mu2)/sigma2)/sqrt(2)) ) / ( Z2 ))
            
            
                    if L < 0: #This if statement shouldn't be used
                        L = 0.0
    
                if L == 0.0: 
                    Is_Bridge_State[z][j] = 1
                    Bridge_Transitions[z][j].append(h)
                
                if H < 1e-13: #To account for numerical errors in the computation of overlaps which may create transitions that do not actually exist
                    H = 0.0
                    Reachable_States[z][j].remove(h)
                    Pre_States[z][h].remove(j)
                
                Lower[z][j][h] = L
                Upper[z][j][h] = H
    
          
    Is_Bridge_State[z][:] = Is_Bridge_State[z][:].astype(int)        
            
    return (Lower,Upper, Reachable_States, Is_Bridge_State, Bridge_Transitions, Pre_States)







def Build_Product_BMDP(T_l, T_u, A, L, Acc, Reachable_States, Is_Bridge_State, Bridge_Transitions):
    
    # Constructs the product between an IMC (defined by lower transition matrices
    # T_l and T_u and an Automata A according to Labeling function L
    # For simplicity, each state has the same number of actions
    
    #For init, we assume initial state is 0
    Num_States = len(Is_Bridge_State[0])
    Num_Act = len(Is_Bridge_State)
    Init = np.zeros((Num_States)) # Corresponds to number of system states
    Init = Init.astype(int)    
    
    Is_A = np.zeros(Num_States*len(A))
    Is_N_A = np.zeros(Num_States*len(A))
    Which_A = [[] for x in range(Num_States*len(A))]
    Which_N_A = [[] for x in range(Num_States*len(A))]
    
    New_Reachable_States = [[[] for x in range(Num_States*len(A))] for y in range(Num_Act)]
    Product_Pre_States = [[[] for x in range(Num_States*len(A))] for y in range(Num_Act)]
    New_Is_Bridge_State = np.zeros((Num_Act, Num_States*len(A)))
    New_Bridge_Transitions = [[[] for x in range(Num_States*len(A))] for y in range(Num_Act)]
    
    IA_l = {'Num_Act': Num_Act, 'Num_S': Num_States*len(A)}
    IA_u = {'Num_Act': Num_Act, 'Num_S': Num_States*len(A)}
    
    for x in range(len(Acc)): # Which states in the product are accepting or not.
        for i in range(len(Acc[x][0])):
            for j in range(Num_States):
                Is_N_A[len(A)*j + Acc[x][0][i]] = 1
                Which_N_A[len(A)*j + Acc[x][0][i]].append(x)
        
        for i in range(len(Acc[x][1])):
            for j in range(Num_States):
                Is_A[len(A)*j + Acc[x][1][i]] = 1
                Which_A[len(A)*j + Acc[x][1][i]].append(x)            
                
    
    for y in range(Num_Act): # For every action (mode)
        for i in range(Num_States):
            for j in range(len(A)):
                for k in range(Num_States):
                    for l in range(len(A)):
                        
                        if L[k] in A[j][l]:
                            
                            if y == 0 and j == 0: # To account for true first state. y == 0 because it goes through it once and for all for the first action (mode)
                                Init[k] = l
    
                            IA_l[y, len(A)*i+j, len(A)*k+l] = T_l[y][i][k] # then, if the transition exists, then 
                            IA_u[y, len(A)*i+j, len(A)*k+l] = T_u[y][i][k] # probability T_up and T_down are the same from the IMC abstraction probability interval
                            
                            # If there is more than 1 mode, then take into account of the action from the policy
                            
                            if T_u[y][i][k] > 0:
                                New_Reachable_States[y][len(A)*i+j].append(len(A)*k+l)
                                Product_Pre_States[y][len(A)*k+l].append(len(A)*i+j)
                                if T_l[y][i][k] == 0:
                                    New_Is_Bridge_State[y, len(A)*i+j] = 1 # Keep track of new bridge states within the IMC states
                                    New_Bridge_Transitions[y][len(A)*i+j].append(len(A)*k+l)
                            

                            

    
                 

    Is_A = Is_A.astype(int)
    Is_N_A = Is_N_A.astype(int)
    New_Is_Bridge_State = New_Is_Bridge_State.astype(int)                         

    return (IA_l, IA_u, Is_A, Is_N_A, Which_A, Which_N_A, New_Reachable_States, New_Is_Bridge_State, New_Bridge_Transitions, Init, Product_Pre_States) 
# Finding transition from each hybrid state to hybrid state

def Build_Product_BMDP_Refinement(T_l, T_u, A, L, Acc, Reachable_States, Is_Bridge_State, Bridge_Transitions , Allowable_Actions, Is_In_Permanent_Comp):
    
    # Constructs the product between an IMC (defined by lower transition matrices
    # T_l and T_u and an Automata A according to Labeling function L
    # For simplicity, each state has the same number of actions
    
    #For init, we assume initial state is 0

    Init = np.zeros((len(T_l[0]))) # Corresponds to number of system states
    Init = Init.astype(int)    
    
    Is_A = np.zeros(len(T_l[0])*len(A))
    Is_N_A = np.zeros(len(T_l[0])*len(A))
    Which_A = [[] for x in range(len(T_l[0])*len(A))]
    Which_N_A = [[] for x in range(len(T_l[0])*len(A))]
    
    New_Reachable_States = [[[] for x in range(len(T_l[0])*len(A))] for y in range(len(T_l))]
    Product_Pre_States = [[set([]) for x in range(len(T_l[0])*len(A))] for y in range(len(T_l))]
    New_Is_Bridge_State = np.zeros((len(T_l), len(T_l[0])*len(A)))
    New_Bridge_Transitions = [[[] for x in range(len(T_l[0])*len(A))] for y in range(len(T_l))]
    
    IA_l = {'Num_Act': len(T_l), 'Num_S': len(T_l[0])*len(A)}
    IA_u = {'Num_Act': len(T_l), 'Num_S': len(T_l[0])*len(A)}

    for x, Accx in enumerate(Acc): # Which states in the product are accepting or not.
        for i in Accx[0]:
            for j in range(len(T_l[0])):
                Is_N_A[len(A)*j + i] = 1
                Which_N_A[len(A)*j + i].append(x)
        
        for i in Accx[1]:
            for j in range(len(T_l[0])):
                Is_A[len(A)*j + i] = 1
                Which_A[len(A)*j + i].append(x)            
                
    for i in range(len(T_l[0])):
        Found_Init = 0
        for j in range(len(A)):
            if Found_Init == 0:
                if L[i] in A[0][j]:
                    Init[i] = j
                    Found_Init = 1
            if Is_In_Permanent_Comp[len(A)*i+j] == 1:
                continue
            Indi = len(A)*i+j
            for y in Allowable_Actions[Indi]: # For every action (mode)
                for k in Reachable_States[y][i]:
                    for l in range(len(A)):
                        
                        if L[k] in A[j][l]:
                            Indk = len(A)*k+l
                            
    
                            IA_l[y, Indi, Indk] = T_l[y][i][k] 
                            IA_u[y, Indi, Indk] = T_u[y][i][k] 
                            
                            
                            if T_u[y][i][k] > 0:
                                New_Reachable_States[y][Indi].append(Indk)
                                Product_Pre_States[y][Indk].add(Indi)
                                if T_l[y][i][k] == 0:
                                    New_Is_Bridge_State[y, Indi] = 1 
                                    New_Bridge_Transitions[y][Indi].append(Indk)
                            
    
                 

    Is_A = Is_A.astype(int)
    Is_N_A = Is_N_A.astype(int)
    New_Is_Bridge_State = New_Is_Bridge_State.astype(int)                         

    return (IA_l, IA_u, Is_A, Is_N_A, Which_A, Which_N_A, New_Reachable_States, New_Is_Bridge_State, New_Bridge_Transitions, Init, Product_Pre_States) 


def Build_Product_BMDP_Continuous(T_l, T_u, A, L, Acc, Reachable_States, Is_Bridge_State, Bridge_Transitions, Discrete_Actions):
    
    # Constructs the product between an IMC (defined by lower transition matrices
    # T_l and T_u and an Automata A according to Labeling function L
    # For simplicity, each state has the same number of actions
    
    #For init, we assume initial state is 0
    Init = np.zeros((T_l.shape[1])) # Corresponds to number of system states
    Init = Init.astype(int)    
    
    Is_A = np.zeros(T_l.shape[1]*len(A))
    Is_N_A = np.zeros(T_l.shape[1]*len(A))
    Which_A = [[] for x in range(T_l.shape[1]*len(A))]
    Which_N_A = [[] for x in range(T_l.shape[1]*len(A))]
    
    New_Reachable_States = [[[] for x in range(T_l.shape[1]*len(A))] for y in range(T_l.shape[0])]
    New_Is_Bridge_State = np.zeros((T_l.shape[0], T_l.shape[1]*len(A)))
    New_Bridge_Transitions = [[[] for x in range(T_l.shape[1]*len(A))] for y in range(T_l.shape[0])]
    
    
    IA_l = {'Num_Act': len(T_l), 'Num_S': len(T_l[0])*len(A)}
    IA_u = {'Num_Act': len(T_l), 'Num_S': len(T_l[0])*len(A)}
    
    for x in range(len(Acc)): # Which states in the product are accepting or not.
        for i in range(len(Acc[x][0])):
            for j in range(T_l.shape[1]):
                Is_N_A[len(A)*j + Acc[x][0][i]] = 1
                Which_N_A[len(A)*j + Acc[x][0][i]].append(x)
        
        for i in range(len(Acc[x][1])):
            for j in range(T_l.shape[1]):
                Is_A[len(A)*j + Acc[x][1][i]] = 1
                Which_A[len(A)*j + Acc[x][1][i]].append(x)            
                
    
    
    for i in range(T_l.shape[1]): 
        for y in range(len(Discrete_Actions[i])):
            for j in range(len(A)):
                for k in Reachable_States[y][i]:    
                    for l in range(len(A)):
                        
                        if L[k] in A[j][l]:
                            
                            if y == 0 and j == 0: 
                                Init[k] = l
    
                            IA_l[y, len(A)*i+j, len(A)*k+l] = T_l[y,i,k] 
                            IA_u[y, len(A)*i+j, len(A)*k+l] = T_u[y,i,k] 
                            
                            
                            if T_u[y,i,k] > 0:
                                New_Reachable_States[y][len(A)*i+j].append(len(A)*k+l)
                                if T_l[y,i,k] == 0:
                                    New_Is_Bridge_State[y, len(A)*i+j] = 1 
                                    New_Bridge_Transitions[y][len(A)*i+j].append(len(A)*k+l)
    
                 

    Is_A = Is_A.astype(int)
    Is_N_A = Is_N_A.astype(int)
    New_Is_Bridge_State = New_Is_Bridge_State.astype(int)                         

    return (IA_l, IA_u, Is_A, Is_N_A, Which_A, Which_N_A, New_Reachable_States, New_Is_Bridge_State, New_Bridge_Transitions, Init) 





def Find_Greatest_Accepting_BSCCs(IA1_l, IA1_u, Is_Acc, Is_NAcc, Wh_Acc_Pair, Wh_NAcc_Pair, Al_Act_Pot, Al_Act_Perm, first, Reachable_States, Bridge_Transition, Is_Bridge_State, Acc, Potential_Policy, Permanent_Policy, Is_In_Permanent_Comp, List_Permanent_Acc_BSCC, Previous_A_BSCC):

    
    G = np.zeros((IA1_l.shape[1],IA1_l.shape[2]))

    
    if first == 1:
        Al_Act_Pot = list([])
        Al_Act_Perm = list([])
        for y in range(IA1_l.shape[1]): # For all the system states
            Al_Act_Pot.append(range(IA1_l.shape[0])) # Appending all allowable actions in the beginning
            Al_Act_Perm.append(range(IA1_l.shape[0]))
    
    
    for k in range(IA1_u.shape[0]): # Out of all the upper bounds of every mode, if at least one is greater than 0, then, G is 1 for that transition between the two states
        for i in range(IA1_u.shape[1]): # Perhaps, might need to specify which actions allow us to make G equal to one.
            for j in range(IA1_u.shape[2]):
                if IA1_u[k,i,j] > 0: # It's an array, not a list. 
                    G[i,j] = 1

    Counter_Status2 = 0 #Indicates which Status2 BSCC we are currently inspecting
    Counter_Status3 = 0 #Indicates which Status2 BSCC we are currently inspecting
    Which_Status2_BSCC = [] #Tells you with respect to which BSCC are the states duplicated
    Has_Found_BSCC_Status_2 = list([]) #0 if you found a BSCC in the duplicate, 1 otherwise
    List_Found_BSCC_Status_2 = list([]) #Will contain the set of states for which an accepting BSCC has been found in duplicates
    Original_SCC_Status_2 = list([]) #Keeps track of the original SCC before duplication
    Which_Status3_BSCC = list([])
    Number_Duplicates2 = 0 #Tells you how many BSCCs have been duplicated so far for status 2
    Number_Duplicates3 = 0 #Tells you how many BSCCs have been duplicated so far for status 3
    Status2_Act = list([]) #List which keeps track of the allowed actions for duplicate states
    Status3_Act = list([])
    List_Status3_Found = list([])

    if first == 0:
        Deleted_States = []
        Prev_A = set().union(*Previous_A_BSCC)
        Deleted_States.extend(list(set(range(G.shape[0])) - set(Prev_A)))
        
        Ind = list(set(Prev_A))
        Ind.sort()
        
        G = np.delete(np.array(G),Deleted_States,axis=0)
        G = np.delete(np.array(G),Deleted_States,axis=1)
        

    else:
        Ind = range(G.shape[0])
             
    first = 0 


   
    C,n = SSCC(G) # Search SCC using G. n = number of SCCs. C contains all the SCCs    
    tag = 0; # Trackers for indices
    m = 0 ;


    
    SCC_Status = [0]*n ###Each SCC has 'status': 0: looking for potential BSCCs, 1: looking for permanent BSCCs, 2: duplicate potential BSCCs , 3: duplicate permanent BSCCs
   
    G_Pot_Acc_BSCCs = list([]) #Contains the set of greatest potential accepting BSCCs
    G_Per_Acc_BSCCs = list([]) #Contains the set of greatest permanent accepting BSCC
    

    
    for i in range(len(List_Permanent_Acc_BSCC)): #Have to add them now since they are deleted from the graph upon searching for the BSCC
        for j in range(len(List_Permanent_Acc_BSCC[i])):
            G_Pot_Acc_BSCCs.append(List_Permanent_Acc_BSCC[i][j])
            G_Per_Acc_BSCCs.append(List_Permanent_Acc_BSCC[i][j])
       
    
    List_G_Pot = [] #Lists the greatest potential BSCCs (which are not cross listed with permanent BSCCs)
    Is_In_Potential_Acc_BSCC = np.zeros(IA1_l.shape[1]) #Is the state in the largest potential accepting BSCC?
    Which_Potential_Acc_BSCC = np.zeros(IA1_l.shape[1]) #Keeps track of which accepting BSCC does each state belong to (if applicable)
    Which_Potential_Acc_BSCC.astype(int)
    Is_In_Potential_Acc_BSCC.astype(int)
    Bridge_Potential_Accepting = [] #List which contains the bridge states for each potential accepting BSCC
    Maybe_Permanent = [] #List which contains potential components before checking whether these components are permanent or not
    
    
    
    while tag == 0:
        
        if len(C) == 0:
            break
        
        
        skip = 1 # To reset skip tag. Assume that I skip
        SCC = C[m];
        

        

        Orig_SCC = []
        for k in range(len(SCC)):
            Orig_SCC.append(Ind[SCC[k]]) # absolute index/label of states is added to the list of states in the Orig_SCC
        BSCC = 1
        # if there are no accepting states in the given SCC, then continue, because it cannot be a winning component.
    
        # Search through Orig_SCC, in which you would find the absolute index/label of states, which I can use to find the corresponding Is_Accepting TRUTH/FALSE value.
        
        if len(Has_Found_BSCC_Status_2) != 0:
            if SCC_Status[m] == 2 and Has_Found_BSCC_Status_2[Which_Status2_BSCC[Counter_Status2]] == 1:
                Counter_Status2 += 1
                if m < (len(C)-1): # To avoid searching beyond the range of the C, which is the list of SCCs. Related to C[m]
                    m += 1 # Tag to move on to the next SCC in the list of SCCs in C.                   
                    continue # then while through the next SCC. Related SCC = C[m]
                else:                   
                    break 



        for l in range(len(Orig_SCC)):
            if Is_Acc[Orig_SCC[l]] == 1: # Keep searching until the given state in the Orig_SCC is accepting.
                skip = 0
                break

        if skip == 1: # If skip tag was activated, because no accepting state was found in the Orig_SCC,
            if SCC_Status[m] == 0: #These states will never be a potential or permanent accepting BSCCs since they do have an accepting state (either from the beginning or accepting states were leaky and removed), removed all allowed actions
                for i in range(len(SCC)):
                    Al_Act_Pot[Ind[SCC[i]]] = list([])
                    Al_Act_Perm[Ind[SCC[i]]] = list([])            
            if m < (len(C)-1): # To avoid searching beyond the range of the C, which is the list of SCCs. Related to C[m]
                m += 1 # Tag to move on to the next SCC in the list of SCCs in C.
                continue # then while through the next SCC. Related SCC = C[m]
            else: 
                break     # if at end of the list. want to break.

            
        
        
        Leak = list([])
        Check_Tag = 1
        Reach_in_R = [[[] for y in range(IA1_u.shape[0])] for x in range(len(Orig_SCC))] # Reach_in_R contains all the reachable non-leaky states inside the SCC, with respect to state i.
        Pre = [[[] for y in range(IA1_u.shape[0])] for x in range(len(Orig_SCC))] # Creating list of list of lists, to account for mode, state, transitions. Modes are nested inside state.
        All_Leaks = list([])
        Check_Orig_SCC = np.zeros(len(Orig_SCC), dtype=int)
        

        while (len(Leak) != 0 or Check_Tag == 1):
                       
          
            
            if SCC_Status[m] == 0:
                                                              
                ind_leak = []
                Leak = []
                      
                for i in range(len(Orig_SCC)): # Original SCC that contains all SCCs
                    if Check_Orig_SCC[i] == -1 :
                        continue # -1 is a tag for leaky state which should be skipped over.
                    tag_m = 0# tag_mode
                    
                    for k in range(len(Al_Act_Pot[Orig_SCC[i]])): # Loop through all the allowable actions from the current state
                            # The state number if the index for the allowable action array
        
                        Set_All_Leaks = set(Orig_SCC) - set(All_Leaks) # Removes all leaky states from Orig_SCC
                        Diff_List1 = list(set(Reachable_States[Al_Act_Pot[Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - Set_All_Leaks) # All Reachable states (that is outside the Set_All_Leaks) of the current state. Al_Act[i][k] means action from the set of allowable actions with respect to current state i
                        Diff_List2 = list(set(Diff_List1) - set(Bridge_Transition[Al_Act_Pot[Orig_SCC[i]][k-tag_m]][Orig_SCC[i]])) # After taking out bridge transitions, then if it's not 0, then it's a leaky state 
                        if Check_Tag == 1: # tag to create "Pre" and Reach_in_R.

                            Reach_in_R[i][Al_Act_Pot[Orig_SCC[i]][k-tag_m]].extend(list(set(Reachable_States[Al_Act_Pot[Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - set(Diff_List1)))
                            for j in range(len(Reach_in_R[i][Al_Act_Pot[Orig_SCC[i]][k-tag_m]])):
                                Pre[Orig_SCC.index(Reach_in_R[i][Al_Act_Pot[Orig_SCC[i]][k-tag_m]][j])][Al_Act_Pot[Orig_SCC[i]][k-tag_m]].append(Orig_SCC[i]) # With respect to state i, "pre" contains all the states inside the SCC that can reach State i. The reason for having "Pre" is to know which states inside the SCC lead to the state i, (which is useful if state i is a leaky state)
                       
    
                        if (len(Diff_List2) != 0) or (sum(IA1_u[Al_Act_Pot[Orig_SCC[i]][k-tag_m], Orig_SCC[i], Reach_in_R[i][Al_Act_Pot[Orig_SCC[i]][k-tag_m]]])<1) : # Reach_in_R means reachable states inside scc
                            # the sum statement means that if all the upper bounds of the transitions within the SCC don't add up to 1, then there is always the possibility of a leakiness.
                            
                            Al_Act_Perm[Orig_SCC[i]].remove(Al_Act_Pot[Orig_SCC[i]][k-tag_m]) #If the action cannot make a potential BSCC, then it cannot make a permanent BSCC either
                            Al_Act_Pot[Orig_SCC[i]].remove(Al_Act_Pot[Orig_SCC[i]][k-tag_m]) # Remove the leaky mode from that list of allowable action for the current state i.
                            tag_m += 1 # To account for the missing index in the "for k" loop.
                            BSCC = 0 # Because once action is removed, might have changed the whole SCC, so might as well just keep BSCC to be 0.
                            
                    if len(Al_Act_Pot[Orig_SCC[i]]) == 0: # If there are no more available actions for the current state i.                       
                        Leak.append(Orig_SCC[i])
                        ind_leak.append(i)
                        
                if len(Leak) != 0: # It means that a new leaky state is found, because Leak=[] every loop of the while
                    All_Leaks.extend(Leak) # Then add to All_leaks. extend means "adding" a list without brackets
                    BSCC = 0 # To confirm that previous SCC is surely not a BSCC, until further verifications to see if SCC is a BSCC
                    for i in range(len(Leak)): # for all the newly found leaky states
                        Check_Orig_SCC[ind_leak[i]] = -1 # The state in the SCC is tagged "leaky"
                        for j in range(len(Pre[ind_leak[i]])): # Loop through all the actions of the pre-states, with respect to the leaky current state i.
                            for k in range(len(Pre[ind_leak[i]][j])): # Loop through all the states in each respective action of the pre-states of the leaky current state i.
                                Reach_in_R[Orig_SCC.index(Pre[ind_leak[i]][j][k])][j].remove(Leak[i]) # Removes leaky state from the reachable set of states of all OTHER states in the SCC, if the leaky state is reachable from those states
            # Have to loop through all the modes for which there are all SCCs.
                Check_Tag = 0  # Changes Check_Tag after having populated the "Pre" and Reach_in_R for all the states in set of SCCs. But now need to do it for all actions (modes).
 
            if SCC_Status[m] == 1:
                
                 
                ind_leak = []
                Leak = []                     
                for i in range(len(Orig_SCC)): # Original SCC that contains all SCCs
                    if Check_Orig_SCC[i] == -1 :
                        continue # -1 is a tag for leaky state which should be skipped over.
                    tag_m = 0# tag_mode
                     


                    for k in range(len(Al_Act_Perm[Orig_SCC[i]])): # Loop through all the allowable actions from the current state
                            # The state number if the index for the allowable action array
              
                
                        Set_All_Leaks = set(Orig_SCC) - set(All_Leaks) # Removes all leaky states from Orig_SCC

                        Diff_List1 = list(set(Reachable_States[Al_Act_Perm[Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - Set_All_Leaks) # All Reachable states (that is outside the Set_All_Leaks) of the current state. Al_Act[i][k] means action from the set of allowable actions with respect to current state i

                        if (len(Diff_List1) != 0): # If some state is reachable outside of the SCC, then the SCC is not permanent for wrt that action                           
                            Al_Act_Perm[Orig_SCC[i]].remove(Al_Act_Perm[Orig_SCC[i]][k-tag_m]) #If the action cannot make a potential BSCC, then it cannot make a permanent BSCC either
                            tag_m += 1 # To account for the missing index in the "for k" loop.
                            BSCC = 0 # Because once action is removed, might have changed the whole SCC, so might as well just keep BSCC to be 0.
                                                    
                    if len(Al_Act_Perm[Orig_SCC[i]]) == 0: # If there are no more available actions for the current state i.                        
                        Leak.append(Orig_SCC[i])
                        ind_leak.append(i)
                if len(Leak) != 0: # It means that a new leaky state is found, because Leak=[] every loop of the while
                    All_Leaks.extend(Leak) # Then add to All_leaks. extend means "adding" a list without brackets
                    BSCC = 0 # To confirm that previous SCC is surely not a BSCC, until further verifications to see if SCC is a BSCC
                    for i in range(len(Leak)): # for all the newly found leaky states
                        Check_Orig_SCC[ind_leak[i]] = -1 # The state in the SCC is tagged "leaky"
                Check_Tag = 0  # Changes Check_Tag after having populated the "Pre" and Reach_in_R for all the states in set of SCCs. But now need to do it for all actions (modes).


            if SCC_Status[m] == 2:
                
                                                             
                ind_leak = []
                Leak = []                      
                for i in range(len(Orig_SCC)): # Original SCC that contains all SCCs
                    if Check_Orig_SCC[i] == -1 :
                        continue # -1 is a tag for leaky state which should be skipped over.
                    tag_m = 0# tag_mode
                    
                    for k in range(len(Status2_Act[Counter_Status2][Orig_SCC[i]])): # Loop through all the allowable actions from the current state
                            # The state number if the index for the allowable action array      
                        Set_All_Leaks = set(Orig_SCC) - set(All_Leaks) # Removes all leaky states from Orig_SCC
                        Diff_List1 = list(set(Reachable_States[Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - Set_All_Leaks) # All Reachable states (that is outside the Set_All_Leaks) of the current state. Al_Act[i][k] means action from the set of allowable actions with respect to current state i
                        Diff_List2 = list(set(Diff_List1) - set(Bridge_Transition[Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]][Orig_SCC[i]])) # After taking out bridge transitions, then if it's not 0, then it's a leaky state 
                        if Check_Tag == 1: # tag to create "Pre" and Reach_in_R.
                            # "Orig_SCC[i]"th state inside the Al_Act, and the kth action of that list inside Al_Act

                            Reach_in_R[i][Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]].extend(list(set(Reachable_States[Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - set(Diff_List1)))
                            for j in range(len(Reach_in_R[i][Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]])):
                                Pre[Orig_SCC.index(Reach_in_R[i][Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]][j])][Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]].append(Orig_SCC[i]) # With respect to state i, "pre" contains all the states inside the SCC that can reach State i. The reason for having "Pre" is to know which states inside the SCC lead to the state i, (which is useful if state i is a leaky state)
                       
    
                        if (len(Diff_List2) != 0) or (sum(IA1_u[Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m], Orig_SCC[i], Reach_in_R[i][Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]]])<1) : # Reach_in_R means reachable states inside scc
                            # the sum statement means that if all the upper bounds of the transitions within the SCC don't add up to 1, then there is always the possibility of a leakiness.
                            
                            Status2_Act[Counter_Status2][Orig_SCC[i]].remove(Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]) # Remove the leaky mode from that list of allowable action for the current state i.
                            tag_m += 1 # To account for the missing index in the "for k" loop.
                            BSCC = 0 # Because once action is removed, might have changed the whole SCC, so might as well just keep BSCC to be 0.
                            
                    if len(Status2_Act[Counter_Status2][Orig_SCC[i]]) == 0: # If there are no more available actions for the current state i.                        
                        Leak.append(Orig_SCC[i])
                        ind_leak.append(i)
                        
                if len(Leak) != 0: # It means that a new leaky state is found, because Leak=[] every loop of the while
                    All_Leaks.extend(Leak) # Then add to All_leaks. extend means "adding" a list without brackets
                    BSCC = 0 # To confirm that previous SCC is surely not a BSCC, until further verifications to see if SCC is a BSCC
                    for i in range(len(Leak)): # for all the newly found leaky states
                        Check_Orig_SCC[ind_leak[i]] = -1 # The state in the SCC is tagged "leaky"
                        for j in range(len(Pre[ind_leak[i]])): # Loop through all the actions of the pre-states, with respect to the leaky current state i.
                            for k in range(len(Pre[ind_leak[i]][j])): # Loop through all the states in each respective action of the pre-states of the leaky current state i.
                                Reach_in_R[Orig_SCC.index(Pre[ind_leak[i]][j][k])][j].remove(Leak[i]) # Removes leaky state from the reachable set of states of all OTHER states in the SCC, if the leaky state is reachable from those states
            # Have to loop through all the modes for which there are all SCCs.
                Check_Tag = 0  # Changes Check_Tag after having populated the "Pre" and Reach_in_R for all the states in set of SCCs. But now need to do it for all actions (modes).
 

            if SCC_Status[m] == 3:
                
                                
                ind_leak = []
                Leak = []                     
                for i in range(len(Orig_SCC)): # Original SCC that contains all SCCs
                    if Check_Orig_SCC[i] == -1 :
                        continue # -1 is a tag for leaky state which should be skipped over.
                    tag_m = 0# tag_mode
            
                    for k in range(len(Status3_Act[Counter_Status3][Orig_SCC[i]])): # Loop through all the allowable actions from the current state
                          
                        # The state number if the index for the allowable action array
                                                   
                                        
                        Set_All_Leaks = set(Orig_SCC) - set(All_Leaks) # Removes all leaky states from Orig_SCC           
                        
                        
                        Diff_List1 = list(set(Reachable_States[Status3_Act[Counter_Status3][Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - Set_All_Leaks) # All Reachable states (that is outside the Set_All_Leaks) of the current state. Al_Act[i][k] means action from the set of allowable actions with respect to current state i
    

    
                        if (len(Diff_List1) != 0): # If some state is reachable outside of the SCC, then the SCC is not permanent for wrt that action                           
                            Status3_Act[Counter_Status3][Orig_SCC[i]].remove(Status3_Act[Counter_Status3][Orig_SCC[i]][k-tag_m]) #If the action cannot make a potential BSCC, then it cannot make a permanent BSCC either
                            tag_m += 1 # To account for the missing index in the "for k" loop.
                            BSCC = 0 # Because once action is removed, might have changed the whole SCC, so might as well just keep BSCC to be 0.
                            
                    if len(Status3_Act[Counter_Status3][Orig_SCC[i]]) == 0: # If there are no more available actions for the current state i.                        
                        Leak.append(Orig_SCC[i])
                        ind_leak.append(i)
                if len(Leak) != 0: # It means that a new leaky state is found, because Leak=[] every loop of the while
                    All_Leaks.extend(Leak) # Then add to All_leaks. extend means "adding" a list without brackets
                    BSCC = 0 # To confirm that previous SCC is surely not a BSCC, until further verifications to see if SCC is a BSCC
                    for i in range(len(Leak)): # for all the newly found leaky states
                        Check_Orig_SCC[ind_leak[i]] = -1 # The state in the SCC is tagged "leaky"
                Check_Tag = 0  # Changes Check_Tag after having populated the "Pre" and Reach_in_R for all the states in set of SCCs. But now need to do it for all actions (modes).



 
               
        if BSCC == 0: # Means actions are modified/removed, and subsequently the state may be removed if all actions are removed. Need to check the connectivity of the remaining states of the remaining states in the SCC.
            

            
            SCC = list(set(Orig_SCC) - set(All_Leaks))
            for k in range(len(SCC)):
                SCC[k] = Ind.index(SCC[k])
                
            

            
            if SCC_Status[m] == 0: #Looking for greatest potential 
                
                
            #Could be optimized, convert back non-leaky states to indices of reduced graph
                                
                if len(SCC) != 0: # if some states are left in the SCC
                    SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                    New_G = np.zeros((len(SCC), len(SCC)))# Create new graph
                    for i in range(len(SCC)):
                        for k in range(len(Al_Act_Pot[Ind[SCC[i]]])):
                            for j in range(len(SCC)):
                                if IA1_u[Al_Act_Pot[Ind[SCC[i]]][k], Ind[SCC[i]], Ind[SCC[j]]] > 0:
                                    New_G[i,j] = 1
                        
                    C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label               
                    for j in range(len(C_new)):
                        for k in range(len(C_new[j])):
                            C_new[j][k] = SCC[C_new[j][k]] # Converting C_new to SCC label of the states
                        C.append(C_new[j]) # put them in the front.
                        SCC_Status.append(0)

            if SCC_Status[m] == 1: #Looking for permanent
                
                
                
            #Could be optimized, convert back non-leaky states to indices of reduced graph               
                if len(SCC) != 0: # if some states are left in the SCC
                    SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                    New_G = np.zeros((len(SCC), len(SCC)))# Create new graph
                    for i in range(len(SCC)):
                        for k in range(len(Al_Act_Perm[Ind[SCC[i]]])):
                            for j in range(len(SCC)):
                                if IA1_u[Al_Act_Perm[Ind[SCC[i]]][k], Ind[SCC[i]], Ind[SCC[j]]] > 0:                                        
                                    New_G[i,j] = 1
                    C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label               
                    

                    
                    for j in range(len(C_new)):
                        for k in range(len(C_new[j])):
                            C_new[j][k] = SCC[C_new[j][k]] # Converting C_new to SCC label of the states
                        C.append(C_new[j]) # put them in the front.
                        SCC_Status.append(1)

            if SCC_Status[m] == 2:
                                                
                            
                if len(SCC) != 0: # if some states are left in the SCC
                    Duplicate_Actions = copy.deepcopy(Status2_Act[Counter_Status2])
                    SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                    New_G = np.zeros((len(SCC), len(SCC)))# Create new graph
                    for i in range(len(SCC)):
                        for k in range(len(Al_Act_Pot[Ind[SCC[i]]])):
                            for j in range(len(SCC)):
                                if IA1_u[Al_Act_Pot[Ind[SCC[i]]][k], Ind[SCC[i]], Ind[SCC[j]]] > 0:
                                    New_G[i,j] = 1

                    C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label               
                    for j in range(len(C_new)):
                        Status2_Act.append(Duplicate_Actions)
                        Which_Status2_BSCC.append(Which_Status2_BSCC[Counter_Status2])
                        for k in range(len(C_new[j])):
                            C_new[j][k] = SCC[C_new[j][k]] # Converting C_new to SCC label of the states
                        C.append(C_new[j]) # put them in the front.
                        SCC_Status.append(2)                                        
                Counter_Status2 += 1

            if SCC_Status[m] == 3: 
                
                
                if len(SCC) != 0: #
                    Duplicate_Actions = copy.deepcopy(Status3_Act[Counter_Status3])
                    SCC = sorted(SCC, key=int)               
                    New_G = np.zeros((len(SCC), len(SCC)))
                    for i in range(len(SCC)):
                        for k in range(len(Al_Act_Pot[Ind[SCC[i]]])):
                            for j in range(len(SCC)):
                                if IA1_u[Al_Act_Pot[Ind[SCC[i]]][k], Ind[SCC[i]], Ind[SCC[j]]] > 0:
                                    New_G[i,j] = 1

                    C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label               
                    for j in range(len(C_new)):
                        Status3_Act.append(Duplicate_Actions)
                        Which_Status3_BSCC.append(Which_Status3_BSCC[Counter_Status3])
                        for k in range(len(C_new[j])):
                            C_new[j][k] = SCC[C_new[j][k]] # Converting C_new to SCC label of the states
                        C.append(C_new[j]) # put them in the front.
                        SCC_Status.append(3)                                        
                Counter_Status3 += 1                
            
        else: # it means the SCC is a BSCC
            

            

            Bridge_States = []           
            if SCC_Status[m] == 0:
                            

                ind_acc = [] #Contains the Automaton indices of accepting states in BSCC
                acc_states = [] #Contains the Accepting States in BSCC
                ind_non_acc = [] #Contains the Automaton indices of non-accepting states in BSCC
                non_acc_states = [] #Contans the non Accepting States in BSCC
                Inevitable = 1 #Tag to see if BSCC is an inevitable BSCC
                                
                for j in range(len(SCC)):
                                       
                    if Is_Acc[Ind[SCC[j]]] == 1: # accepting, then add it as accepting and vice versa
                        acc_states.append(SCC[j])
                        indices = [] # Establish index list
                        for n in range(len(Wh_Acc_Pair[Ind[SCC[j]]])): # loop through to find which accepting pair-conditions are sufficed by the given state
                            indices.append(Wh_Acc_Pair[Ind[SCC[j]]][n]) # Add the respective absolute index/label for the states
                        ind_acc.append(indices) # Add the accumulated list of indices with respect to the BSCC

                    if Is_NAcc[Ind[SCC[j]]] == 1: # do the same thing for non-accepting states in the BSCC
                        non_acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Wh_NAcc_Pair[Ind[SCC[j]]])):
                            indices.append(Wh_NAcc_Pair[Ind[SCC[j]]][n])
                        ind_non_acc.append(indices)                         
                          
                    for i in range(len(Al_Act_Pot[Ind[SCC[j]]])): # For a given list of allowable actions for a given state in the SCC
                        if Is_Bridge_State[Al_Act_Pot[Ind[SCC[j]]][i]][Ind[SCC[j]]] == 1: # Is the given state in the SCC a bridge state? If yes,                                                                  
                            Diff_List = np.setdiff1d(Reachable_States[Al_Act_Pot[Ind[SCC[j]]][i]][Ind[SCC[j]]], Orig_SCC) # Subtract all states within SCC from the list of reachable states from the current state. Then-
                            if len(Diff_List) != 0: # is there anything that remains?
                                Inevitable = 0  # If so, then inevitability/permanence = 0, which means that the current SCC is no bueno in the permanency test
                            Bridge_States.append(Ind[SCC[j]]) # Then bridge_states list is constructed for the given SCC.
                            #not using this Bridge_States variable at the moment

                Acc_Tag = 0
                Accept = [] #Contains unmatched accepting states
                                                   
                if len(non_acc_states) == 0: # If there are no non-accepting states,
                    Acc_Tag = 1 # then activate tag for accepting BSCC.
                    for j in range(len(acc_states)): # Subsequently, add to the list of accepting BSCC.
                        Accept.append(acc_states[j])
                
                else:
                    
                    Non_Accept_Remove = [[] for x in range(len(Acc))] # Contains all non-accepting states which prevent the bscc to be accepting for all pairs                        
                    for j in range(len(ind_acc)): # Recall that ind_acc contains the accumulated list of indices of the pairs with respect to the BSCC
                        for l in range(len(ind_acc[j])): # ind_acc[j] contains the indices for the relevant DRA pair, to which the state of the BSCC was complying for acceptance
                            Check_Tag = 0
                            Keep_Going = 0
                            for w in range(len(ind_non_acc)): # Same thing for non accepting
                                if ind_acc[j][l] in ind_non_acc[w]: # Checks if accepting index is in list of non_accepting indices ?? YES
                                    Check_Tag = 1 # Means that the current index in the accepting states doesn't make the BSCC accepting
                                    if len(Non_Accept_Remove[ind_acc[j][l]]) == 0: # If the list of non-accepting states that must be removed is empty, then
                                        Non_Accept_Remove[ind_acc[j][l]].append(non_acc_states[w]) # add to list of Non-Accepting states to be removed in the BSCC.
                                        Keep_Going = 1 # Stops checking the state, because we know it has to be removed
                                    elif Keep_Going == 0: # when there are no states to be removed in the BSCC
                                        break
                            if Check_Tag == 0: 
                                Accept.append(acc_states[j]) # add the list of accepting states
                                Acc_Tag = 1  
                

                if Acc_Tag == 1: #If the potential greatest BSCC is accepting
                    #Compute the policy that maximizes the lower bound probability of reaching unmatched accepting states 
                    SCC.sort()
                    Accept.sort()
                    Potential_Policy_BSCC = np.zeros(len(SCC))
                    Permanent_Policy_BSCC = np.zeros(len(SCC))  
                    for i in range(len(Accept)):   #Converts indices of SCC for reachability computation
                        Act1 = Al_Act_Perm[Ind[Accept[i]]][0]
                        Act2 = Al_Act_Pot[Ind[Accept[i]]][0] 
                        Accept[i] = SCC.index(Accept[i])
                        Permanent_Policy_BSCC[Accept[i]] = Act1
                        Potential_Policy_BSCC[Accept[i]] = Act2
                           
                    # Creating the list of reachable states etc.  (very computationally inefficient)
                    BSCC_Reachable_States = []
                    Indices = []
                    for i in range(IA1_l.shape[0]):
                        BSCC_Reachable_States.append([])
                        for j in range(len(SCC)):
                            if i == 0:
                                Indices.append(Ind[SCC[j]])
                            BSCC_Reachable_States[-1].append([])                    
                    
                    IA1_l_BSCC = np.array(IA1_l[:,Indices,:])
                    IA1_l_BSCC = np.array(IA1_l_BSCC[:,:,Indices])
                    IA1_u_BSCC = np.array(IA1_u[:,Indices,:])
                    IA1_u_BSCC = np.array(IA1_u_BSCC[:,:,Indices])
                    

                    for i in range(len(SCC)):
                        for j in range(len(SCC)):
                            for l in range(len(Al_Act_Pot[Ind[SCC[i]]])):
                                if IA1_u_BSCC[Al_Act_Pot[Ind[SCC[i]]][l], i,j] > 0:
                                    BSCC_Reachable_States[Al_Act_Pot[Ind[SCC[i]]][l]][i].append(j)
                    
                    
                    BSCC_Allowed_Actions = []
                    for i in range(len(Indices)):
                        BSCC_Allowed_Actions.append(list(Al_Act_Pot[Indices[i]]))  
                    
                    # Computes the optimal action to maximize the upper-bound   
                    (Dummy_Reach, Dummy_Upp_Bounds, Dummy_Chain, Potential_Policy_BSCC, Dum) = Maximize_Upper_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Accept, 0, 0, BSCC_Reachable_States, [], Potential_Policy_BSCC, BSCC_Allowed_Actions, [], []) # Minimizes Upper Bound
                    for i in range(len(SCC)):
                        Potential_Policy[Ind[SCC[i]]] = Potential_Policy_BSCC[i]
                        G_Pot_Acc_BSCCs.append(Ind[SCC[i]])

            

                    Maybe_Permanent.append(SCC)
                    
                                       
                    # Now, need to check if the BSCC doesn't leak, and if it doesn't, we can directly check if it is permanent or not   
                                       
                    if Inevitable == 1: #The current allowed actions cannot make the BSCC leak. To check for permanence (that is, no possibility of creating a sub non-accepting BSCC with this BSCC), we compute the policy that maximizes the lower_bound probability of reaching the accepting states. If this lower bound is zero for some states, then these are not permanent with respect to the BSCC


                          
                        
                        (Dummy_Reach, Dummy_Low_Bounds, Dummy_Chain, Permanent_Policy_BSCC, Dum) = Maximize_Lower_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Accept, 0, 0, BSCC_Reachable_States, [], Permanent_Policy_BSCC, BSCC_Allowed_Actions, [], []) # Minimizes Upper Bound
                        Bad_States = []
                        for i in range(len(Dummy_Low_Bounds)):
                            if Dummy_Low_Bounds[i] == 0: #If some states have a lower-bound zero of reaching an accepting state inside the BSCC, then it means there will always be a scenario where those states form a non-accepting BSCC, and therefore cannot be part of a permanent BSCC
                                Bad_States.append(SCC[i])
                        if len(Bad_States) == 0:                            
                            List_Permanent_Acc_BSCC.append([])
                            for i in range(len(SCC)):                                
                                Permanent_Policy[Ind[SCC[i]]] = Permanent_Policy_BSCC[i]
                                Potential_Policy[Ind[SCC[i]]] = Permanent_Policy[Ind[SCC[i]]] #Make both policies equal to avoid any bridge state
                                G_Per_Acc_BSCCs.append(Ind[SCC[i]])
                                if Ind[SCC[i]] not in G_Pot_Acc_BSCCs:
                                    G_Pot_Acc_BSCCs.append(Ind[SCC[i]])
                                Is_In_Permanent_Comp[Ind[SCC[i]]] = 1
                                List_Permanent_Acc_BSCC[-1].append(Ind[SCC[i]])
                            Maybe_Permanent.pop()
                            
                        else:
                                  

                            SCC_New = list(set(SCC) - set(Bad_States)) #Create new set of states without the states to be removed
                            SCC_New.sort()        
                        #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                            if len(SCC_New) != 0: # if some states are left in the SCC
                                New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                                for i in range(len(SCC_New)):
                                    for k in range(len(Al_Act_Pot[Ind[SCC_New[i]]])):
                                        for j in range(len(SCC_New)):
                                            if IA1_u[Al_Act_Pot[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                                New_G[i,j] = 1
                                    
                                C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label
                              
                                for j in range(len(C_new)):
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states
                                    C.append(C_new[j]) # put them in the front.
                                    SCC_Status.append(1)
                          

                            SCC_New = list(Bad_States) #Create new set of states without the states to be removed
                            SCC_New.sort()        
                        #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                            if len(SCC_New) != 0: # if some states are left in the SCC
                                New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                                for i in range(len(SCC_New)):
                                    for k in range(len(Al_Act_Pot[Ind[SCC_New[i]]])):
                                        for j in range(len(SCC_New)):
                                            if IA1_u[Al_Act_Pot[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                                New_G[i,j] = 1
                                    
                                C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label
                              
                                for j in range(len(C_new)):
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states
                                    C.append(C_new[j]) # put them in the front.
                                    SCC_Status.append(1)                            
                             
                                                  
                    else:
                        #We now need to check this SCC for permanence
                        C.append(SCC)
                        SCC_Status.append(1)
                        
                                                    
                else: #If it is not accepting, then we need to remove the non-accepting states prevening it from being accepting                    
                    
                    Check_Tag2 = 0
                    Count_Duplicates = 0
                    #We now have BSCCs which share the same states. Need to figure out what to do with respect to the actions
                    
                    
                    for j in range(len(Non_Accept_Remove)):
                        if len(Non_Accept_Remove[j]) != 0:
                            Count_Duplicates += 1
                    
                    for j in range(len(Non_Accept_Remove)): #Loop through the set of states to remove
                        if len(Non_Accept_Remove[j]) != 0:
                    
                            if Check_Tag2 == 0 and Count_Duplicates > 1:
                                Duplicate_Actions = copy.deepcopy(Al_Act_Pot)
                                Has_Found_BSCC_Status_2.append(0)
                                List_Found_BSCC_Status_2.append([])
                                Original_SCC_Status_2.append(SCC)
                                Check_Tag2 = 1
                            SCC_New = list(set(SCC) - set(Non_Accept_Remove[j])) #Create new set of states without the states to be removed
                            SCC_New.sort()

                        #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                            if len(SCC_New) != 0: # if some states are left in the SCC
                                New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                                for i in range(len(SCC_New)):
                                    for k in range(len(Al_Act_Pot[Ind[SCC_New[i]]])):
                                        for j in range(len(SCC_New)):
                                            if IA1_u[Al_Act_Pot[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                                New_G[i,j] = 1
                                    
                                C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label
                                
                                for j in range(len(C_new)):
                                    if Count_Duplicates > 1:
                                        Status2_Act.append(Duplicate_Actions)
                                        Which_Status2_BSCC.append(Number_Duplicates2)                                    
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states
                                    
                                    C.append(C_new[j]) # put them in the front.
                                   
                                    if Count_Duplicates > 1:
                                        SCC_Status.append(2)
                                    else:    
                                        SCC_Status.append(0)
                                        
                    if Check_Tag2 == 1 and Count_Duplicates > 1:
                        Number_Duplicates2 += 1

                        
            elif SCC_Status[m] == 1: #Checking for permanence of the SCC
                

                ind_acc = [] #Contains the Automaton indices of accepting states in BSCC
                acc_states = [] #Contains the Accepting States in BSCC
                ind_non_acc = [] #Contains the Automaton indices of non-accepting states in BSCC
                non_acc_states = [] #Contans the non Accepting States in BSCC
                Inevitable = 1 #Tag to see if BSCC is an inevitable BSCC
                
                               
                
                for j in range(len(SCC)):
                                       
                    if Is_Acc[Ind[SCC[j]]] == 1: # accepting, then add it as accepting and vice versa
                        acc_states.append(SCC[j])
                        indices = [] # Establish index list
                        for n in range(len(Wh_Acc_Pair[Ind[SCC[j]]])): # loop through to find which accepting pair-conditions are sufficed by the given state
                            indices.append(Wh_Acc_Pair[Ind[SCC[j]]][n]) # Add the respective absolute index/label for the states
                        ind_acc.append(indices) # Add the accumulated list of indices with respect to the BSCC

                    if Is_NAcc[Ind[SCC[j]]] == 1: # do the same thing for non-accepting states in the BSCC
                        non_acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Wh_NAcc_Pair[Ind[SCC[j]]])):
                            indices.append(Wh_NAcc_Pair[Ind[SCC[j]]][n])
                        ind_non_acc.append(indices)                         
                          
                Acc_Tag = 0
                Accept = [] #Contains unmatched accepting states
                                                    
                if len(non_acc_states) == 0: # If there are no non-accepting states,
                    Acc_Tag = 1 # then activate tag for accepting BSCC.
                    for j in range(len(acc_states)): # Subsequently, add to the list of accepting BSCC.
                        Accept.append(acc_states[j])
                
                else:                                        
                  
                    Non_Accept_Remove = [[] for x in range(len(Acc))] # Contains all non-accepting states which prevent the bscc to be accepting for all pairs                        
                    for j in range(len(ind_acc)): # Recall that ind_acc contains the accumulated list of indices of the pairs with respect to the BSCC
                        for l in range(len(ind_acc[j])): # ind_acc[j] contains the indices for the relevant DRA pair, to which the state of the BSCC was complying for acceptance
                            Check_Tag = 0
                            Keep_Going = 0
                            for w in range(len(ind_non_acc)): # Same thing for non accepting
                                if ind_acc[j][l] in ind_non_acc[w]: # Checks if accepting index is in list of non_accepting indices ?? YES
                                    Check_Tag = 1 # Means that the current index in the accepting states doesn't make the BSCC accepting
                                    if len(Non_Accept_Remove[ind_acc[j][l]]) == 0: # If the list of non-accepting states that must be removed is empty, then
                                        Non_Accept_Remove[ind_acc[j][l]].append(non_acc_states[w]) # add to list of Non-Accepting states to be removed in the BSCC.
                                        Keep_Going = 1 # Stops checking the state, because we know it has to be removed
                                    elif Keep_Going == 0: # when there are no states to be removed in the BSCC
                                        break
                            if Check_Tag == 0: 
                                Accept.append(acc_states[j]) # add the list of accepting states
                                Acc_Tag = 1  


                if Acc_Tag == 1: #If the BSCC is accepting
                   
                    
                    
                    #Compute the policy that maximizes the lower bound probability of reaching unmatched accepting states 
                    SCC.sort()
                    Accept.sort()
                    Permanent_Policy_BSCC = np.zeros(len(SCC))
                    for i in range(len(Accept)):   #Converts indices of SCC for reachability computation                       
                        Act = Al_Act_Perm[Ind[Accept[i]]][0]
                        Accept[i] = SCC.index(Accept[i])
                        Permanent_Policy_BSCC[i] = Act
                        
                                      
                    # Creating the list of reachable states etc.  (very computationally inefficient)
                    BSCC_Reachable_States = []
                    Indices = []
                    for i in range(IA1_l.shape[0]):
                        BSCC_Reachable_States.append([])
                        for j in range(len(SCC)):
                            if i == 0:
                                Indices.append(Ind[SCC[j]])
                            BSCC_Reachable_States[-1].append([])                    
                    
                    IA1_l_BSCC = np.array(IA1_l[:,Indices,:])
                    IA1_l_BSCC = np.array(IA1_l_BSCC[:,:,Indices])
                    IA1_u_BSCC = np.array(IA1_u[:,Indices,:])
                    IA1_u_BSCC = np.array(IA1_u_BSCC[:,:,Indices])
                    
                    for i in range(len(SCC)):
                        for j in range(len(SCC)):
                            for l in range(len(Al_Act_Perm[Ind[SCC[i]]])):
                                if IA1_u_BSCC[Al_Act_Perm[Ind[SCC[i]]][l], i,j] > 0:
                                    BSCC_Reachable_States[Al_Act_Perm[Ind[SCC[i]]][l]][i].append(j)
                    
                    
                    BSCC_Allowed_Actions = []
                    for i in range(len(Indices)):
                        BSCC_Allowed_Actions.append(list(Al_Act_Perm[Indices[i]]))  
                    
                    
                                     
                    (Dummy_Reach, Dummy_Low_Bounds, Dummy_Chain, Permanent_Policy_BSCC, Dum) = Maximize_Lower_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Accept, 0, 0, BSCC_Reachable_States, [], Permanent_Policy_BSCC, BSCC_Allowed_Actions, [], []) # Minimizes Upper Bound
                    Bad_States = []
                    for i in range(len(Dummy_Low_Bounds)):
                        if Dummy_Low_Bounds[i] == 0: #If some states have a lower-bound zero of reaching an accepting state inside the BSCC, then it means there will always be a scenario where those states form a non-accepting BSCC, and therefore cannot be part of a permanent BSCC
                            Bad_States.append(SCC[i])


                    if len(Bad_States) == 0:                    
                        if SCC not in List_Permanent_Acc_BSCC:
                            List_Permanent_Acc_BSCC.append([])
                            for i in range(len(SCC)):
                                Permanent_Policy[Ind[SCC[i]]] = Permanent_Policy_BSCC[i]
                                Potential_Policy[Ind[SCC[i]]] = Permanent_Policy[Ind[SCC[i]]] #To avoid bridge states during refinement
                                G_Per_Acc_BSCCs.append(Ind[SCC[i]])
                                Is_In_Permanent_Comp[Ind[SCC[i]]] = 1
                                List_Permanent_Acc_BSCC[-1].append(Ind[SCC[i]])
                    else:                      
                        SCC_New = list(set(SCC) - set(Bad_States)) #Create new set of states without the states to be removed
                        SCC_New.sort()          
                    #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                        if len(SCC_New) != 0: # if some states are left in the SCC
                            SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                            New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                            for i in range(len(SCC_New)):
                                for k in range(len(Al_Act_Perm[Ind[SCC_New[i]]])):
                                    for j in range(len(SCC_New)):
                                        if IA1_u[Al_Act_Perm[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                            New_G[i,j] = 1
                                
                            C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label              
                            for j in range(len(C_new)):
                                for k in range(len(C_new[j])):
                                    C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states
                                C.append(C_new[j]) # put them in the front.
                                SCC_Status.append(1)
                         
                        SCC_New = list(Bad_States) 
                        SCC_New.sort()          
                    #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                        if len(SCC_New) != 0: # if some states are left in the SCC
                            SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                            New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                            for i in range(len(SCC_New)):
                                for k in range(len(Al_Act_Perm[Ind[SCC_New[i]]])):
                                    for j in range(len(SCC_New)):
                                        if IA1_u[Al_Act_Perm[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                            New_G[i,j] = 1
                                
                            C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label              
                            for j in range(len(C_new)):
                                for k in range(len(C_new[j])):
                                    C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states
                                C.append(C_new[j]) # put them in the front.
                                SCC_Status.append(1)


                else: #If the BSCC is not accepting, then we need to remove the non-accepting states prevening it from being accepting                                        
                    #We now have BSCCs which share the same states. Need to figure out what to do with respect to the actions
                    Check_Tag3 = 0
                    
                    
                    Count_Duplicates = 0                   
                    for j in range(len(Non_Accept_Remove)):
                        if len(Non_Accept_Remove[j]) != 0:
                            Count_Duplicates += 1
                                                               
                            
                    for j in range(len(Non_Accept_Remove)): #Loop through the set of states to remove
                        if len(Non_Accept_Remove[j]) != 0: 
                            if Check_Tag3 == 0 and Count_Duplicates > 1:
                                Duplicate_Actions = copy.deepcopy(Al_Act_Pot)
                                Check_Tag3 = 1
                            SCC_New = list(set(SCC) - set(Non_Accept_Remove[j])) #Create new set of states without the states to be removed
                            SCC_New.sort()
                        #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                            if len(SCC_New) != 0: # if some states are left in the SCC
                                #SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                                New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                                for i in range(len(SCC_New)):
                                    for k in range(len(Al_Act_Perm[Ind[SCC_New[i]]])):
                                        for j in range(len(SCC_New)):
                                            if IA1_u[Al_Act_Perm[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                                New_G[i,j] = 1
                                  
                                C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label
                                for j in range(len(C_new)):

                                    if Count_Duplicates > 1:
                                        Status3_Act.append(Duplicate_Actions)
                                        Which_Status3_BSCC.append(Number_Duplicates3)
                                        List_Status3_Found.append([])#Will contain all permanent BSCCs found after duplication
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states                                       
                                    C.append(C_new[j]) # put them in the front.
                                    
                                    if Count_Duplicates > 1:
                                        
                                        SCC_Status.append(3)
                                    else:
                                        SCC_Status.append(1)
#                   
                    if Check_Tag3 == 1 and Count_Duplicates > 1:
                        Number_Duplicates3 += 1
                    
                        
            elif SCC_Status[m] == 2:
                
                ind_acc = [] #Contains the Automaton indices of accepting states in BSCC
                acc_states = [] #Contains the Accepting States in BSCC
                ind_non_acc = [] #Contains the Automaton indices of non-accepting states in BSCC
                non_acc_states = [] #Contans the non Accepting States in BSCC
                Inevitable = 1 #Tag to see if BSCC is an inevitable BSCC
                       
                
                
                for j in range(len(SCC)):
                                       
                    if Is_Acc[Ind[SCC[j]]] == 1: # accepting, then add it as accepting and vice versa
                        acc_states.append(SCC[j])
                        indices = [] # Establish index list
                        for n in range(len(Wh_Acc_Pair[Ind[SCC[j]]])): # loop through to find which accepting pair-conditions are sufficed by the given state
                            indices.append(Wh_Acc_Pair[Ind[SCC[j]]][n]) # Add the respective absolute index/label for the states
                        ind_acc.append(indices) # Add the accumulated list of indices with respect to the BSCC

                    if Is_NAcc[Ind[SCC[j]]] == 1: # do the same thing for non-accepting states in the BSCC
                        non_acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Wh_NAcc_Pair[Ind[SCC[j]]])):
                            indices.append(Wh_NAcc_Pair[Ind[SCC[j]]][n])
                        ind_non_acc.append(indices)                         
                          
                    for i in range(len(Status2_Act[Counter_Status2][Ind[SCC[j]]])): # For a given list of allowable actions for a given state in the SCC
                        if Is_Bridge_State[Status2_Act[Counter_Status2][Ind[SCC[j]]][i]][Ind[SCC[j]]] == 1: # Is the given state in the SCC a bridge state? If yes,                                                                  
                            Diff_List = np.setdiff1d(Reachable_States[Status2_Act[Counter_Status2][Ind[SCC[j]]][i]][Ind[SCC[j]]], Orig_SCC) # Subtract all states within SCC from the list of reachable states from the current state. Then-
                            if len(Diff_List) != 0: # is there anything that remains?
                                Inevitable = 0  # If so, then inevitability/permanence = 0, which means that the current SCC is no bueno in the permanency test
                            Bridge_States.append(Ind[SCC[j]]) # Then bridge_states list is constructed for the given SCC.
                            #not using this Bridge_States variable at the moment

                Acc_Tag = 0
                Accept = [] #Contains unmatched accepting states
                                                   
                if len(non_acc_states) == 0: # If there are no non-accepting states,
                    Acc_Tag = 1 # then activate tag for accepting BSCC.
                    for j in range(len(acc_states)): # Subsequently, add to the list of accepting BSCC.
                        Accept.append(acc_states[j])
                
                else:
                                               
                    Non_Accept_Remove = [[] for x in range(len(Acc))] # Contains all non-accepting states which prevent the bscc to be accepting for all pairs                        
                    for j in range(len(ind_acc)): # Recall that ind_acc contains the accumulated list of indices of the pairs with respect to the BSCC
                        for l in range(len(ind_acc[j])): # ind_acc[j] contains the indices for the relevant DRA pair, to which the state of the BSCC was complying for acceptance
                            Check_Tag = 0
                            Keep_Going = 0
                            for w in range(len(ind_non_acc)): # Same thing for non accepting
                                if ind_acc[j][l] in ind_non_acc[w]: # Checks if accepting index is in list of non_accepting indices ?? YES
                                    Check_Tag = 1 # Means that the current index in the accepting states doesn't make the BSCC accepting
                                    if len(Non_Accept_Remove[ind_acc[j][l]]) == 0: # If the list of non-accepting states that must be removed is empty, then
                                        Non_Accept_Remove[ind_acc[j][l]].append(non_acc_states[w]) # add to list of Non-Accepting states to be removed in the BSCC.
                                        Keep_Going = 1 # Stops checking the state, because we know it has to be removed
                                    elif Keep_Going == 0: # when there are no states to be removed in the BSCC
                                        break
                            if Check_Tag == 0: 
                                Accept.append(acc_states[j]) # add the list of accepting states
                                Acc_Tag = 1  
                

                if Acc_Tag == 1: #If the potential greatest BSCC is accepting
                    #Compute the policy that maximizes the lower bound probability of reaching unmatched accepting states 
                    
                    Has_Found_BSCC_Status_2[Which_Status2_BSCC[Counter_Status2]] = 1
                    SCC.sort()
                    Accept.sort()
                    Potential_Policy_BSCC = np.zeros(len(SCC)) 
                    for i in range(len(Accept)):   #Converts indices of SCC for reachability computation
                        Act = Al_Act_Pot[Ind[Accept[i]]][0]
                        Accept[i] = SCC.index(Accept[i])
                        Potential_Policy_BSCC[i] = Act
                                      
                    # Creating the list of reachable states etc.  (very computationally inefficient)
                    BSCC_Reachable_States = []
                    Indices = []
                    for i in range(IA1_l.shape[0]):
                        BSCC_Reachable_States.append([])
                        for j in range(len(SCC)):
                            if i == 0:
                                Indices.append(Ind[SCC[j]])
                            BSCC_Reachable_States[-1].append([])                    
                    
                    IA1_l_BSCC = np.array(IA1_l[:,Indices,:])
                    IA1_l_BSCC = np.array(IA1_l_BSCC[:,:,Indices])
                    IA1_u_BSCC = np.array(IA1_u[:,Indices,:])
                    IA1_u_BSCC = np.array(IA1_u_BSCC[:,:,Indices])
                    

                    for i in range(len(SCC)):
                        for j in range(len(SCC)):
                            for l in range(len(Status2_Act[Counter_Status2][Ind[SCC[i]]])):
                                if IA1_u_BSCC[Status2_Act[Counter_Status2][Ind[SCC[i]]][l], i,j] > 0:
                                    BSCC_Reachable_States[Status2_Act[Counter_Status2][Ind[SCC[i]]][l]][i].append(j)
                    
                    
                    BSCC_Allowed_Actions = []
                    for i in range(len(Indices)):
                        BSCC_Allowed_Actions.append(list(Status2_Act[Counter_Status2][Indices[i]]))  
                    
                    # Computes the optimal action to maximize the upper-bound   
                    (Dummy_Reach, Dummy_Upp_Bounds, Dummy_Chain, Potential_Policy_BSCC, Dum) = Maximize_Upper_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Accept, 0, 0, BSCC_Reachable_States, [], Potential_Policy_BSCC, BSCC_Allowed_Actions, [], []) # Minimizes Upper Bound
                    for i in range(len(SCC)):
                        Potential_Policy[Ind[SCC[i]]] = Potential_Policy_BSCC[i]
                        List_Found_BSCC_Status_2[Which_Status2_BSCC[Counter_Status2]].append(Ind[SCC[i]])
                         
                    C.append(Original_SCC_Status_2[Which_Status2_BSCC[Counter_Status2]]) #Feeding the original SCC for Permanence check
                    
                    SCC_Status.append(1)                

                else: #If it is not accepting, then we need to remove the non-accepting states prevening it from being accepting                    
                    
                    
                    Check_Tag2 = 0
                    #We now have BSCCs which share the same states. Need to figure out what to do with respect to the actions
                    
                    for j in range(len(Non_Accept_Remove)): #Loop through the set of states to remove
                        if len(Non_Accept_Remove[j]) != 0:
                            if Check_Tag2 == 0:
                                Duplicate_Actions = copy.deepcopy(Status2_Act[Counter_Status2])
                                Check_Tag2 = 1
                            SCC_New = list(set(SCC) - set(Non_Accept_Remove[j])) #Create new set of states without the states to be removed
                            SCC_New.sort()

                        #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                            if len(SCC_New) != 0: # if some states are left in the SCC
                                New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                                for i in range(len(SCC_New)):
                                    for k in range(len(Al_Act_Pot[Ind[SCC_New[i]]])):
                                        for j in range(len(SCC_New)):
                                            if IA1_u[Al_Act_Pot[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                                New_G[i,j] = 1
                                    
                                C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label
                                
                                for j in range(len(C_new)):
                                    Status2_Act.append(Duplicate_Actions)
                                    Which_Status2_BSCC.append(Which_Status2_BSCC[Counter_Status2])                                   
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states
                                    
                                    C.append(C_new[j]) # put them in the front.
                                    SCC_Status.append(2)
                        

                                                                            
                Counter_Status2 +=1


            elif SCC_Status[m] == 3:  

                
                ind_acc = [] #Contains the Automaton indices of accepting states in BSCC
                acc_states = [] #Contains the Accepting States in BSCC
                ind_non_acc = [] #Contains the Automaton indices of non-accepting states in BSCC
                non_acc_states = [] #Contans the non Accepting States in BSCC
                
                for j in range(len(SCC)):
                                       
                    if Is_Acc[Ind[SCC[j]]] == 1: # accepting, then add it as accepting and vice versa
                        acc_states.append(SCC[j])
                        indices = [] # Establish index list
                        for n in range(len(Wh_Acc_Pair[Ind[SCC[j]]])): # loop through to find which accepting pair-conditions are sufficed by the given state
                            indices.append(Wh_Acc_Pair[Ind[SCC[j]]][n]) # Add the respective absolute index/label for the states
                        ind_acc.append(indices) # Add the accumulated list of indices with respect to the BSCC

                    if Is_NAcc[Ind[SCC[j]]] == 1: # do the same thing for non-accepting states in the BSCC
                        non_acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Wh_NAcc_Pair[Ind[SCC[j]]])):
                            indices.append(Wh_NAcc_Pair[Ind[SCC[j]]][n])
                        ind_non_acc.append(indices)                         
                          
                Acc_Tag = 0
                Accept = [] #Contains unmatched accepting states
                                                    
                if len(non_acc_states) == 0: # If there are no non-accepting states,
                    Acc_Tag = 1 # then activate tag for accepting BSCC.
                    for j in range(len(acc_states)): # Subsequently, add to the list of accepting BSCC.
                        Accept.append(acc_states[j])
                
                else:                                        
                  
                    Non_Accept_Remove = [[] for x in range(len(Acc))] # Contains all non-accepting states which prevent the bscc to be accepting for all pairs                        
                    for j in range(len(ind_acc)): # Recall that ind_acc contains the accumulated list of indices of the pairs with respect to the BSCC
                        for l in range(len(ind_acc[j])): # ind_acc[j] contains the indices for the relevant DRA pair, to which the state of the BSCC was complying for acceptance
                            Check_Tag = 0
                            Keep_Going = 0
                            for w in range(len(ind_non_acc)): # Same thing for non accepting
                                if ind_acc[j][l] in ind_non_acc[w]: # Checks if accepting index is in list of non_accepting indices ?? YES
                                    Check_Tag = 1 # Means that the current index in the accepting states doesn't make the BSCC accepting
                                    if len(Non_Accept_Remove[ind_acc[j][l]]) == 0: # If the list of non-accepting states that must be removed is empty, then
                                        Non_Accept_Remove[ind_acc[j][l]].append(non_acc_states[w]) # add to list of Non-Accepting states to be removed in the BSCC.
                                        Keep_Going = 1 # Stops checking the state, because we know it has to be removed
                                    elif Keep_Going == 0: # when there are no states to be removed in the BSCC
                                        break
                            if Check_Tag == 0: 
                                Accept.append(acc_states[j]) # add the list of accepting states
                                Acc_Tag = 1  


                if Acc_Tag == 1: #If the BSCC is accepting                
                
                    #Compute the policy that maximizes the lower bound probability of reaching unmatched accepting states 
                    SCC.sort()
                    Accept.sort()
                    Permanent_Policy_BSCC = np.zeros(len(SCC))  
                    for i in range(len(Accept)):   #Converts indices of SCC for reachability computation
                        Act = Al_Act_Perm[Ind[Accept[i]]][0]
                        Accept[i] = SCC.index(Accept[i])
                        Permanent_Policy_BSCC[i] = Act
                          
                                    
                    # Creating the list of reachable states etc.  (very computationally inefficient)
                    BSCC_Reachable_States = []
                    Indices = []
                    for i in range(IA1_l.shape[0]):
                        BSCC_Reachable_States.append([])
                        for j in range(len(SCC)):
                            if i == 0:
                                Indices.append(Ind[SCC[j]])
                            BSCC_Reachable_States[-1].append([])                    
                    
                    IA1_l_BSCC = np.array(IA1_l[:,Indices,:])
                    IA1_l_BSCC = np.array(IA1_l_BSCC[:,:,Indices])
                    IA1_u_BSCC = np.array(IA1_u[:,Indices,:])
                    IA1_u_BSCC = np.array(IA1_u_BSCC[:,:,Indices])
                    
                    for i in range(len(SCC)):
                        for j in range(len(SCC)):
                            for l in range(len(Status3_Act[Counter_Status3][Ind[SCC[i]]])):
                                if IA1_u_BSCC[Status3_Act[Counter_Status3][Ind[SCC[i]]][l], i,j] > 0:
                                    BSCC_Reachable_States[Status3_Act[Counter_Status3][Ind[SCC[i]]][l]][i].append(j)
                    
                    
                    BSCC_Allowed_Actions = []
                    for i in range(len(Indices)):
                        BSCC_Allowed_Actions.append(list(Status3_Act[Counter_Status3][Indices[i]]))  
                    
                                     
                    (Dummy_Reach, Dummy_Low_Bounds, Dummy_Chain, Permanent_Policy_BSCC, Dum) = Maximize_Lower_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Accept, 0, 0, BSCC_Reachable_States, [], Permanent_Policy_BSCC, BSCC_Allowed_Actions, [], []) # Minimizes Upper Bound
                    Bad_States = []
                    for i in range(len(Accept)):
                        Permanent_Policy_BSCC[Accept[i]] = Status3_Act[Counter_Status3][Ind[SCC[Accept[i]]]][0]
                        
                    for i in range(len(Dummy_Low_Bounds)):
                        if Dummy_Low_Bounds[i] == 0: #If some states have a lower-bound zero of reaching an accepting state inside the BSCC, then it means there will always be a scenario where those states form a non-accepting BSCC, and therefore cannot be part of a permanent BSCC
                            Bad_States.append(SCC[i])
                    
                    if len(Bad_States) == 0:
                        Existing_Lists = []
                        for i in range(len(List_Status3_Found[Which_Status3_BSCC[Counter_Status3]])):
                            Existing_Lists.append(List_Status3_Found[Which_Status3_BSCC[Counter_Status3]][i][0])
                        
                        if SCC not in Existing_Lists:
                            List_Status3_Found[Which_Status3_BSCC[Counter_Status3]].append([])
                            List_Status3_Found[Which_Status3_BSCC[Counter_Status3]][-1].append([]) #First list will contain the states, second list the actions
                            List_Status3_Found[Which_Status3_BSCC[Counter_Status3]][-1].append([]) #First list will contain the states, second list the actions                        
                            for i in range(len(SCC)):                            
                                List_Status3_Found[Which_Status3_BSCC[Counter_Status3]][-1][0].append(Ind[SCC[i]])
                                List_Status3_Found[Which_Status3_BSCC[Counter_Status3]][-1][1].append(Permanent_Policy_BSCC[i])
                                Is_In_Permanent_Comp[Ind[SCC[i]]] = 1
                    else:                      
                        SCC_New = list(set(SCC) - set(Bad_States)) #Create new set of states without the states to be removed
                        SCC_New.sort()          
                    #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                        if len(SCC_New) != 0: # if some states are left in the SCC
                            Duplicate_Actions = copy.deepcopy(Status3_Act[Counter_Status3])
                            SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                            New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                            for i in range(len(SCC_New)):
                                for k in range(len(Status3_Act[Counter_Status3][Ind[SCC_New[i]]])):
                                    for j in range(len(SCC_New)):
                                        if IA1_u[Status3_Act[Counter_Status3][Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                            New_G[i,j] = 1
                                
                            C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label              
                            for j in range(len(C_new)):
                                Status3_Act.append(Duplicate_Actions)
                                Which_Status3_BSCC.append(Which_Status3_BSCC[Counter_Status3])                                
                                for k in range(len(C_new[j])):
                                    C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states
                                C.append(C_new[j]) # put them in the front.
                                SCC_Status.append(3)
  

                        SCC_New = list(Bad_States) #Create new set of states without the states to be removed
                        SCC_New.sort()          
                    #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                        if len(SCC_New) != 0: # if some states are left in the SCC
                            Duplicate_Actions = copy.deepcopy(Status3_Act[Counter_Status3])
                            SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                            New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                            for i in range(len(SCC_New)):
                                for k in range(len(Status3_Act[Counter_Status3][Ind[SCC_New[i]]])):
                                    for j in range(len(SCC_New)):
                                        if IA1_u[Status3_Act[Counter_Status3][Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                            New_G[i,j] = 1
                                
                            C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label              
                            for j in range(len(C_new)):
                                Status3_Act.append(Duplicate_Actions)
                                Which_Status3_BSCC.append(Which_Status3_BSCC[Counter_Status3])                                
                                for k in range(len(C_new[j])):
                                    C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states
                                C.append(C_new[j]) # put them in the front.
                                SCC_Status.append(3)                       

                else: #If the BSCC is not accepting, then we need to remove the non-accepting states prevening it from being accepting                                        
                    #We now have BSCCs which share the same states. Need to figure out what to do with respect to the actions

                    
                    Check_Tag3 = 0 
                    for j in range(len(Non_Accept_Remove)): #Loop through the set of states to remove
                        if len(Non_Accept_Remove[j]) != 0: 
                            if Check_Tag3 == 0:
                                Duplicate_Actions = copy.deepcopy(Status3_Act[Counter_Status3])
                                Check_Tag3 = 1
                            SCC_New = list(set(SCC) - set(Non_Accept_Remove[j])) #Create new set of states without the states to be removed
                            SCC_New.sort()
                        #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                            if len(SCC_New) != 0: # if some states are left in the SCC
                                New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                                for i in range(len(SCC_New)):
                                    for k in range(len(Status3_Act[Counter_Status3][Ind[SCC_New[i]]])):
                                        for j in range(len(SCC_New)):
                                            if IA1_u[Status3_Act[Counter_Status3][Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                                New_G[i,j] = 1
                                  
                                C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label
                                for j in range(len(C_new)):
                                    Status3_Act.append(Duplicate_Actions)
                                    Which_Status3_BSCC.append(Which_Status3_BSCC[Counter_Status3])
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states
                                    C.append(C_new[j]) # put them in the front.
                                    SCC_Status.append(3)
#                   
                
                Counter_Status3 +=1                 
                    
        m +=1
        if m == len(C): tag = 1
      
    for i in range(len(Maybe_Permanent)): #Looping through the potential BSCC to see which ones turned out to be permanent
        List_Potential_States = []
        List_Bridge_States = []
        BSCC_Converted_Indices = []
        for j in range(len(Maybe_Permanent[i])):            
            if Is_In_Permanent_Comp[Ind[Maybe_Permanent[i][j]]] == 0: #If the state turns out not to be a permanent component
                BSCC_Converted_Indices.append(Ind[Maybe_Permanent[i][j]])
                List_Potential_States.append(Ind[Maybe_Permanent[i][j]])
                Which_Potential_Acc_BSCC[Ind[Maybe_Permanent[i][j]]] = len(List_G_Pot)
                Is_In_Potential_Acc_BSCC[Ind[Maybe_Permanent[i][j]]] = 1
                if Is_Bridge_State[Potential_Policy[Ind[Maybe_Permanent[i][j]]]][Ind[Maybe_Permanent[i][j]]] == 1:
                    List_Bridge_States.append(Ind[Maybe_Permanent[i][j]])
        if len(List_Potential_States) != 0: #That is, the BSCC is not entirely Permanent 
            List_G_Pot.append(BSCC_Converted_Indices)
            Bridge_Potential_Accepting.append(List_Bridge_States)



            
    for i in range(Number_Duplicates2): #Taking care of the states who were in duplicate SCCs when searching potential BSCCs

        if Has_Found_BSCC_Status_2[i] == 1: #We now know that the original SCC (which was not accepting due to some non-accepting states) can potentially be made accepting        
            Non_Permanent_States = []
            for j in range(len(Original_SCC_Status_2[i])):
                G_Pot_Acc_BSCCs.append(Ind[Original_SCC_Status_2[i][j]])
                if Is_In_Permanent_Comp[Ind[Original_SCC_Status_2[i][j]]] == 0:
                    Non_Permanent_States.append(Ind[Original_SCC_Status_2[i][j]])
                    Is_In_Potential_Acc_BSCC[Ind[Original_SCC_Status_2[i][j]]] = 1
                    Which_Potential_Acc_BSCC[Ind[Original_SCC_Status_2[i][j]]] = len(List_G_Pot)
            List_G_Pot.append(Non_Permanent_States) 
            Remaining_States = list(set(Original_SCC_Status_2[i]) - set(List_Found_BSCC_Status_2[i]))
            for j in range(len(Remaining_States)):
                Potential_Policy[Ind[Remaining_States[i]]] = Al_Act_Pot[Ind[Remaining_States[i]]][0] #Any action that could generate the duplicated BSCC works                
            List_Bridge_States = []
            for j in range(len(Non_Permanent_States)):          
                if Is_Bridge_State[Potential_Policy[Non_Permanent_States[j]]][Non_Permanent_States[j]] == 1: 
                    List_Bridge_States.append(Non_Permanent_States[j])
            Bridge_Potential_Accepting.append(List_Bridge_States) 


                       
    for i in range(Number_Duplicates3): #Taking care of the states who were in duplicate SCCs when searching permanent BSCCs
        if (len(List_Status3_Found[i])!= 0):
            Graph = np.zeros((len(List_Status3_Found[i]),len(List_Status3_Found[i])))
            for j in range(len(List_Status3_Found[i])): #Check connectedness of all states
                Graph[j,j] = 1
                for k in range(j+1, len(List_Status3_Found[i])):
                                     
                    if (set(List_Status3_Found[i][j][0]).intersection(set(List_Status3_Found[i][k][0]))) != 0:
                        Graph[j,k] = 1
                        Graph[k,j] = 1
            
            

            Comp_Graph = csr_matrix(Graph)
            Num_Comp, labels =  connected_components(csgraph=Comp_Graph, directed=False, return_labels=True)            
            C = [[] for x in range(Num_Comp)]
    
            for k in range(len(labels)):
                C[labels[i]].append(i)
                
            for k in range(len(C)):
        
                Component = []
                if len(C[k]) == 1:   #if the component is not connected to any other
                    for l in range(len(List_Status3_Found[i][C[k][0]][0])):

                        Permanent_Policy[Ind[List_Status3_Found[i][C[k][0]][0][l]]] = List_Status3_Found[i][C[k][0]][1][l]
                        Component.append(Ind[List_Status3_Found[i][C[k][0]][0][l]])
                    
                    List_Permanent_Acc_BSCC.append(Component)
  
                      
                else:
                                        
                    States_To_Reach = []
                    for l in range(len(List_Status3_Found[i][C[k][0]][0])):
                        Permanent_Policy[Ind[List_Status3_Found[i][C[k][0]][0][l]]] = List_Status3_Found[i][C[k][0]][1][l]
                        States_To_Reach.append(Ind[List_Status3_Found[i][C[k][0]][0][l]])
                        Component.append(Ind[List_Status3_Found[i][C[k][0]][0][l]])
                   
                    States_For_Reachability = []
                    for l in range(1, len(C[k])):
                        for m in range(len(List_Status3_Found[i][C[k][l]][0])):
                            if Ind[List_Status3_Found[i][C[k][l]][0][m]] not in States_To_Reach:
                                States_For_Reachability.append(Ind[List_Status3_Found[i][C[k][l]][0][m]])
                                Component.append(Ind[List_Status3_Found[i][C[k][l]][0][m]])


                    Component.sort()
                    Policy_BSCC = np.zeros(len(Component))                   
                    # Creating the list of reachable states etc.  (very computationally inefficient)
                    BSCC_Reachable_States = []
                    Indices = []
                    for y in range(IA1_l.shape[0]):
                        BSCC_Reachable_States.append([])
                        for x in range(len(Component)):
                            if y == 0:
                                Indices.append(Component[x])
                            BSCC_Reachable_States[-1].append([])
                            
                    
                    Target = []
                    for y in range(len(States_To_Reach)):
                        Target.append(Indices.index(States_To_Reach[y]))

                    IA1_l_BSCC = np.array(IA1_l[:,Indices,:])
                    IA1_l_BSCC = np.array(IA1_l_BSCC[:,:,Indices])
                    IA1_u_BSCC = np.array(IA1_u[:,Indices,:])
                    IA1_u_BSCC = np.array(IA1_u_BSCC[:,:,Indices])
                    

                    for y in range(len(Component)):
                        for x in range(len(Component)):
                            for l in range(len(Al_Act_Perm[Component[y]])):
                                if IA1_u_BSCC[Al_Act_Perm[Component[y]][l], y,x] > 0:
                                    BSCC_Reachable_States[Al_Act_Perm[Component[y]][l]][y].append(x)                                        
                    BSCC_Allowed_Actions = []
                    for y in range(len(Indices)):
                        BSCC_Allowed_Actions.append(list(Al_Act_Perm[Indices[y]]))  
                    
                    # Computes the optimal action to maximize the upper-bound   
                    (Dummy_Reach, Dummy_Upp_Bounds, Dummy_Chain, Policy_BSCC, Dum) = Maximize_Lower_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Target, 0, 0, BSCC_Reachable_States, [], Policy_BSCC, BSCC_Allowed_Actions, [], [])

                    for y in range(len(States_For_Reachability)):
                        Permanent_Policy[States_For_Reachability[y]] = Policy_BSCC[Component.index(States_For_Reachability[y])]
                    
                    List_Permanent_Acc_BSCC.append(Component)
                    
                         



                       
    return G_Pot_Acc_BSCCs, G_Per_Acc_BSCCs, Potential_Policy, Permanent_Policy, Al_Act_Pot, Al_Act_Perm, first, Is_In_Permanent_Comp, List_Permanent_Acc_BSCC, List_G_Pot, Which_Potential_Acc_BSCC, Is_In_Potential_Acc_BSCC, Bridge_Potential_Accepting







def Find_Greatest_Winning_Components(IA1_l, IA1_u, Is_Accepting, Is_Non_Accepting, Which_Accepting_Pair, Which_Non_Accepting_Pair, Allowable_Actions, Product_Reachable_States, Product_Bridge_Transitions, Product_Is_Bridge_State, Automata_Accepting, Optimal_Policy, Is_In_Permanent_Comp, List_Permanent_Accepting_BSCC, List_Potential_Accepting_BSCC, Greatest_Permanent_Accepting_BSCCs, Is_In_Potential_Accepting_BSCC, Which_Potential_Accepting_BSCC, Bridge_Accepting_BSCC, Len_Automata, Init, Acc):
    
    
    Greatest_Permanent_Winning_Component = list(Greatest_Permanent_Accepting_BSCCs)
    

    
    #We allow all actions for now, will modify if too computationally expensive               
    Previous_Permanent = [] #List of previous permanent components
    Check_Loop = 0
    Potential_To_Delete = []
    Potential_To_Check = []
    

    
    (Low_Bound, Low_Bounds_Prod, Worst_Markov_Chain, Optimal_Policy, List_Values_Low) = Maximize_Lower_Bound_Reachability(IA1_l, IA1_u, Greatest_Permanent_Winning_Component, IA1_l.shape[1]/Len_Automata, Len_Automata, Product_Reachable_States, Init, Optimal_Policy, Allowable_Actions, [], []) # Maximizes Lower Bound

    States_To_Check = list(set(range(len(Low_Bounds_Prod))) - set(Greatest_Permanent_Accepting_BSCCs))
   
    for i in range(len(States_To_Check)):
        if Low_Bounds_Prod[States_To_Check[i]] == 1:
            Is_In_Permanent_Comp[States_To_Check[i]] = 1
            Greatest_Permanent_Winning_Component.append(States_To_Check[i])
            if Is_In_Potential_Accepting_BSCC[States_To_Check[i]] == 1:
                Is_In_Potential_Accepting_BSCC[States_To_Check[i]] = 0
                if Which_Potential_Accepting_BSCC[States_To_Check[i]] not in Potential_To_Check:
                    Potential_To_Check.append(Which_Potential_Accepting_BSCC[States_To_Check[i]])
                    Potential_To_Delete.append(Which_Potential_Accepting_BSCC[States_To_Check[i]])   
                List_Potential_Accepting_BSCC[Which_Potential_Accepting_BSCC[States_To_Check[i]]].remove(States_To_Check[i])
                
    for n in range(len(Potential_To_Check)):
        Current_BSCC = list(List_Potential_Accepting_BSCC[Potential_To_Check[n]])
        if len(Current_BSCC) == 0:
            continue
        
        Check_Accepting = 0 #If the set of states does not contain an accepting state, we can just continue
        for k in range(len(Current_BSCC)):
            if Is_Accepting[Current_BSCC[k]] == 1:
                Check_Accepting = 1
                break
                
        
        if Check_Accepting == 0:
            continue
        
        for k in range(len(Current_BSCC)):
            Is_In_Potential_Accepting_BSCC[Current_BSCC[k]] = 0
            
        
        IA1_l_Reduced = IA1_l[:,Current_BSCC, :]
        IA1_u_Reduced = IA1_u[:,Current_BSCC, :]
        
        
        #CREATE SINK STATES TO REPRESENT TRANSITIONS OUTSIDE OF THESE PREVIOUS COMPONENTS
        
        IA1_l_Reduced = np.append(IA1_l_Reduced, np.zeros((IA1_l_Reduced.shape[0], IA1_l_Reduced.shape[1], 1)), 2)
        IA1_u_Reduced = np.append(IA1_u_Reduced, np.zeros((IA1_u_Reduced.shape[0], IA1_u_Reduced.shape[1], 1)), 2)
              
        for i in range(IA1_l_Reduced.shape[0]):
            for j in range(IA1_l_Reduced.shape[1]):
                IA1_l_Reduced[i,j,-1] = sum(np.delete(IA1_l_Reduced[i][j][:],Current_BSCC))
                IA1_u_Reduced[i,j,-1] = min(sum(np.delete(IA1_u_Reduced[i][j][:],Current_BSCC)),1.0)

        IA1_l_Reduced = IA1_l_Reduced[:,:, list(np.append(Current_BSCC, IA1_l_Reduced.shape[2]-1))]
        IA1_u_Reduced = IA1_u_Reduced[:,:, list(np.append(Current_BSCC, IA1_u_Reduced.shape[2]-1))]

                
        IA1_l_Reduced = np.append(IA1_l_Reduced, np.zeros((IA1_l_Reduced.shape[0], 1, IA1_l_Reduced.shape[2])), 1)
        IA1_u_Reduced = np.append(IA1_u_Reduced, np.zeros((IA1_u_Reduced.shape[0], 1, IA1_u_Reduced.shape[2])), 1)

        #SELECTING THE BRIDGE STATES/TRANSITIONS/ACCEPTING/REACHABLE STATES AND NON ACCEPTING STATES OF THE REDUCED PRODUCT
        Product_Reachable_States_Reduced = [[] for x in range(IA1_l_Reduced.shape[0])]
        Product_Is_Bridge_State_Reduced = np.zeros((IA1_l_Reduced.shape[0],IA1_l_Reduced.shape[1]))
        Product_Bridge_Transitions_Reduced = [[] for x in range(IA1_l_Reduced.shape[0])]
        Which_Accepting_Pair_Reduced = [[] for x in range(IA1_l_Reduced.shape[1])]
        Which_Non_Accepting_Pair_Reduced = [[] for x in range(IA1_l_Reduced.shape[1])]
        Is_Accepting_Reduced = np.zeros(IA1_l_Reduced.shape[1])
        Is_Non_Accepting_Reduced = np.zeros(IA1_l_Reduced.shape[1])
        Is_In_Permanent_Comp_Reduced = np.zeros(IA1_l_Reduced.shape[1])
        Previous_Accepting_BSCC = []
        Indices = np.zeros(IA1_l.shape[1])
        Indices = Indices.astype(int)

        
        for i in range(len(Current_BSCC)):
            Indices[Current_BSCC[i]] = i
        
        for i in range(IA1_l_Reduced.shape[0]):
            IA1_l_Reduced[i][-1][-1] = 1.0
            IA1_u_Reduced[i][-1][-1] = 1.0 #Sink State transition values
            
            for j in range(IA1_l_Reduced.shape[1] - 1):
                
                if i == 0:
                    if Is_Accepting[Current_BSCC[j]] == 1:
                        Is_Accepting_Reduced[j] = 1
                        Which_Accepting_Pair_Reduced[j] = list(Which_Accepting_Pair[Current_BSCC[j]])
                    if Is_Non_Accepting[Current_BSCC[j]] == 1:
                        Is_Non_Accepting_Reduced[j] = 1                    
                        Which_Non_Accepting_Pair_Reduced[j] = list(Which_Non_Accepting_Pair[Current_BSCC[j]])
                
                        
                Reach = list([])
                Bridge = list([])
                Differential = list(set(Product_Reachable_States[i][Current_BSCC[j]]) - set(Current_BSCC))
                if len(Differential) != 0:
                    Reach.append(IA1_l_Reduced.shape[1] - 1) #Then the sink state is reachable
                    if set(Differential) == (set(Product_Bridge_Transitions[i][Current_BSCC[j]]) - set(Current_BSCC)):
                        Bridge.append(IA1_l_Reduced.shape[1] - 1) #If these transitions are bridges, then there is a bridge to the sink state
                
                List_Reach = list(set(Product_Reachable_States[i][Current_BSCC[j]]).intersection(Current_BSCC))
                for k in range(len(List_Reach)):
                    Reach.append(Indices[List_Reach[k]])

                List_Bridge = list(set(Product_Bridge_Transitions[i][Current_BSCC[j]]).intersection(Current_BSCC))    
                for k in range(len(List_Bridge)):
                    Bridge.append(Indices[List_Bridge[k]])
                Reach.sort()
                Bridge.sort()
                Product_Reachable_States_Reduced[i].append(Reach)
                Product_Is_Bridge_State_Reduced[i][j] = Product_Is_Bridge_State[i][Current_BSCC[j]]
                Product_Bridge_Transitions_Reduced[i].append(Bridge)
            

        first = 1;
        Allowable_Action_Potential = list([]) #Actions that could make the state a potential BSCC
        Allowable_Action_Permanent = list([]) #Actions that could make the state a permanent BSCC        
       
        Optimal_Policy_BSCC = np.zeros(IA1_l.shape[1])
        Optimal_Policy_BSCC = Optimal_Policy_BSCC.astype(int)           
        Potential_Policy_BSCC = np.zeros(IA1_l.shape[1]) #Policy to generate the "best" best-case (maximize upper bound)
        Potential_Policy_BSCC = Potential_Policy_BSCC.astype(int)
       
        (Greatest_Potential_Accepting_BSCCs_Reduced, Greatest_Permanent_Accepting_BSCCs_Reduced, Potential_Policy_BSCC, Optimal_Policy_BSCC, Allowable_Action_Potential, Allowable_Action_Permanent, first, Is_In_Permanent_Comp_Reduced, List_Permanent_Accepting_BSCC_Reduced, List_Potential_Accepting_BSCC_Reduced, Which_Potential_Accepting_BSCC_Reduced, Is_In_Potential_Accepting_BSCC_Reduced, Bridge_Accepting_BSCC_Reduced) = Find_Greatest_Accepting_BSCCs(IA1_l_Reduced, IA1_u_Reduced, Is_Accepting_Reduced, Is_Non_Accepting_Reduced, Which_Accepting_Pair_Reduced, Which_Non_Accepting_Pair_Reduced, Allowable_Action_Potential, Allowable_Action_Permanent, first, Product_Reachable_States_Reduced, Product_Bridge_Transitions_Reduced, Product_Is_Bridge_State_Reduced, Automata_Accepting, Potential_Policy_BSCC, Optimal_Policy_BSCC, Is_In_Permanent_Comp_Reduced, [], [range(IA1_l_Reduced.shape[1]-1)]) # Will return greatest potential accepting bsccs

        Length_List = len(List_Potential_Accepting_BSCC)
        for i in range(len(List_Potential_Accepting_BSCC_Reduced)):
           List_Potential_Accepting_BSCC.append(list([]))
           Bridge_Accepting_BSCC.append(list([]))
           for x in range(len(List_Potential_Accepting_BSCC_Reduced[i])):
               Is_In_Potential_Accepting_BSCC[Current_BSCC[List_Potential_Accepting_BSCC_Reduced[i][x]]] = 1
               List_Potential_Accepting_BSCC[-1].append(Current_BSCC[List_Potential_Accepting_BSCC_Reduced[i][x]])
               Which_Potential_Accepting_BSCC[Current_BSCC[List_Potential_Accepting_BSCC_Reduced[i][x]]] = Length_List+i

           for x in range(len(Bridge_Accepting_BSCC_Reduced[i])):
               Bridge_Accepting_BSCC[-1].append(Current_BSCC[Bridge_Accepting_BSCC_Reduced[i][x]])
       

        
        
    Previous_Permanent = list(Greatest_Permanent_Winning_Component)              
    Original_Allowable_Actions = copy.deepcopy(Allowable_Actions)


    
    while Check_Loop == 0:
        
        Allowable_Actions = copy.deepcopy(Original_Allowable_Actions)
        
        List_BSCCs = copy.deepcopy(List_Potential_Accepting_BSCC)
        for ele in sorted(Potential_To_Delete, reverse = True):  
            del List_BSCCs[ele] 
        
        for z in range(len(List_BSCCs)):
            
            List_All_BSCCs = [list(List_BSCCs[z])]
            tag = 0 
            m = 0
            Remove_From_BSCC = []
            
            if Is_In_Permanent_Comp[List_All_BSCCs[m][0]] == 1:
                continue
            
            while (tag == 0):
                

            
                Cur_BSCC = list(List_All_BSCCs[m])
                



                Orig_SCC = list(Cur_BSCC)
                BSCC = 0
                
                while(len(Cur_BSCC) != 0 or BSCC == 0):
                    Leak = list([])
                    Check_Tag = 1
                    BSCC = 1
                    All_Leaks = list([])
                    Check_Orig_SCC = np.zeros(len(Orig_SCC), dtype=int)
                    
                    
                    while (len(Leak) != 0 or Check_Tag == 1):
                    
                        ind_leak = []
                        Leak = []                     
                        for i in range(len(Orig_SCC)): # Original SCC that contains all SCCs
                            if Check_Orig_SCC[i] == -1 :
                                continue # -1 is a tag for leaky state which should be skipped over.
                            tag_m = 0# tag_mode
                             
                            for k in range(len(Allowable_Actions[Orig_SCC[i]])): # Loop through all the allowable actions from the current state
                                    # The state number if the index for the allowable action array                        
                                Set_All_Leaks = set(Orig_SCC) - set(All_Leaks) # Removes all leaky states from Orig_SCC
                                Diff_List1 = list(set(Product_Reachable_States[Allowable_Actions[Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - Set_All_Leaks.union(Greatest_Permanent_Winning_Component)) # All Reachable states (that is outside the Set_All_Leaks) of the current state. Al_Act[i][k] means action from the set of allowable actions with respect to current state i

                                if (len(Diff_List1) != 0): # If some state is reachable outside of the SCC, then the SCC is not permanent for wrt that action                           
                                    Allowable_Actions[Orig_SCC[i]].remove(Allowable_Actions[Orig_SCC[i]][k-tag_m]) #If the action cannot make a potential BSCC, then it cannot make a permanent BSCC either
                                    tag_m += 1 # To account for the missing index in the "for k" loop.
                                    BSCC = 0 # Because once action is removed, might have changed the whole SCC, so might as well just keep BSCC to be 0.
                                                            
                            if len(Allowable_Actions[Orig_SCC[i]]) == 0: # If there are no more available actions for the current state i.                        
                                Leak.append(Orig_SCC[i])
                                ind_leak.append(i)
                        if len(Leak) != 0: # It means that a new leaky state is found, because Leak=[] every loop of the while
                            All_Leaks.extend(Leak) # Then add to All_leaks. extend means "adding" a list without brackets
                            BSCC = 0 # To confirm that previous SCC is surely not a BSCC, until further verifications to see if SCC is a BSCC
                            for i in range(len(Leak)): # for all the newly found leaky states
                                Check_Orig_SCC[ind_leak[i]] = -1 # The state in the SCC is tagged "leaky"
                        Check_Tag = 0  # Changes Check_Tag after having populated the "Pre" and Reach_in_R for all the states in set of SCCs. But now need to do it for all actions (modes).
                    
                    if BSCC == 0: #The BSCC leaks all the time outside of the union of the permanent component and the bscc. We try to create a BSCC with a subset of these
                            
                        Current_BSCC = list(set(Cur_BSCC) - set(All_Leaks))
                        Current_BSCC.sort()
                        
                        if len(Current_BSCC) != 0:
                                                        
                        
                            IA1_l_Reduced = IA1_l[:,Current_BSCC, :]
                            IA1_u_Reduced = IA1_u[:,Current_BSCC, :]
                            
                            IA1_l_Reduced = np.append(IA1_l_Reduced, np.zeros((IA1_l_Reduced.shape[0], IA1_l_Reduced.shape[1], 1)), 2)
                            IA1_u_Reduced = np.append(IA1_u_Reduced, np.zeros((IA1_u_Reduced.shape[0], IA1_u_Reduced.shape[1], 1)), 2)
                                  
                            for i in range(IA1_l_Reduced.shape[0]):
                                for j in range(IA1_l_Reduced.shape[1]):
                                    IA1_l_Reduced[i,j,-1] = sum(np.delete(IA1_l_Reduced[i][j][:],Current_BSCC))
                                    IA1_u_Reduced[i,j,-1] = min(sum(np.delete(IA1_u_Reduced[i][j][:],Current_BSCC)),1.0)
                    
                            IA1_l_Reduced = IA1_l_Reduced[:,:, list(np.append(Current_BSCC, IA1_l_Reduced.shape[2]-1))]
                            IA1_u_Reduced = IA1_u_Reduced[:,:, list(np.append(Current_BSCC, IA1_u_Reduced.shape[2]-1))]
                    
                                    
                            IA1_l_Reduced = np.append(IA1_l_Reduced, np.zeros((IA1_l_Reduced.shape[0], 1, IA1_l_Reduced.shape[2])), 1)
                            IA1_u_Reduced = np.append(IA1_u_Reduced, np.zeros((IA1_u_Reduced.shape[0], 1, IA1_u_Reduced.shape[2])), 1)
                    
                            #SELECTING THE BRIDGE STATES/TRANSITIONS/ACCEPTING/REACHABLE STATES AND NON ACCEPTING STATES OF THE REDUCED PRODUCT
                            Product_Reachable_States_Reduced = [[] for x in range(IA1_l_Reduced.shape[0])]
                            Product_Is_Bridge_State_Reduced = np.zeros((IA1_l_Reduced.shape[0],IA1_l_Reduced.shape[1]))
                            Product_Bridge_Transitions_Reduced = [[] for x in range(IA1_l_Reduced.shape[0])]
                            Which_Accepting_Pair_Reduced = [[] for x in range(IA1_l_Reduced.shape[1])]
                            Which_Non_Accepting_Pair_Reduced = [[] for x in range(IA1_l_Reduced.shape[1])]
                            Is_Accepting_Reduced = np.zeros(IA1_l_Reduced.shape[1])
                            Is_Non_Accepting_Reduced = np.zeros(IA1_l_Reduced.shape[1])
                            Is_In_Permanent_Comp_Reduced = np.zeros(IA1_l_Reduced.shape[1])
                            Previous_Accepting_BSCC = []
                            Indices = np.zeros(IA1_l.shape[1])
                            Indices = Indices.astype(int)
                    
                            
                            for i in range(len(Current_BSCC)):
                                Indices[Current_BSCC[i]] = i
                            
                            for i in range(IA1_l_Reduced.shape[0]):
                                IA1_l_Reduced[i][-1][-1] = 1.0
                                IA1_u_Reduced[i][-1][-1] = 1.0 #Sink State transition values
                                
                                for j in range(IA1_l_Reduced.shape[1] - 1):
                                    
                                    if i == 0:
                                        if Is_Accepting[Current_BSCC[j]] == 1:
                                            Is_Accepting_Reduced[j] = 1
                                            Which_Accepting_Pair_Reduced[j] = list(Which_Accepting_Pair[Current_BSCC[j]])
                                        if Is_Non_Accepting[Current_BSCC[j]] == 1:
                                            Is_Non_Accepting_Reduced[j] = 1                    
                                            Which_Non_Accepting_Pair_Reduced[j] = list(Which_Non_Accepting_Pair[Current_BSCC[j]])
                                    
                                            
                                    Reach = list([])
                                    Bridge = list([])
                                    Differential = list(set(Product_Reachable_States[i][Current_BSCC[j]]) - set(Current_BSCC))
                                    if len(Differential) != 0:
                                        Reach.append(IA1_l_Reduced.shape[1] - 1) #Then the sink state is reachable
                                        if set(Differential) == (set(Product_Bridge_Transitions[i][Current_BSCC[j]]) - set(Current_BSCC)):
                                            Bridge.append(IA1_l_Reduced.shape[1] - 1) #If these transitions are bridges, then there is a bridge to the sink state
                                    
                                    List_Reach = list(set(Product_Reachable_States[i][Current_BSCC[j]]).intersection(Current_BSCC))
                                    for k in range(len(List_Reach)):
                                        Reach.append(Indices[List_Reach[k]])
                    
                                    List_Bridge = list(set(Product_Bridge_Transitions[i][Current_BSCC[j]]).intersection(Current_BSCC))    
                                    for k in range(len(List_Bridge)):
                                        Bridge.append(Indices[List_Bridge[k]])
                                                        
                                    Reach.sort()
                                    Bridge.sort()
                                    Product_Reachable_States_Reduced[i].append(Reach)
                                    Product_Is_Bridge_State_Reduced[i][j] = Product_Is_Bridge_State[i][Current_BSCC[j]]
                                    Product_Bridge_Transitions_Reduced[i].append(Bridge)
                                
                    
                            first = 0;
                            Allowable_Action_Potential = list([]) #Actions that could make the state a potential BSCC
                            Allowable_Action_Permanent = list([])
                            for i in range(len(Current_BSCC)):
                                Allowable_Action_Potential.append(list(Allowable_Actions[Current_BSCC[i]])) #Actions that could make the state a potential BSCC
                                Allowable_Action_Permanent.append(list(Allowable_Actions[Current_BSCC[i]]))        
                           
                            
                            Optimal_Policy_BSCC = np.zeros(IA1_l.shape[1])
                            Optimal_Policy_BSCC = Optimal_Policy_BSCC.astype(int)           
                            Potential_Policy_BSCC = np.zeros(IA1_l.shape[1]) #Policy to generate the "best" best-case (maximize upper bound)
                            Potential_Policy_BSCC = Potential_Policy_BSCC.astype(int)
                           
                            (Greatest_Potential_Accepting_BSCCs_Reduced, Greatest_Permanent_Accepting_BSCCs_Reduced, Potential_Policy_BSCC, Optimal_Policy_BSCC, Allowable_Action_Potential, Allowable_Action_Permanent, first, Is_In_Permanent_Comp_Reduced, List_Permanent_Accepting_BSCC_Reduced, List_Potential_Accepting_BSCC_Reduced, Which_Potential_Accepting_BSCC_Reduced, Is_In_Potential_Accepting_BSCC_Reduced, Bridge_Accepting_BSCC_Reduced) = Find_Greatest_Accepting_BSCCs(IA1_l_Reduced, IA1_u_Reduced, Is_Accepting_Reduced, Is_Non_Accepting_Reduced, Which_Accepting_Pair_Reduced, Which_Non_Accepting_Pair_Reduced, Allowable_Action_Potential, Allowable_Action_Permanent, first, Product_Reachable_States_Reduced, Product_Bridge_Transitions_Reduced, Product_Is_Bridge_State_Reduced, Automata_Accepting, Potential_Policy_BSCC, Optimal_Policy_BSCC, Is_In_Permanent_Comp_Reduced, [], [range(IA1_l_Reduced.shape[1]-1)]) # Will return greatest potential accepting bsccs

                            #NEED TO ADD THE NEW BSCCs INDIVIDUALLY
                            Cur_BSCC = list([])
                                                           
                             
                            for i in range(len(List_Potential_Accepting_BSCC_Reduced)):
                                New_BSCC = list([])
                                for w in range(len(List_Potential_Accepting_BSCC_Reduced[i])):
                                    New_BSCC.append(Current_BSCC[List_Potential_Accepting_BSCC_Reduced[i][w]])
                                    Allowable_Actions[Current_BSCC[List_Potential_Accepting_BSCC_Reduced[i][w]]] = list(Allowable_Action_Potential[List_Potential_Accepting_BSCC_Reduced[i][w]])
                                List_All_BSCCs.append(New_BSCC)    
                        else:
                            Cur_BSCC = list([])
                            BSCC = 1 #Tag to break the loop, but actually not a BSCC since the list is empty
                    else:
                        break
                    
                if len(Cur_BSCC) == 0:
                    m += 1
                    if m == len(List_All_BSCCs):
                        tag = 1  
                    continue
                
                #OTHERWISE, WE CHECK TO SEE WETHER THE BSCC CAN CREATE SUB NON-ACCEPTING BSCCs WITH THE REMAINING ACTIONS
    
                ind_acc = [] #Contains the Automaton indices of accepting states in BSCC
                acc_states = [] #Contains the Accepting States in BSCC
                ind_non_acc = [] #Contains the Automaton indices of non-accepting states in BSCC
                non_acc_states = [] #Contans the non Accepting States in BSCC
                                          
                for j in range(len(Cur_BSCC)):
                                       
                    if Is_Accepting[Cur_BSCC[j]] == 1: # accepting, then add it as accepting and vice versa
                        acc_states.append(Cur_BSCC[j])
                        indices = [] # Establish index list
                        for n in range(len(Which_Accepting_Pair[Cur_BSCC[j]])): # loop through to find which accepting pair-conditions are sufficed by the given state
                            indices.append(Which_Accepting_Pair[Cur_BSCC[j]][n]) # Add the respective absolute index/label for the states
                        ind_acc.append(indices) # Add the accumulated list of indices with respect to the BSCC
    
                    if Is_Non_Accepting[Cur_BSCC[j]] == 1: # do the same thing for non-accepting states in the BSCC
                        non_acc_states.append(Cur_BSCC[j])
                        indices = []
                        for n in range(len(Which_Non_Accepting_Pair[Cur_BSCC[j]])):
                            indices.append(Which_Non_Accepting_Pair[Cur_BSCC[j]][n])
                        ind_non_acc.append(indices)                         
                          
                Acc_Tag = 0 #In theory, the set of states should always be accepting here because we checked it beforehand
                Accept = [] #Contains unmatched accepting states
                                                    
                if len(non_acc_states) == 0: # If there are no non-accepting states,
                    Acc_Tag = 1 # then activate tag for accepting BSCC.
                    for j in range(len(acc_states)): # Subsequently, add to the list of accepting BSCC.
                        Accept.append(acc_states[j])
                
                else:                                        
                  
                    Non_Accept_Remove = [[] for x in range(len(Acc))] # Contains all non-accepting states which prevent the bscc to be accepting for all pairs                        
                    for j in range(len(ind_acc)): # Recall that ind_acc contains the accumulated list of indices of the pairs with respect to the BSCC
                        for l in range(len(ind_acc[j])): # ind_acc[j] contains the indices for the relevant DRA pair, to which the state of the BSCC was complying for acceptance
                            Check_Tag = 0
                            Keep_Going = 0
                            for w in range(len(ind_non_acc)): # Same thing for non accepting
                                if ind_acc[j][l] in ind_non_acc[w]: # Checks if accepting index is in list of non_accepting indices ?? YES
                                    Check_Tag = 1 # Means that the current index in the accepting states doesn't make the BSCC accepting
                                    if len(Non_Accept_Remove[ind_acc[j][l]]) == 0: # If the list of non-accepting states that must be removed is empty, then
                                        Non_Accept_Remove[ind_acc[j][l]].append(non_acc_states[w]) # add to list of Non-Accepting states to be removed in the BSCC.
                                        Keep_Going = 1 # Stops checking the state, because we know it has to be removed
                                    elif Keep_Going == 0: # when there are no states to be removed in the BSCC
                                        break
                            if Check_Tag == 0: 
                                Accept.append(acc_states[j]) # add the list of accepting states
                                Acc_Tag = 1  
    
    
                if Acc_Tag == 1: #If the BSCC is accepting

                    
                    #Compute the policy that maximizes the lower bound probability of reaching unmatched accepting states 
                    Cur_BSCC.sort()
                    Accept.sort()
                    Permanent_Policy_BSCC = np.zeros(len(Cur_BSCC))
                    for i in range(len(Accept)):   #Converts indices of SCC for reachability computation                       
                        Act = Allowable_Actions[Accept[i]][0]
                        Accept[i] = Cur_BSCC.index(Accept[i])
                        Permanent_Policy_BSCC[i] = Act
                        
                                      
                    # Creating the list of reachable states etc.  (very computationally inefficient)
                    BSCC_Reachable_States = []
                    Indices = []
                    for i in range(IA1_l.shape[0]):
                        BSCC_Reachable_States.append([])
                        for j in range(len(Cur_BSCC)):
                            if i == 0:
                                Indices.append(Cur_BSCC[j])
                            BSCC_Reachable_States[-1].append([])                    
                    
                    IA1_l_BSCC = np.array(IA1_l[:,Indices,:])
                    IA1_l_BSCC = np.array(IA1_l_BSCC[:,:,Indices])
                    IA1_u_BSCC = np.array(IA1_u[:,Indices,:])
                    IA1_u_BSCC = np.array(IA1_u_BSCC[:,:,Indices])
                    
                    for i in range(len(Cur_BSCC)):
                        for j in range(len(Cur_BSCC)):
                            for l in range(len(Allowable_Actions[Cur_BSCC[i]])):
                                if IA1_u_BSCC[Allowable_Actions[Cur_BSCC[i]][l], i,j] > 0:
                                    BSCC_Reachable_States[Allowable_Actions[Cur_BSCC[i]][l]][i].append(j)
                    
                    
                    BSCC_Allowed_Actions = []
                    for i in range(len(Indices)):
                        BSCC_Allowed_Actions.append(list(Allowable_Actions[Indices[i]]))  
                    

                    (Dummy_Reach, Dummy_Low_Bounds, Dummy_Chain, Permanent_Policy_BSCC, Dum) = Maximize_Lower_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Accept, 0, 0, BSCC_Reachable_States, [], Permanent_Policy_BSCC, BSCC_Allowed_Actions, [], []) # Minimizes Upper Bound
                    Bad_States = []
                    for i in range(len(Dummy_Low_Bounds)):
                        if Dummy_Low_Bounds[i] == 0: #If some states have a lower-bound zero of reaching an accepting state inside the BSCC, then it means there will always be a scenario where those states form a non-accepting BSCC, and therefore cannot be part of a permanent BSCC
                            Bad_States.append(Cur_BSCC[i])
                            

    
                    if len(Bad_States) == 0:                    
                            List_Permanent_Accepting_BSCC.append([])
                            for i in range(len(Cur_BSCC)):
                                Remove_From_BSCC.append(Cur_BSCC[i])
                                Optimal_Policy[Cur_BSCC[i]] = Permanent_Policy_BSCC[i]
                                Greatest_Permanent_Winning_Component.append(Cur_BSCC[i])
                                Is_In_Permanent_Comp[Cur_BSCC[i]] = 1
                                if Is_In_Potential_Accepting_BSCC[Cur_BSCC[i]] == 1:
                                    Is_In_Potential_Accepting_BSCC[Cur_BSCC[i]] = 0
                                    if Which_Potential_Accepting_BSCC[Cur_BSCC[i]] not in Potential_To_Check:
                                        Potential_To_Check.append(Which_Potential_Accepting_BSCC[Cur_BSCC[i]])
                                        Potential_To_Delete.append(Which_Potential_Accepting_BSCC[Cur_BSCC[i]])
                                        List_Potential_Accepting_BSCC[Which_Potential_Accepting_BSCC[Cur_BSCC[i]]].remove(Cur_BSCC[i])
                    
                
                            #List_Permanent_Accepting_BSCC[-1].append(Cur_BSCC[i])
                    else:                      
                        Current_BSCC = list(set(Cur_BSCC) - set(Bad_States)) #Create new set of states without the states to be removed
                        Current_BSCC.sort()
                        
                        IA1_l_Reduced = IA1_l[:,Current_BSCC, :]
                        IA1_u_Reduced = IA1_u[:,Current_BSCC, :]
                        
                        IA1_l_Reduced = np.append(IA1_l_Reduced, np.zeros((IA1_l_Reduced.shape[0], IA1_l_Reduced.shape[1], 1)), 2)
                        IA1_u_Reduced = np.append(IA1_u_Reduced, np.zeros((IA1_u_Reduced.shape[0], IA1_u_Reduced.shape[1], 1)), 2)
                              
                        for i in range(IA1_l_Reduced.shape[0]):
                            for j in range(IA1_l_Reduced.shape[1]):
                                IA1_l_Reduced[i,j,-1] = sum(np.delete(IA1_l_Reduced[i][j][:],Current_BSCC))
                                IA1_u_Reduced[i,j,-1] = min(sum(np.delete(IA1_u_Reduced[i][j][:],Current_BSCC)),1.0)
                            
                        print Current_BSCC
                        
                        IA1_l_Reduced = IA1_l_Reduced[:,:, list(np.append(Current_BSCC, IA1_l_Reduced.shape[2]-1))]
                        IA1_u_Reduced = IA1_u_Reduced[:,:, list(np.append(Current_BSCC, IA1_u_Reduced.shape[2]-1))]
                
                                
                        IA1_l_Reduced = np.append(IA1_l_Reduced, np.zeros((IA1_l_Reduced.shape[0], 1, IA1_l_Reduced.shape[2])), 1)
                        IA1_u_Reduced = np.append(IA1_u_Reduced, np.zeros((IA1_u_Reduced.shape[0], 1, IA1_u_Reduced.shape[2])), 1)
                
                        #SELECTING THE BRIDGE STATES/TRANSITIONS/ACCEPTING/REACHABLE STATES AND NON ACCEPTING STATES OF THE REDUCED PRODUCT
                        Product_Reachable_States_Reduced = [[] for x in range(IA1_l_Reduced.shape[0])]
                        Product_Is_Bridge_State_Reduced = np.zeros((IA1_l_Reduced.shape[0],IA1_l_Reduced.shape[1]))
                        Product_Bridge_Transitions_Reduced = [[] for x in range(IA1_l_Reduced.shape[0])]
                        Which_Accepting_Pair_Reduced = [[] for x in range(IA1_l_Reduced.shape[1])]
                        Which_Non_Accepting_Pair_Reduced = [[] for x in range(IA1_l_Reduced.shape[1])]
                        Is_Accepting_Reduced = np.zeros(IA1_l_Reduced.shape[1])
                        Is_Non_Accepting_Reduced = np.zeros(IA1_l_Reduced.shape[1])
                        Is_In_Permanent_Comp_Reduced = np.zeros(IA1_l_Reduced.shape[1])
                        Previous_Accepting_BSCC = []
                        Indices = np.zeros(IA1_l.shape[1])
                        Indices = Indices.astype(int)
                
                        
                        for i in range(len(Current_BSCC)):
                            Indices[Current_BSCC[i]] = i
                        
                        for i in range(IA1_l_Reduced.shape[0]):
                            IA1_l_Reduced[i][-1][-1] = 1.0
                            IA1_u_Reduced[i][-1][-1] = 1.0 #Sink State transition values
                            
                            for j in range(IA1_l_Reduced.shape[1] - 1):
                                
                                if i == 0:
                                    if Is_Accepting[Current_BSCC[j]] == 1:
                                        Is_Accepting_Reduced[j] = 1
                                        Which_Accepting_Pair_Reduced[j] = list(Which_Accepting_Pair[Current_BSCC[j]])
                                    if Is_Non_Accepting[Current_BSCC[j]] == 1:
                                        Is_Non_Accepting_Reduced[j] = 1                    
                                        Which_Non_Accepting_Pair_Reduced[j] = list(Which_Non_Accepting_Pair[Current_BSCC[j]])
                                
                #                    Reach_Outside = 0
                                        
                                Reach = list([])
                                Bridge = list([])
                                Differential = list(set(Product_Reachable_States[i][Current_BSCC[j]]) - set(Current_BSCC))
                                if len(Differential) != 0:
                                    Reach.append(IA1_l_Reduced.shape[1] - 1) #Then the sink state is reachable
                                    if set(Differential) == (set(Product_Bridge_Transitions[i][Current_BSCC[j]]) - set(Current_BSCC)):
                                        Bridge.append(IA1_l_Reduced.shape[1] - 1) #If these transitions are bridges, then there is a bridge to the sink state
                                
                                List_Reach = list(set(Product_Reachable_States[i][Current_BSCC[j]]).intersection(Current_BSCC))
                                for k in range(len(List_Reach)):
                                    Reach.append(Indices[List_Reach[k]])
                
                                List_Bridge = list(set(Product_Bridge_Transitions[i][Current_BSCC[j]]).intersection(Current_BSCC))    
                                for k in range(len(List_Bridge)):
                                    Bridge.append(Indices[List_Bridge[k]])
                                    
                                Reach.sort()
                                Bridge.sort()
                                Product_Reachable_States_Reduced[i].append(Reach)
                                Product_Is_Bridge_State_Reduced[i][j] = Product_Is_Bridge_State[i][Current_BSCC[j]]
                                Product_Bridge_Transitions_Reduced[i].append(Bridge)
                            
                
                        first = 0;
                        Allowable_Action_Potential = list([]) #Actions that could make the state a potential BSCC
                        Allowable_Action_Permanent = list([])
                        for i in range(len(Current_BSCC)):
                            Allowable_Action_Potential.append(list(Allowable_Actions[Current_BSCC[i]])) #Actions that could make the state a potential BSCC
                            Allowable_Action_Permanent.append(list(Allowable_Actions[Current_BSCC[i]]))        
                       
                        Optimal_Policy_BSCC = np.zeros(IA1_l.shape[1])
                        Optimal_Policy_BSCC = Optimal_Policy_BSCC.astype(int)           
                        Potential_Policy_BSCC = np.zeros(IA1_l.shape[1]) #Policy to generate the "best" best-case (maximize upper bound)
                        Potential_Policy_BSCC = Potential_Policy_BSCC.astype(int)
                       
                        (Greatest_Potential_Accepting_BSCCs_Reduced, Greatest_Permanent_Accepting_BSCCs_Reduced, Potential_Policy_BSCC, Optimal_Policy_BSCC, Allowable_Action_Potential, Allowable_Action_Permanent, first, Is_In_Permanent_Comp_Reduced, List_Permanent_Accepting_BSCC_Reduced, List_Potential_Accepting_BSCC_Reduced, Which_Potential_Accepting_BSCC_Reduced, Is_In_Potential_Accepting_BSCC_Reduced, Bridge_Accepting_BSCC_Reduced) = Find_Greatest_Accepting_BSCCs(IA1_l_Reduced, IA1_u_Reduced, Is_Accepting_Reduced, Is_Non_Accepting_Reduced, Which_Accepting_Pair_Reduced, Which_Non_Accepting_Pair_Reduced, Allowable_Action_Potential, Allowable_Action_Permanent, first, Product_Reachable_States_Reduced, Product_Bridge_Transitions_Reduced, Product_Is_Bridge_State_Reduced, Automata_Accepting, Potential_Policy_BSCC, Optimal_Policy_BSCC, Is_In_Permanent_Comp_Reduced, [], [range(IA1_l_Reduced.shape[1]-1)]) # Will return greatest potential accepting bsccs
                        
                        #NEED TO ADD THE NEW BSCCs INDIVIDUALLY
                        
                        for i in range(len(List_Potential_Accepting_BSCC_Reduced)):
                            New_BSCC = list([])
                            for w in range(len(List_Potential_Accepting_BSCC_Reduced[i])):
                                New_BSCC.append(Current_BSCC[List_Potential_Accepting_BSCC_Reduced[i][w]])
                                Allowable_Actions[Current_BSCC[List_Potential_Accepting_BSCC_Reduced[i][w]]] = list(Allowable_Action_Potential[List_Potential_Accepting_BSCC_Reduced[i][w]])
                            List_All_BSCCs.append(New_BSCC) 
                 
                m += 1    
                if m == len(List_All_BSCCs):
                    tag = 1
                    
            Allowable_Actions = copy.deepcopy(Original_Allowable_Actions)
            
            if len(Remove_From_BSCC) != 0:

                Current_BSCC = list(set(List_All_BSCCs[0]) - set(Remove_From_BSCC))               
                if len(Current_BSCC) != 0:
                                    
                    Check_Accepting = 0 #If the set of states does not contain an accepting state, we can just continue
                    for k in range(len(Current_BSCC)):
                        if Is_Accepting[Current_BSCC[k]] == 1:
                            Check_Accepting = 1
                            break
                            
                    
                    if Check_Accepting == 1:
                        
                        for k in range(len(Current_BSCC)):
                            Is_In_Potential_Accepting_BSCC[Current_BSCC[k]] = 0
                        
                
                        IA1_l_Reduced = IA1_l[:,Current_BSCC, :]
                        IA1_u_Reduced = IA1_u[:,Current_BSCC, :]
                        
                        
                        #CREATE SINK STATES TO REPRESENT TRANSITIONS OUTSIDE OF THESE PREVIOUS COMPONENTS
                        
                        IA1_l_Reduced = np.append(IA1_l_Reduced, np.zeros((IA1_l_Reduced.shape[0], IA1_l_Reduced.shape[1], 1)), 2)
                        IA1_u_Reduced = np.append(IA1_u_Reduced, np.zeros((IA1_u_Reduced.shape[0], IA1_u_Reduced.shape[1], 1)), 2)
                              
                        for i in range(IA1_l_Reduced.shape[0]):
                            for j in range(IA1_l_Reduced.shape[1]):
                                IA1_l_Reduced[i,j,-1] = sum(np.delete(IA1_l_Reduced[i][j][:],Current_BSCC))
                                IA1_u_Reduced[i,j,-1] = min(sum(np.delete(IA1_u_Reduced[i][j][:],Current_BSCC)),1.0)
                
                        IA1_l_Reduced = IA1_l_Reduced[:,:, list(np.append(Current_BSCC, IA1_l_Reduced.shape[2]-1))]
                        IA1_u_Reduced = IA1_u_Reduced[:,:, list(np.append(Current_BSCC, IA1_u_Reduced.shape[2]-1))]
                
                                
                        IA1_l_Reduced = np.append(IA1_l_Reduced, np.zeros((IA1_l_Reduced.shape[0], 1, IA1_l_Reduced.shape[2])), 1)
                        IA1_u_Reduced = np.append(IA1_u_Reduced, np.zeros((IA1_u_Reduced.shape[0], 1, IA1_u_Reduced.shape[2])), 1)
                
                        #SELECTING THE BRIDGE STATES/TRANSITIONS/ACCEPTING/REACHABLE STATES AND NON ACCEPTING STATES OF THE REDUCED PRODUCT
                        Product_Reachable_States_Reduced = [[] for x in range(IA1_l_Reduced.shape[0])]
                        Product_Is_Bridge_State_Reduced = np.zeros((IA1_l_Reduced.shape[0],IA1_l_Reduced.shape[1]))
                        Product_Bridge_Transitions_Reduced = [[] for x in range(IA1_l_Reduced.shape[0])]
                        Which_Accepting_Pair_Reduced = [[] for x in range(IA1_l_Reduced.shape[1])]
                        Which_Non_Accepting_Pair_Reduced = [[] for x in range(IA1_l_Reduced.shape[1])]
                        Is_Accepting_Reduced = np.zeros(IA1_l_Reduced.shape[1])
                        Is_Non_Accepting_Reduced = np.zeros(IA1_l_Reduced.shape[1])
                        Is_In_Permanent_Comp_Reduced = np.zeros(IA1_l_Reduced.shape[1])
                        Previous_Accepting_BSCC = []
                        Indices = np.zeros(IA1_l.shape[1])
                        Indices = Indices.astype(int)
                
                        
                        for i in range(len(Current_BSCC)):
                            Indices[Current_BSCC[i]] = i
                        
                        for i in range(IA1_l_Reduced.shape[0]):
                            IA1_l_Reduced[i][-1][-1] = 1.0
                            IA1_u_Reduced[i][-1][-1] = 1.0 #Sink State transition values
                            
                            for j in range(IA1_l_Reduced.shape[1] - 1):
                                
                                if i == 0:
                                    if Is_Accepting[Current_BSCC[j]] == 1:
                                        Is_Accepting_Reduced[j] = 1
                                        Which_Accepting_Pair_Reduced[j] = list(Which_Accepting_Pair[Current_BSCC[j]])
                                    if Is_Non_Accepting[Current_BSCC[j]] == 1:
                                        Is_Non_Accepting_Reduced[j] = 1                    
                                        Which_Non_Accepting_Pair_Reduced[j] = list(Which_Non_Accepting_Pair[Current_BSCC[j]])
                                
                #                    Reach_Outside = 0
                                        
                                Reach = list([])
                                Bridge = list([])
                                Differential = list(set(Product_Reachable_States[i][Current_BSCC[j]]) - set(Current_BSCC))
                                if len(Differential) != 0:
                                    Reach.append(IA1_l_Reduced.shape[1] - 1) #Then the sink state is reachable
                                    if set(Differential) == (set(Product_Bridge_Transitions[i][Current_BSCC[j]]) - set(Current_BSCC)):
                                        Bridge.append(IA1_l_Reduced.shape[1] - 1) #If these transitions are bridges, then there is a bridge to the sink state
                                
                                List_Reach = list(set(Product_Reachable_States[i][Current_BSCC[j]]).intersection(Current_BSCC))
                                for k in range(len(List_Reach)):
                                    Reach.append(Indices[List_Reach[k]])
                
                                List_Bridge = list(set(Product_Bridge_Transitions[i][Current_BSCC[j]]).intersection(Current_BSCC))    
                                for k in range(len(List_Bridge)):
                                    Bridge.append(Indices[List_Bridge[k]])
                    
                                Reach.sort()
                                Bridge.sort()
                                Product_Reachable_States_Reduced[i].append(Reach)
                                Product_Is_Bridge_State_Reduced[i][j] = Product_Is_Bridge_State[i][Current_BSCC[j]]
                                Product_Bridge_Transitions_Reduced[i].append(Bridge)
                            
                
                        first = 1;
                        Allowable_Action_Potential = list([]) #Actions that could make the state a potential BSCC
                        Allowable_Action_Permanent = list([]) #Actions that could make the state a permanent BSCC        
                       
                        Optimal_Policy_BSCC = np.zeros(IA1_l.shape[1])
                        Optimal_Policy_BSCC = Optimal_Policy_BSCC.astype(int)           
                        Potential_Policy_BSCC = np.zeros(IA1_l.shape[1]) #Policy to generate the "best" best-case (maximize upper bound)
                        Potential_Policy_BSCC = Potential_Policy_BSCC.astype(int)
                       
                        (Greatest_Potential_Accepting_BSCCs_Reduced, Greatest_Permanent_Accepting_BSCCs_Reduced, Potential_Policy_BSCC, Optimal_Policy_BSCC, Allowable_Action_Potential, Allowable_Action_Permanent, first, Is_In_Permanent_Comp_Reduced, List_Permanent_Accepting_BSCC_Reduced, List_Potential_Accepting_BSCC_Reduced, Which_Potential_Accepting_BSCC_Reduced, Is_In_Potential_Accepting_BSCC_Reduced, Bridge_Accepting_BSCC_Reduced) = Find_Greatest_Accepting_BSCCs(IA1_l_Reduced, IA1_u_Reduced, Is_Accepting_Reduced, Is_Non_Accepting_Reduced, Which_Accepting_Pair_Reduced, Which_Non_Accepting_Pair_Reduced, Allowable_Action_Potential, Allowable_Action_Permanent, first, Product_Reachable_States_Reduced, Product_Bridge_Transitions_Reduced, Product_Is_Bridge_State_Reduced, Automata_Accepting, Potential_Policy_BSCC, Optimal_Policy_BSCC, Is_In_Permanent_Comp_Reduced, [], [range(IA1_l_Reduced.shape[1]-1)]) # Will return greatest potential accepting bsccs
                

                        
                        Length_List = len(List_Potential_Accepting_BSCC)
                        for i in range(len(List_Potential_Accepting_BSCC_Reduced)):
                           List_Potential_Accepting_BSCC.append(list([]))
                           Bridge_Accepting_BSCC.append(list([]))
                           for x in range(len(List_Potential_Accepting_BSCC_Reduced[i])):
                               Is_In_Potential_Accepting_BSCC[Current_BSCC[List_Potential_Accepting_BSCC_Reduced[i][x]]] = 1
                               List_Potential_Accepting_BSCC[-1].append(Current_BSCC[List_Potential_Accepting_BSCC_Reduced[i][x]])
                               Which_Potential_Accepting_BSCC[Current_BSCC[List_Potential_Accepting_BSCC_Reduced[i][x]]] = Length_List+i
                
                           for x in range(len(Bridge_Accepting_BSCC_Reduced[i])):
                               Bridge_Accepting_BSCC[-1].append(Current_BSCC[Bridge_Accepting_BSCC_Reduced[i][x]])

        
                
        Allowable_Actions_Quanti = copy.deepcopy(Original_Allowable_Actions)
           
        (Low_Bound, Low_Bounds_Prod, Worst_Markov_Chain, Optimal_Policy, List_Values_Low) = Maximize_Lower_Bound_Reachability(IA1_l, IA1_u, Greatest_Permanent_Winning_Component, IA1_l.shape[1]/Len_Automata, Len_Automata, Product_Reachable_States, Init, Optimal_Policy, Allowable_Actions_Quanti, [], []) # Maximizes Lower Bound
        
        States_To_Check = list(set(range(len(Low_Bounds_Prod))) - set(Greatest_Permanent_Winning_Component))
        for i in range(len(States_To_Check)):
            if Low_Bounds_Prod[States_To_Check[i]] == 1:
                Is_In_Permanent_Comp[States_To_Check[i]] = 1
                Greatest_Permanent_Winning_Component.append(States_To_Check[i])
                if Is_In_Potential_Accepting_BSCC[States_To_Check[i]] == 1:
                    Is_In_Potential_Accepting_BSCC[States_To_Check[i]] = 0
                    if Which_Potential_Accepting_BSCC[States_To_Check[i]] not in Potential_To_Check:
                        Potential_To_Check.append(Which_Potential_Accepting_BSCC[States_To_Check[i]])
                        Potential_To_Delete.append(Which_Potential_Accepting_BSCC[States_To_Check[i]])
                    List_Potential_Accepting_BSCC[Which_Potential_Accepting_BSCC[States_To_Check[i]]].remove(States_To_Check[i])
             

        
        for n in range(len(Potential_To_Check)):
            Current_BSCC = list(List_Potential_Accepting_BSCC[Potential_To_Check[n]])
            if len(Current_BSCC) == 0:
                continue
            
            Check_Accepting = 0 #If the set of states does not contain an accepting state, we can just continue
            for k in range(len(Current_BSCC)):
                if Is_Accepting[Current_BSCC[k]] == 1:
                    Check_Accepting = 1
                    break
                    
            
            if Check_Accepting == 0:
                continue
            
            for k in range(len(Current_BSCC)):
                Is_In_Potential_Accepting_BSCC[Current_BSCC[k]] = 0
                
            IA1_l_Reduced = IA1_l[:,Current_BSCC, :]
            IA1_u_Reduced = IA1_u[:,Current_BSCC, :]
            
            
            #CREATE SINK STATES TO REPRESENT TRANSITIONS OUTSIDE OF THESE PREVIOUS COMPONENTS
            
            IA1_l_Reduced = np.append(IA1_l_Reduced, np.zeros((IA1_l_Reduced.shape[0], IA1_l_Reduced.shape[1], 1)), 2)
            IA1_u_Reduced = np.append(IA1_u_Reduced, np.zeros((IA1_u_Reduced.shape[0], IA1_u_Reduced.shape[1], 1)), 2)
                  
            for i in range(IA1_l_Reduced.shape[0]):
                for j in range(IA1_l_Reduced.shape[1]):
                    IA1_l_Reduced[i,j,-1] = sum(np.delete(IA1_l_Reduced[i][j][:],Current_BSCC))
                    IA1_u_Reduced[i,j,-1] = min(sum(np.delete(IA1_u_Reduced[i][j][:],Current_BSCC)),1.0)
    
            IA1_l_Reduced = IA1_l_Reduced[:,:, list(np.append(Current_BSCC, IA1_l_Reduced.shape[2]-1))]
            IA1_u_Reduced = IA1_u_Reduced[:,:, list(np.append(Current_BSCC, IA1_u_Reduced.shape[2]-1))]
    
                    
            IA1_l_Reduced = np.append(IA1_l_Reduced, np.zeros((IA1_l_Reduced.shape[0], 1, IA1_l_Reduced.shape[2])), 1)
            IA1_u_Reduced = np.append(IA1_u_Reduced, np.zeros((IA1_u_Reduced.shape[0], 1, IA1_u_Reduced.shape[2])), 1)
    
            #SELECTING THE BRIDGE STATES/TRANSITIONS/ACCEPTING/REACHABLE STATES AND NON ACCEPTING STATES OF THE REDUCED PRODUCT
            Product_Reachable_States_Reduced = [[] for x in range(IA1_l_Reduced.shape[0])]
            Product_Is_Bridge_State_Reduced = np.zeros((IA1_l_Reduced.shape[0],IA1_l_Reduced.shape[1]))
            Product_Bridge_Transitions_Reduced = [[] for x in range(IA1_l_Reduced.shape[0])]
            Which_Accepting_Pair_Reduced = [[] for x in range(IA1_l_Reduced.shape[1])]
            Which_Non_Accepting_Pair_Reduced = [[] for x in range(IA1_l_Reduced.shape[1])]
            Is_Accepting_Reduced = np.zeros(IA1_l_Reduced.shape[1])
            Is_Non_Accepting_Reduced = np.zeros(IA1_l_Reduced.shape[1])
            Is_In_Permanent_Comp_Reduced = np.zeros(IA1_l_Reduced.shape[1])
            Previous_Accepting_BSCC = []
            Indices = np.zeros(IA1_l.shape[1])
            Indices = Indices.astype(int)
    
            
            for i in range(len(Current_BSCC)):
                Indices[Current_BSCC[i]] = i
            
            for i in range(IA1_l_Reduced.shape[0]):
                IA1_l_Reduced[i][-1][-1] = 1.0
                IA1_u_Reduced[i][-1][-1] = 1.0 #Sink State transition values
                
                for j in range(IA1_l_Reduced.shape[1] - 1):
                    
                    if i == 0:
                        if Is_Accepting[Current_BSCC[j]] == 1:
                            Is_Accepting_Reduced[j] = 1
                            Which_Accepting_Pair_Reduced[j] = list(Which_Accepting_Pair[Current_BSCC[j]])
                        if Is_Non_Accepting[Current_BSCC[j]] == 1:
                            Is_Non_Accepting_Reduced[j] = 1                    
                            Which_Non_Accepting_Pair_Reduced[j] = list(Which_Non_Accepting_Pair[Current_BSCC[j]])
                    
    #                    Reach_Outside = 0
                            
                    Reach = list([])
                    Bridge = list([])
                    Differential = list(set(Product_Reachable_States[i][Current_BSCC[j]]) - set(Current_BSCC))
                    if len(Differential) != 0:
                        Reach.append(IA1_l_Reduced.shape[1] - 1) #Then the sink state is reachable
                        if set(Differential) == (set(Product_Bridge_Transitions[i][Current_BSCC[j]]) - set(Current_BSCC)):
                            Bridge.append(IA1_l_Reduced.shape[1] - 1) #If these transitions are bridges, then there is a bridge to the sink state
                    
                    List_Reach = list(set(Product_Reachable_States[i][Current_BSCC[j]]).intersection(Current_BSCC))
                    for k in range(len(List_Reach)):
                        Reach.append(Indices[List_Reach[k]])
    
                    List_Bridge = list(set(Product_Bridge_Transitions[i][Current_BSCC[j]]).intersection(Current_BSCC))    
                    for k in range(len(List_Bridge)):
                        Bridge.append(Indices[List_Bridge[k]])
                    
                    Reach.sort()
                    Bridge.sort()
                    Product_Reachable_States_Reduced[i].append(Reach)
                    Product_Is_Bridge_State_Reduced[i][j] = Product_Is_Bridge_State[i][Current_BSCC[j]]
                    Product_Bridge_Transitions_Reduced[i].append(Bridge)
                
    
            first = 1;
            Allowable_Action_Potential = list([]) #Actions that could make the state a potential BSCC
            Allowable_Action_Permanent = list([]) #Actions that could make the state a permanent BSCC        
           
            Optimal_Policy_BSCC = np.zeros(IA1_l.shape[1])
            Optimal_Policy_BSCC = Optimal_Policy_BSCC.astype(int)           
            Potential_Policy_BSCC = np.zeros(IA1_l.shape[1]) #Policy to generate the "best" best-case (maximize upper bound)
            Potential_Policy_BSCC = Potential_Policy_BSCC.astype(int)
           
            (Greatest_Potential_Accepting_BSCCs_Reduced, Greatest_Permanent_Accepting_BSCCs_Reduced, Potential_Policy_BSCC, Optimal_Policy_BSCC, Allowable_Action_Potential, Allowable_Action_Permanent, first, Is_In_Permanent_Comp_Reduced, List_Permanent_Accepting_BSCC_Reduced, List_Potential_Accepting_BSCC_Reduced, Which_Potential_Accepting_BSCC_Reduced, Is_In_Potential_Accepting_BSCC_Reduced, Bridge_Accepting_BSCC_Reduced) = Find_Greatest_Accepting_BSCCs(IA1_l_Reduced, IA1_u_Reduced, Is_Accepting_Reduced, Is_Non_Accepting_Reduced, Which_Accepting_Pair_Reduced, Which_Non_Accepting_Pair_Reduced, Allowable_Action_Potential, Allowable_Action_Permanent, first, Product_Reachable_States_Reduced, Product_Bridge_Transitions_Reduced, Product_Is_Bridge_State_Reduced, Automata_Accepting, Potential_Policy_BSCC, Optimal_Policy_BSCC, Is_In_Permanent_Comp_Reduced, [], range(IA1_l_Reduced.shape[1] - 1)) # Will return greatest potential accepting bsccs
    
            Length_List = len(List_Potential_Accepting_BSCC)
            for i in range(len(List_Potential_Accepting_BSCC_Reduced)):
               List_Potential_Accepting_BSCC.append(list([]))
               Bridge_Accepting_BSCC.append(list([]))
               for x in range(len(List_Potential_Accepting_BSCC_Reduced[i])):
                   Is_In_Potential_Accepting_BSCC[Current_BSCC[List_Potential_Accepting_BSCC_Reduced[i][x]]] = 1
                   List_Potential_Accepting_BSCC[-1].append(Current_BSCC[List_Potential_Accepting_BSCC_Reduced[i][x]])
                   Which_Potential_Accepting_BSCC[Current_BSCC[List_Potential_Accepting_BSCC_Reduced[i][x]]] = Length_List+i
    
               for x in range(len(Bridge_Accepting_BSCC_Reduced[i])):
                   Bridge_Accepting_BSCC[-1].append(Current_BSCC[Bridge_Accepting_BSCC_Reduced[i][x]])
           
                                           
                    
        if len(list(set(Greatest_Permanent_Winning_Component) - set(Previous_Permanent))) == 0:
            Check_Loop = 1
        
        Previous_Permanent = list(Greatest_Permanent_Winning_Component)

    Greatest_Permanent_Winning_Component.sort()
    for ele in sorted(Potential_To_Delete, reverse = True):  
            del List_Potential_Accepting_BSCC[ele] 
    
    
    return Greatest_Permanent_Winning_Component, Optimal_Policy, Is_In_Permanent_Comp, List_Potential_Accepting_BSCC, Is_In_Potential_Accepting_BSCC, Bridge_Accepting_BSCC


def Maximize_Lower_Bound_Reachability(IA_l, IA_u, Q1, Num_States, Automata_size, Reach, Init, Optimal_Policy, Actions, Product_Pre_States, Q0):
    
    
    Ascending_Order = []
    
    if len(Product_Pre_States) != 0:
         Index_Vector = np.zeros((IA_l['Num_S'],1))
         Is_In_Q1 = np.zeros((IA_l['Num_S']))
         Is_In_Q0 = np.zeros((IA_l['Num_S'])) 
         Iter_Int = IA_l['Num_S']
    else:
        Index_Vector = np.zeros((IA_l.shape[1],1))
        Is_In_Q1 = np.zeros((IA_l.shape[1]))
        Is_In_Q0 = np.zeros((IA_l.shape[1]))
        Iter_Int = IA_l.shape[1]
    States_To_Consider = set([])
    
    if len(Product_Pre_States) != 0:
        Bounds_All_Act = [[] for x in range(IA_l['Num_S'])]

    Set_Q1 = set(Q1)
    Set_Q0 = set(Q0)
    

    
    for k in range(Iter_Int):                               
        if k in Set_Q1:            
            Index_Vector[k,0] = 1.0
            Ascending_Order.append(k)
            Is_In_Q1[k] = 1           
            
            if len(Product_Pre_States) != 0: #We optimize this function for large scale computations             
                Bounds_All_Act[k].append(1.0)
                for n in Product_Pre_States: 
                  States_To_Consider = States_To_Consider.union(n[k])

        elif k not in Set_Q0:            
            Ascending_Order.insert(0,k)
            if len(Product_Pre_States) != 0:
                for n in range(len(Actions[k])):
                   Bounds_All_Act[k].append(0.0)
            
    for k in Q0:
        Is_In_Q0[k] = 1
        Bounds_All_Act[k].append(0.0)
        Ascending_Order.insert(0,k)            

    d = {k:v for v,k in enumerate(Ascending_Order)} 
    Sort_Reach = []

    
    
    if len(Product_Pre_States) != 0: #If want to optimize
        
        States_To_Consider = list(States_To_Consider)
        New_States_To_Consider = []
        for i in States_To_Consider:
            if Is_In_Q1[i] == 1 or Is_In_Q0[i] == 1:
                continue
            New_States_To_Consider.append(i)
            Sort_Reach.append([])
            for n in Actions[i]:
                Reach[n][i].sort(key=d.get)
                Sort_Reach[-1].append(list(Reach[n][i]))
        States_To_Consider = list(New_States_To_Consider) 
           
    else:        
        for i in range(len(Reach)):
            Sort_Reach.append([])
            for j in range(IA_l.shape[1]):
                Sort_Reach[-1].append([])   
                
        for j in range(IA_l.shape[1]):        
            if Is_In_Q1[j] == 0:
                for k in range(len(Actions[j])):
                    Reach[Actions[j][k]][j].sort(key=d.get)
                    Sort_Reach[Actions[j][k]][j] = list(Reach[Actions[j][k]][j])
            else:
                continue
      
    Time_Prob = 0.0    
    start13 = timeit.default_timer()
    if len(Product_Pre_States) != 0:
        Phi_Min = {}
        Phi_Min, List_Values  = Phi_Synthesis_Max_Lower_Parallel(IA_l, IA_u, Ascending_Order, Q1, Reach, Sort_Reach, Actions, States_To_Consider,Phi_Min, Index_Vector)
    else:   
        Phi_Min = Phi_Synthesis_Max_Lower(IA_l, IA_u, Ascending_Order, Q1, Reach, Sort_Reach, Actions)

    Time_Prob += timeit.default_timer() - start13

    Time_Max = 0.0

        
    if len(Product_Pre_States) != 0:
        

        New_States_To_Consider = set([])
        start13 = timeit.default_timer()

        for i, State_Con in enumerate(States_To_Consider):
            
            Values = list(List_Values[i])
            Bounds_All_Act[State_Con] = list(Values)
            Index_Max = np.argmax(Values)
            Index_Vector[State_Con,0] = Values[Index_Max]
            Optimal_Policy[State_Con] = Actions[State_Con][Index_Max]
   
            if Index_Vector[State_Con,0] > 0:

                for n in Product_Pre_States:
                    for k in n[State_Con]:                    
                        if Is_In_Q0[k] != 1 and Is_In_Q1[k] != 1:
                            New_States_To_Consider.add(k)
                
                
        States_To_Consider = list(New_States_To_Consider)
         
        Time_Max += timeit.default_timer() - start13

    else:
        
        Steps_Low = sparse.csr_matrix.dot(sparse.csr_matrix(Phi_Min), Index_Vector)
        
        for i in range(IA_l.shape[1]):
            if Is_In_Q1[i] == 1: continue
            List_Values = []
            for k in range(len(Actions[i])):
                List_Values.append(IA_l.shape[0]*i+Actions[i][k])

            Values = Steps_Low[List_Values]
            Index_Vector[i,0] = np.amax(Values)
            Optimal_Policy[i] = Actions[i][np.argmax(Values)]
            

    

    for i in range(len(Q1)):    
        Index_Vector[Q1[i],0] = 1.0

    Success_Intervals = []  
    
    if len(Product_Pre_States) != 0:     
        for i in range(IA_l['Num_S']):       
            Success_Intervals.append(Index_Vector[i,0]) 
    else:
        for i in range(IA_l.shape[1]):       
            Success_Intervals.append(Index_Vector[i,0])         
        
     
    Terminate_Check = 0
    Convergence_threshold = 0.001

## NEED TO OPTIMIZE BELOW HERE           
    
    while Terminate_Check == 0:
        
       
        Ascending_Order = np.array(range(len(Success_Intervals)))
        Success_Array = np.array(Success_Intervals)
        Ascending_Order = list(Ascending_Order[(Success_Array).argsort()]) 
        
        d = {k:v for v,k in enumerate(Ascending_Order)} 
        Sort_Reach = []
        
        if len(Product_Pre_States) != 0:
            for i in States_To_Consider:
                Sort_Reach.append([])        
                for n in Actions[i]:
                    Reach[n][i].sort(key=d.get)
                    Sort_Reach[-1].append(list(Reach[n][i]))
        else: 
                        
            for i in range(len(Reach)):
                Sort_Reach.append([])
                for j in range(IA_l.shape[1]):
                    Sort_Reach[-1].append([])
            
            for j in range(IA_l.shape[1]):        
                if Is_In_Q1[j] == 0:
                    for k in range(len(Actions[j])):
                        Reach[Actions[j][k]][j].sort(key=d.get)
                        Sort_Reach[Actions[j][k]][j] = list(Reach[Actions[j][k]][j])
                else:
                    continue
   
        start13 = timeit.default_timer()
        if len(Product_Pre_States)!= 0:
            Phi_Min, List_Values = Phi_Synthesis_Max_Lower_Parallel(IA_l, IA_u, Ascending_Order, Q1, Reach, Sort_Reach, Actions, States_To_Consider, Phi_Min, Index_Vector) 
        else:            
            Phi_Min = Phi_Synthesis_Max_Lower(IA_l, IA_u, Ascending_Order, Q1, Reach, Sort_Reach, Actions)

        Time_Prob += timeit.default_timer() - start13
        
        if len(Product_Pre_States) != 0:
            

            start1 = timeit.default_timer()
            New_States_To_Consider = set([])
            Max_Difference = 0
            for i, State_Con in enumerate(States_To_Consider):
                Values = list(List_Values[i])
                Bounds_All_Act[State_Con] = list(Values)
                Index_Max = np.argmax(Values)
                Index_Vector[State_Con,0] = Values[Index_Max]
                Optimal_Policy[State_Con] = Actions[State_Con][Index_Max]
                Abs_Diff = abs(Index_Vector[State_Con,0] - Success_Intervals[State_Con])
                if Abs_Diff > 0:
                    Max_Difference = max(Max_Difference, Abs_Diff)

                    for n in Product_Pre_States:
                        for k in n[State_Con]:                    
                            if Is_In_Q0[k] != 1 and Is_In_Q1[k] != 1:
                                New_States_To_Consider.add(k)


                    Success_Intervals[State_Con] = Index_Vector[State_Con,0]                    
            States_To_Consider = list(New_States_To_Consider)
            Time_Max += timeit.default_timer() - start1
        
        else:
    
            Steps_Low = sparse.csr_matrix.dot(sparse.csr_matrix(Phi_Min), Index_Vector)
            
            List_Values = list([])
            Bounds_All_Act = list([])
            for i in range(IA_l.shape[1]):
                if Is_In_Q1[i] == 1: 
                    Bounds_All_Act.append(list([]))
                    for j in range(len(Actions[i])):
                        Bounds_All_Act[-1].append(1.0)
                    continue
                List_Values.append([])
                for k in range(len(Actions[i])):
                    List_Values[-1].append(IA_l.shape[0]*i+Actions[i][k])
                Values = list(Steps_Low[List_Values[-1]])
                Bounds_All_Act.append(Values)
                Index_Vector[i,0] = np.amax(Values)
                Optimal_Policy[i] = Actions[i][np.argmax(Values)]    

            for i in range(len(Q1)):    
                Index_Vector[Q1[i],0] = 1.0
            
                             
            Max_Difference = 0
                          
            for i in range(IA_l.shape[1]):
                                      
                Max_Difference = max(Max_Difference, abs(Success_Intervals[i] - Index_Vector[i,0]))        
                Success_Intervals[i] = Index_Vector[i,0]
                
            
        if Max_Difference < Convergence_threshold:              
            Terminate_Check = 1    
    
    Bounds = []
    

    
    
    if len(Product_Pre_States) != 0:

        rows, cols, vals = [], [], []
        for key in Phi_Min:
                for i in range(len(Phi_Min[key][Optimal_Policy[key]][0])):
                    rows.append(key)
                    cols.append(Phi_Min[key][Optimal_Policy[key]][0][i])
                    vals.append(Phi_Min[key][Optimal_Policy[key]][1][i])       
        Phi_Min = sparse.csr_matrix((vals, (rows, cols)), shape=(len(Optimal_Policy), len(Optimal_Policy)))    
        
    else:    
        Indices = [int(i*IA_l.shape[0]+Optimal_Policy[i]) for i in range(len(Optimal_Policy))]
        Phi_Min = np.array(Phi_Min[Indices,:])
    
    for i in range(Num_States):
        Bounds.append(Success_Intervals[i*Automata_size+Init[i]])
    
        
    return (Bounds, Success_Intervals, Phi_Min, Optimal_Policy, Bounds_All_Act)





def Maximize_Lower_Bound_Reachability_Continuous(Q1, State_Space, Num_States, Automata_size, Reach, Init, Optimal_Policy, Actions, Reachable_Sets, Pre_Quantitative_Product, Q0):
    
    
    
    Ascending_Order = []
    Index_Vector = np.zeros((len(State_Space)*Automata_size,1))
    Is_In_Q1 = np.zeros((len(State_Space)*Automata_size))
    Is_In_Q0 = np.zeros((len(State_Space)*Automata_size))
    States_To_Consider = set([])
    
    Set_Q1 = set(Q1)
    Set_Q0 = set(Q0)
    

    for k in range(len(State_Space)*Automata_size):                               
        if k in Set_Q1:            
            Index_Vector[k,0] = 1.0
            Ascending_Order.append(k)
            Is_In_Q1[k] = 1
            States_To_Consider = States_To_Consider.union(Pre_Quantitative_Product[k])
                          
        elif k not in Set_Q0:            
            Ascending_Order.insert(0,k)
            
    for k in Q0:
        Is_In_Q0[k] = 1
        Ascending_Order.insert(0,k)

    d = {k:v for v,k in enumerate(Ascending_Order)} 
    Sort_Reach = []
    
    States_To_Consider = list(States_To_Consider)

    New_States_To_Consider = []
    for i in States_To_Consider:
        if Is_In_Q1[i] == 1 or Is_In_Q0[i] == 1:
            continue
        New_States_To_Consider.append(i)
        Sort_Reach.append([])
        Reach[i].sort(key=d.get)
        Sort_Reach[-1] = list(Reach[i])
    
    States_To_Consider = list(New_States_To_Consider)
    Phi_Min = {}        

    Phi_Min, Index_Vector, Optimal_Policy = Phi_Synthesis_Max_Continuous_Parallel(State_Space, Ascending_Order, Q1, Reach, Sort_Reach, Actions, Reachable_Sets, Index_Vector, Optimal_Policy, Automata_size, Q0, Phi_Min, States_To_Consider)


    Success_Intervals = []
    States_To_Consider = set([])
  
    for i in range(len(State_Space)*Automata_size):
        if Index_Vector[i,0] > 0 and Is_In_Q1[i] != 1:

             for k in Pre_Quantitative_Product[i]:
                if Is_In_Q0[k] != 1 and Is_In_Q1[k] != 1:
                    States_To_Consider.add(k)
 
            
            
        Success_Intervals.append(float(Index_Vector[i,0]))

    States_To_Consider = list(States_To_Consider)
    Terminate_Check = 0
    Convergence_threshold = 0.01
    Previous_Max_Difference = 1
    count = 0
           
    
    while Terminate_Check == 0:
        

                                  
        for i in Q1:
            Success_Intervals[i] = 1.0
       
        Ascending_Order = np.array(range(len(Success_Intervals)))
        Success_Array = np.array(Success_Intervals)
        Ascending_Order = list(Ascending_Order[(Success_Array).argsort()]) 
        
        d = {k:v for v,k in enumerate(Ascending_Order)} 
        Sort_Reach = []

        
        for i in States_To_Consider:
            Sort_Reach.append([])        
            Reach[i].sort(key=d.get)
            Sort_Reach[-1] = list(Reach[i])
        
       
        New_States_To_Consider = set([])
        Max_Difference = 0


        Phi_Min, Index_Vector, Optimal_Policy = Phi_Synthesis_Max_Continuous_Parallel(State_Space, Ascending_Order, Q1, Reach, Sort_Reach, Actions, Reachable_Sets, Index_Vector, Optimal_Policy, Automata_size, Q0, Phi_Min, States_To_Consider)
                                                        
                      
            
        for i in States_To_Consider:
            if abs(Success_Intervals[i] - Index_Vector[i,0]) > 0:
                                         

                for j in Pre_Quantitative_Product[i]:
                    if Is_In_Q1[j] == 0 and Is_In_Q0[j] == 0:                     
                        New_States_To_Consider.add(j)  

            Max_Difference = max(Max_Difference, abs(Success_Intervals[i] - Index_Vector[i,0]))        
            Success_Intervals[i] = float(Index_Vector[i,0])
        
           
            
        count += 1
        
        
        
        
        States_To_Consider = list(New_States_To_Consider)
            
        if Max_Difference < Convergence_threshold or count > Count_Cut_Off: #Numerical error can keep this process in an infinite loop             
            Terminate_Check = 1    
    
    Bounds = []
    Prod_Bounds = []
    
    rows, cols, vals = [], [], []
    for key in Phi_Min:
            for i in range(len(Phi_Min[key][0])):
                rows.append(key)
                cols.append(Phi_Min[key][0][i])
                vals.append(Phi_Min[key][1][i])       
    Phi_Min = sparse.csr_matrix((vals, (rows, cols)), shape=(len(Optimal_Policy), len(Optimal_Policy)))    


    
    for i in range(Num_States):
        Bounds.append(Success_Intervals[i*Automata_size+Init[i]])
    
    for i in range(len(Success_Intervals)):
        if Success_Intervals[i] > 1.0: #To account for numerical errors which can slightly push probability above one
            Success_Intervals[i] = 1.0
        Prod_Bounds.append(Success_Intervals[i])
        
    return (Bounds, Prod_Bounds, Phi_Min, Optimal_Policy)
















def Maximize_Upper_Bound_Reachability(IA_l, IA_u, Q1, Num_States,Automata_size, Reach, Init, Optimal_Policy, Actions, Product_Pre_States, Q0):
    
    
    Descending_Order = []
    if len(Product_Pre_States)!= 0:
        Index_Vector = np.zeros((IA_l['Num_S'],1)) 
        Is_In_Q1 = np.zeros((IA_l['Num_S']))
        Is_In_Q0 = np.zeros((IA_l['Num_S']))
        Int_Iter = IA_l['Num_S']
        
    else:   
        Index_Vector = np.zeros((IA_l.shape[1],1)) 
        Is_In_Q1 = np.zeros((IA_l.shape[1]))
        Is_In_Q0 = np.zeros((IA_l.shape[1]))
        Int_Iter = IA_l.shape[1]
    States_To_Consider = set([])
    
    Set_Q1 = set(Q1)
    
    
    if len(Product_Pre_States) != 0:
        Bounds_All_Act = [[] for x in range(IA_l['Num_S'])]
    
    for k in range(Int_Iter):
        if k in Set_Q1:            
            Index_Vector[k,0] = 1.0
            Descending_Order.insert(0,k)
            Is_In_Q1[k] = 1

            if len(Product_Pre_States) != 0: #We optimize this function for large scale computations             
                for n in Product_Pre_States: 
                  Bounds_All_Act[k].append(1.0)
                  States_To_Consider = States_To_Consider.union(n[k])
            
           
        else:            
            Descending_Order.append(k)
            if len(Product_Pre_States) != 0:
                for n in range(len(Actions[k])):
                   Bounds_All_Act[k].append(0.0) 
                   
    for k in Q0:
        Is_In_Q0[k] = 1
        Bounds_All_Act[k].append(0.0)
        Descending_Order.append(k)                   

    d = {k:v for v,k in enumerate(Descending_Order)} 
    Sort_Reach = []

    if len(Product_Pre_States) != 0: #If want to optimize
        
        States_To_Consider = list(States_To_Consider)
        New_States_To_Consider = []
        for i in States_To_Consider:
            if Is_In_Q1[i] == 1 or Is_In_Q0[i] == 1:
                continue
            New_States_To_Consider.append(i)
            Sort_Reach.append([])
            for n in Actions[i]:
                Reach[n][i].sort(key=d.get)
                Sort_Reach[-1].append(list(Reach[n][i]))
        States_To_Consider = list(New_States_To_Consider) 
    
    else:    

        for i in range(len(Reach)):
            Sort_Reach.append([])
            for j in range(IA_l.shape[1]):
                Sort_Reach[-1].append([])
        
        for j in range(IA_l.shape[1]):        
            if Is_In_Q1[j] == 0:
                for k in range(len(Actions[j])):
                    Reach[Actions[j][k]][j].sort(key=d.get)
                    Sort_Reach[Actions[j][k]][j] = list(Reach[Actions[j][k]][j])
            else:
                continue
    
    if len(Product_Pre_States) != 0:
        Phi_Max = {}
        Phi_Max, List_Values = Phi_Synthesis_Max_Upper_Parallel(IA_l, IA_u, Descending_Order, Q1, Reach, Sort_Reach, Actions, States_To_Consider, Phi_Max, Index_Vector)   
    else:           
        Phi_Max = Phi_Synthesis_Max_Upper(IA_l, IA_u, Descending_Order, Q1, Reach, Sort_Reach, Actions)   

 
    if len(Product_Pre_States) != 0:
                
        New_States_To_Consider = set([])
        for i, State_Con in enumerate(States_To_Consider):
            
            Values = list(List_Values[i])
            Bounds_All_Act[State_Con] = list(Values)
            Index_Max = np.argmax(Values)
            Index_Vector[State_Con,0] = Values[Index_Max]
            Optimal_Policy[State_Con] = Actions[State_Con][Index_Max]

                 
            if Index_Vector[State_Con,0] > 0:
                for n in range(len(Product_Pre_States)):
                    for k in Product_Pre_States[n][State_Con]:                    
                        if Is_In_Q0[k] != 1 and Is_In_Q1[k] != 1:
                            New_States_To_Consider.add(k)
                
                
        States_To_Consider = list(New_States_To_Consider)
        
    else:  
        
        Steps_Low = sparse.csr_matrix.dot(sparse.csr_matrix(Phi_Max), Index_Vector)
        
        for i in range(IA_l.shape[1]):
            if Is_In_Q1[i] == 1: continue
            List_Values = []
            for k in range(len(Actions[i])):
                List_Values.append(IA_l.shape[0]*i+Actions[i][k])
            Values = list(Steps_Low[List_Values])
            Index_Vector[i,0] = np.amax(Values)
            Optimal_Policy[i] = Actions[i][np.argmax(Values)]

    for i in range(len(Q1)):    
        Index_Vector[Q1[i],0] = 1.0
        
    
    Success_Intervals = list([])    
 
    if len(Product_Pre_States) != 0:
        for i in range(IA_l['Num_S']):       
            Success_Intervals.append(Index_Vector[i,0])         
    else:    
        for i in range(IA_l.shape[1]):       
            Success_Intervals.append(Index_Vector[i,0]) 
        

    Terminate_Check = 0
    Convergence_threshold = 0.001
    count = 0
           

    while Terminate_Check == 0:
        
        count += 1
                   
       
        Descending_Order = np.array(range(len(Success_Intervals)))
        Success_Array = np.array(Success_Intervals)
        Descending_Order = list(Descending_Order[(-Success_Array).argsort()]) 


        
        d = {k:v for v,k in enumerate(Descending_Order)} 
        Sort_Reach = list([])
        
        if len(Product_Pre_States) != 0:
            for i in States_To_Consider:
                Sort_Reach.append([])        
                for n in Actions[i]:
                    Reach[n][i].sort(key=d.get)
                    Sort_Reach[-1].append(list(Reach[n][i]))
        else:
            
            for i in range(len(Reach)):
                Sort_Reach.append([])
                for j in range(IA_l.shape[1]):
                    Sort_Reach[-1].append([])
            
            for j in range(IA_l.shape[1]):        
                if Is_In_Q1[j] == 0:
                    for k in range(len(Actions[j])):
                        Reach[Actions[j][k]][j].sort(key=d.get)
                        Sort_Reach[Actions[j][k]][j] = list(Reach[Actions[j][k]][j])
                else:
                    continue
        if len(Product_Pre_States)!= 0:
            Phi_Max, List_Values = Phi_Synthesis_Max_Upper_Parallel(IA_l, IA_u, Descending_Order, Q1, Reach, Sort_Reach, Actions, States_To_Consider, Phi_Max, Index_Vector)                     
        else:
            Phi_Max = Phi_Synthesis_Max_Upper(IA_l, IA_u, Descending_Order, Q1, Reach, Sort_Reach, Actions)



        if len(Product_Pre_States) != 0:
            

            
            New_States_To_Consider = set([])
            Max_Difference = 0

            for i, State_Con in enumerate(States_To_Consider):

                Values = list(List_Values[i])
                Bounds_All_Act[State_Con] = list(Values)
                Index_Max = np.argmax(Values)
                Index_Vector[State_Con,0] = Values[Index_Max]
                Optimal_Policy[State_Con] = Actions[State_Con][Index_Max]
                Abs_Diff = abs(Index_Vector[State_Con,0] - Success_Intervals[State_Con]) 


                if Abs_Diff > 0:
                    
                    Max_Difference = max(Max_Difference, Abs_Diff)
                    for n in Product_Pre_States:
                        for k in n[State_Con]:                    
                            if Is_In_Q0[k] != 1 and Is_In_Q1[k] != 1:
                                New_States_To_Consider.add(k)
                    
                    Success_Intervals[State_Con] = Index_Vector[State_Con,0]
                    
            States_To_Consider = list(New_States_To_Consider)
            
        else:    

            Steps_Low = sparse.csr_matrix.dot(sparse.csr_matrix(Phi_Max), Index_Vector[:,0])

            List_Values = list([])
            Bounds_All_Act = list([])
        
            for i in range(IA_l.shape[1]):
                if Is_In_Q1[i] == 1:
                    Bounds_All_Act.append(list([]))
                    for j in range(len(Actions[i])):
                        Bounds_All_Act[-1].append(1.0)
                    continue                
                List_Values.append([])         
                for k in range(len(Actions[i])):
                    List_Values[-1].append(IA_l.shape[0]*i+Actions[i][k])    
                Values = list(Steps_Low[List_Values[-1]])  
                Bounds_All_Act.append(Values)
                Index_Vector[i,0] = np.amax(Values)
                Optimal_Policy[i] = Actions[i][np.argmax(Values)] 
            
            for i in range(len(Q1)):    
                Index_Vector[Q1[i],0] = 1.0
                                       
            Max_Difference = 0
                               
            for i in range(IA_l.shape[1]):                            
                Max_Difference = max(Max_Difference, abs(Success_Intervals[i] - Index_Vector[i,0]))        
                Success_Intervals[i] = Index_Vector[i,0]
           
        if Max_Difference < Convergence_threshold:              
            Terminate_Check = 1    
    
    Bounds = []
    if len(Product_Pre_States) != 0:
        rows, cols, vals = [], [], []
        for key in Phi_Max:
                for i in range(len(Phi_Max[key][Optimal_Policy[key]][0])):
                    rows.append(key)
                    cols.append(Phi_Max[key][Optimal_Policy[key]][0][i])
                    vals.append(Phi_Max[key][Optimal_Policy[key]][1][i])       
        Phi_Max = sparse.csr_matrix((vals, (rows, cols)), shape=(len(Optimal_Policy), len(Optimal_Policy)))    

    else:    
        Indices = [int(i*IA_l.shape[0]+Optimal_Policy[i]) for i in range(len(Optimal_Policy))]
        Phi_Max = np.array(Phi_Max[Indices,:])
    
    for i in range(Num_States):
        Bounds.append(Success_Intervals[i*Automata_size+Init[i]])
            
    return (Bounds, Success_Intervals, Phi_Max, Optimal_Policy, Bounds_All_Act)






def Maximize_Upper_Bound_Reachability_Continuous(Q1, State_Space, Num_States, Automata_size, Reach, Init, Potential_Policy, Actions, Reachable_Sets, Low_Bounds_Prod, Pre_Quantitative_Product, Q0):
    
  
    
    Descending_Order = []
    Index_Vector = np.zeros((len(State_Space)*Automata_size,1))
    Is_In_Q1 = np.zeros((len(State_Space)*Automata_size))
    Is_In_Q0 = np.zeros((len(State_Space)*Automata_size))

    Part_Actions = [[] for x in range(len(State_Space)*Automata_size)]
    Was_Action_Deleted = [[] for x in range(len(State_Space)*Automata_size)] #Keeps track of the set of inputs from which a partitioned region has been removed
    States_To_Consider = set([])
    
    Set_Q1 = set(Q1)
    Set_Q0 = set(Q0)
    
    for k in range(len(State_Space)*Automata_size):

         
        if k in Set_Q1:            
            Index_Vector[k,0] = 1.0
            Descending_Order.insert(0,k)
            Is_In_Q1[k] = 1
            States_To_Consider = States_To_Consider.union(Pre_Quantitative_Product[k])
          
        elif k not in Set_Q0:  
           
            
            Descending_Order.append(k)
            
            #PARTITIONING THE INPUT SPACE OF EACH STATE NOT IN THE GREATEST LARGEST COMPONENT
            for j in range(len(Actions[k])):
                Part_Actions[k].append(list([]))
                Step_x = (Actions[k][j][1][0] - Actions[k][j][0][0])/float(Partition_Parameter)
                Step_y = (Actions[k][j][1][1] - Actions[k][j][0][1])/float(Partition_Parameter)
                Was_Action_Deleted[k].append(0)
                for y in range(int(Partition_Parameter)):
                    for x in range(int(Partition_Parameter)):
                        Part_Actions[k][j].append(list([[Actions[k][j][0][0]+x*Step_x, Actions[k][j][0][1]+y*Step_y], [Actions[k][j][0][0]+(x+1)*Step_x,Actions[k][j][0][1]+(y+1)*Step_y]]))
   

    for k in Q0:
        Is_In_Q0[k] = 1
        Descending_Order.append(k)
     
    States_To_Consider = list(States_To_Consider)    

    d = {k:v for v,k in enumerate(Descending_Order)} 
    Sort_Reach = []
    



    New_States_To_Consider = []
    for i in States_To_Consider:
        if Is_In_Q1[i] == 1 or Is_In_Q0[i] == 1:
            continue
        New_States_To_Consider.append(i)
        Sort_Reach.append([])
        Reach[i].sort(key=d.get)
        Sort_Reach[-1] = list(Reach[i])
        
            
    
    States_To_Consider = list(New_States_To_Consider)
    Phi_Min = {}

    Prob_Per_Mode = [[] for x in range(len(State_Space)*Automata_size)]
    
              
    Phi_Min, Index_Vector, Potential_Policy, Prob_Per_Mode = Phi_Synthesis_Max_Continuous_Removal_Parallel(State_Space, Descending_Order, Q1, Reach, Sort_Reach, Part_Actions, Reachable_Sets, Index_Vector, Potential_Policy, Automata_size, Q0, States_To_Consider, Phi_Min, Prob_Per_Mode)
    

    Copy_Prob_Per_Mode = []
    Success_Intervals = []
    States_To_Consider = set([] )      
    for i in range(len(State_Space)*Automata_size):
        if Index_Vector[i,0] > 0 and Is_In_Q1[i] != 1:

            for k in Pre_Quantitative_Product[i]:
                if Is_In_Q0[k] != 1 and Is_In_Q1[k] != 1:
                    States_To_Consider.add(k)
                
        
        Success_Intervals.append(float(Index_Vector[i,0]))
        Copy_Prob_Per_Mode.append(list([]))       
        Copy_Prob_Per_Mode[i] = list(Prob_Per_Mode[i])
    
        
    States_To_Consider = list(States_To_Consider)   
     
    Terminate_Check = 0
    Convergence_threshold = 0.01
    Previous_Max_Difference = 1
    count = 0
           
    
    while Terminate_Check == 0:
                   
               
        for i in Q1:
            Success_Intervals[i] = 1.0
       
        Descending_Order = np.array(range(len(Success_Intervals)))
        Success_Array = np.array(Success_Intervals)
        Descending_Order = list(Descending_Order[(-Success_Array).argsort()]) 
        
        d = {k:v for v,k in enumerate(Descending_Order)} 
        Sort_Reach = []

        
        for i in States_To_Consider:
            Sort_Reach.append([])        
            Reach[i].sort(key=d.get)
            Sort_Reach[-1] = list(Reach[i])

        
        Phi_Min, Index_Vector, Potential_Policy, Prob_Per_Mode = Phi_Synthesis_Max_Continuous_Removal_Parallel(State_Space, Descending_Order, Q1, Reach, Sort_Reach, Part_Actions, Reachable_Sets, Index_Vector, Potential_Policy, Automata_size, Q0, States_To_Consider, Phi_Min, Prob_Per_Mode)
        
                         
        Max_Difference = 0

        New_States_To_Consider = set([])
        
        
        for i in States_To_Consider:                               
            Success_Intervals[i] = Index_Vector[i,0]
            Max_Diff_State = 0
            if len(Copy_Prob_Per_Mode[i]) == 0:
                for k in Prob_Per_Mode[i]:
                    Max_Diff_State = max(Max_Diff_State, abs(k - 0.0))  
                    Max_Difference = max(Max_Difference, abs(k - 0.0))        
                    Copy_Prob_Per_Mode[i].append(k)               
            else:    
                for k in range(len(Copy_Prob_Per_Mode[i])):
                    Max_Diff_State = max(Max_Diff_State, abs(Prob_Per_Mode[i][k] - Copy_Prob_Per_Mode[i][k]))  
                    Max_Difference = max(Max_Difference, abs(Prob_Per_Mode[i][k] - Copy_Prob_Per_Mode[i][k]))        
                    Copy_Prob_Per_Mode[i][k] = Prob_Per_Mode[i][k]
                
            if Max_Diff_State > 0:

                for j in Pre_Quantitative_Product[i]:
                    if Is_In_Q1[j] == 0 and Is_In_Q0[j] == 0:
                        New_States_To_Consider.add(j)
                           
            
        States_To_Consider = list(New_States_To_Consider)    

        count = count + 1
        
        if Max_Difference < Convergence_threshold or count > Count_Cut_Off:  #To prevent infinite loop due to numerical error            
            Terminate_Check = 1    
    
    Bounds = []
    
    rows, cols, vals = [], [], []
    for key in Phi_Min:
            for i in range(len(Phi_Min[key][0])):
                rows.append(key)
                cols.append(Phi_Min[key][0][i])
                vals.append(Phi_Min[key][1][i])       
    Phi_Min = sparse.csr_matrix((vals, (rows, cols)), shape=(len(Potential_Policy), len(Potential_Policy)))    

    
    for i in range(Num_States):
        Bounds.append(Success_Intervals[i*Automata_size+Init[i]])
    
        
    #Deleting Suboptimal Actions
    
    
    List_Optimality_Factors = []
        
    

    for i in range(len(Copy_Prob_Per_Mode)):
        if Is_In_Q1[i] == 0 and Is_In_Q0[i] == 0 and len(Copy_Prob_Per_Mode[i]) != 0:
            m = 0
            Tag = 0
            Worst_Factor = 0.0
            To_Delete_Actions = list([])
            Original_Length = len(list(Actions[i]))
                       
            while (m != Original_Length):
                To_Delete = list([])
                for k in range(int(Partition_Parameter)*int(Partition_Parameter)):
                    if Copy_Prob_Per_Mode[i][int(Partition_Parameter)*int(Partition_Parameter)*m + k] < Low_Bounds_Prod[i]:
                        To_Delete.append(k)
                        Was_Action_Deleted[i][m] = 1
                    else:
                        Worst_Factor = max(Worst_Factor, Copy_Prob_Per_Mode[i][int(Partition_Parameter)*int(Partition_Parameter)*m + k] - Low_Bounds_Prod[i] )
                    
                if Was_Action_Deleted[i][m] == 1:
                    Tag = 1
                    To_Delete_Actions.append(m)
                    
                    
                    for ele in sorted(To_Delete, reverse = True):  
                        del Part_Actions[i][m][ele]

                    for x in range(len(Part_Actions[i][m])):
                        Actions[i].append(list(Part_Actions[i][m][x]))
                m = m+1

                    
            if Tag == 1:

                if len(To_Delete_Actions) == len(Actions[i]): #Due to numerical errors when the states are realy small, the algorithm might delete all the available actions when all remaining inputs are pretty much equivalent. To avoid this, we randomly keep an action
                    To_Delete_Actions.pop()
                    
                for ele in sorted(To_Delete_Actions, reverse = True):  
                        del Actions[i][ele] 
                 
            List_Optimality_Factors.append(Worst_Factor)
            
        else:
            List_Optimality_Factors.append(0.0)
     

    return (Bounds, Success_Intervals, Phi_Min, Potential_Policy, List_Optimality_Factors, Actions)






def Phi_Synthesis_Max_Lower(Lower, Upper, Order_A, q1, Reach, Reach_Sort, Action):
    
    Phi_min = np.zeros((Upper.shape[1]*Upper.shape[0], Upper.shape[1]))

    for j in range(Upper.shape[1]):
        
        if j in q1:
            continue
        else:
            
            for k in range(len(Action[j])):
                if len(Reach[Action[j][k]][j]) == 0:
                    continue
                Up = Upper[Action[j][k]][j][:]
                Low = Lower[Action[j][k]][j][:]                 
                Sum_1_A = 0.0
                Sum_2_A = sum(Low[Reach[Action[j][k]][j]])
                Phi_min[j*Upper.shape[0]+Action[j][k]][Reach_Sort[Action[j][k]][j][0]] = min(Low[Reach_Sort[Action[j][k]][j][0]] + 1 - Sum_2_A, Up[Reach_Sort[Action[j][k]][j][0]])  
          
                for i in range(1, len(Reach_Sort[Action[j][k]][j])):
                                 
                    Sum_1_A = Sum_1_A + Phi_min[j*Upper.shape[0]+Action[j][k]][Reach_Sort[Action[j][k]][j][i-1]]
                    if Sum_1_A >= 1:
                        break
                    Sum_2_A = Sum_2_A - Low[Reach_Sort[Action[j][k]][j][i-1]]
                    Phi_min[j*Upper.shape[0]+Action[j][k]][Reach_Sort[Action[j][k]][j][i]] = min(Low[Reach_Sort[Action[j][k]][j][i]] + 1 - (Sum_1_A+Sum_2_A), Up[Reach_Sort[Action[j][k]][j][i]])                 
    return Phi_min


def Phi_Synthesis_Max_Lower_Parallel(Lower, Upper, Order_A, q1, Reach, Reach_Sort, Action, States_To_Consider, Phi_min, Index_Vector):
    
    
    List_Values = []
    
    for j, State in enumerate(States_To_Consider):
            List_Values.append([])            
            Phi_min[State] = {}
            for k, Act in enumerate(Action[State]):
                Phi_min[State][Act] = [Reach_Sort[j][k],[]]

                
                Low = [Lower[Act,State,Reach_Sort[j][k][n]] for n in range(len(Reach_Sort[j][k]))]


                Sum_1_A = 0.0
                Sum_2_A = sum(Low)
                
                
                Values = Upper[Act,State,Reach_Sort[j][k][0]]
                Phi_min[State][Act][1].append(Values)
                Val = Values*Index_Vector[Reach_Sort[j][k][0],0]
                
                for i, Reach_State in enumerate(Reach_Sort[j][k][1:], 1):                                
                    Sum_1_A += Values
                    if Sum_1_A >= 1:
                        Phi_min[State][Act][1].extend([0]*(len(Reach_Sort[j][k]) - i))
                        break
                    Sum_2_A -= Low[i-1]
                    Values = min(Low[i] + 1 - (Sum_1_A+Sum_2_A), Upper[Act,State,Reach_State])
                    Phi_min[State][Act][1].append(Values)
                    Val += Values*Index_Vector[Reach_State,0]                
                List_Values[-1].append(Val)
    
             

    return Phi_min, List_Values


def Low_Parallel(States_To_Consider, Action, Reach_Sort, Upper_Shape0, Upper_Shape1, Up_list, Low_list):
    
    Phi_min_list = []
    
    for k in range(len(Action)):                
        Phi_min = np.zeros(Upper_Shape1)
        Up = np.array(Up_list[k])
        Low = np.array(Low_list[k])                 
        Sum_1_A = 0.0
        Sum_2_A = sum(Low)    
        Phi_min[0] = min( Low[0] + 1 - Sum_2_A, Up[0] )  

        for i in range(1, len(Reach_Sort[k])):                                
            Sum_1_A = Sum_1_A + Phi_min[i-1]
            if Sum_1_A >= 1:
                break
            Sum_2_A = Sum_2_A - Low[i-1]
            Phi_min[i] = min(Low[i] + 1 - (Sum_1_A+Sum_2_A), Up[i])                 
        
        Phi_min_list.append(Phi_min)
    
    return Phi_min_list

def Phi_Synthesis_Max_Continuous_Parallel(State_Space, Order_A, q1, Reach, Reach_Sort, Inputs, Reachable_Sets, Index_Vector, Optimal_Policy, Automata_size, q0, Phi_min, States_To_Consider):
    
    start = timeit.default_timer()
    
    res = Parallel(n_jobs=4)(delayed(Parallel_Optimization_Lower)(States_To_Consider[j], Reach_Sort[j], Inputs[States_To_Consider[j]], Index_Vector, j, Reachable_Sets[States_To_Consider[j]/Automata_size][0][0], Reachable_Sets[States_To_Consider[j]/Automata_size][1][0], Reachable_Sets[States_To_Consider[j]/Automata_size][0][1], Reachable_Sets[States_To_Consider[j]/Automata_size][1][1], [ [[State_Space[Reach_Sort[j][k]/Automata_size][0][0], State_Space[Reach_Sort[j][k]/Automata_size][0][1]], [State_Space[Reach_Sort[j][k]/Automata_size][1][0], State_Space[Reach_Sort[j][k]/Automata_size][1][1]]] for k in range(len(Reach_Sort[j]))] ) for j in range(len(States_To_Consider)))
           
    
    for j, res_j in enumerate(res):
        State_Con = States_To_Consider[j]
        Phi_min[State_Con] = [Reach_Sort[j], []]
        for k in res_j[2]:
            Phi_min[State_Con][1].append(k) 
        Index_Vector[State_Con,0] = res_j[0]
        Optimal_Policy[State_Con] = list(res_j[1])
                        
    return Phi_min, Index_Vector, Optimal_Policy



def Parallel_Optimization_Lower(States_To_Consider, Reach_Sort, Inputs, Index_Vector, j, Reach_a_x, Reach_b_x, Reach_a_y, Reach_b_y, L):
       
    f_up_list = []
    f_low_list = []
    f_up_jac = list([])
    f_low_jac = list([])            
                
    for L_k in L:

        a_x = L_k[0][0]
        b_x = L_k[1][0]
        a_y = L_k[0][1]
        b_y = L_k[1][1]        
        
        f_up_list.append(lambda x, a_x=a_x, a_y = a_y, b_x= b_x, b_y=b_y, Reach_a_x = Reach_a_x ,Reach_b_x = Reach_b_x, Reach_a_y = Reach_a_y, Reach_b_y= Reach_b_y: Upper_Bound_Func(x[0],x[1],a_x,b_x,a_y,b_y,Reach_a_x,Reach_b_x,Reach_a_y,Reach_b_y))
        f_low_list.append(lambda x, a_x=a_x, a_y = a_y, b_x= b_x, b_y=b_y, Reach_a_x = Reach_a_x ,Reach_b_x = Reach_b_x, Reach_a_y = Reach_a_y, Reach_b_y= Reach_b_y: Lower_Bound_Func(x[0],x[1],a_x,b_x,a_y,b_y,Reach_a_x,Reach_b_x,Reach_a_y,Reach_b_y))


    Rea = [[] for x in range(j+1)]
    Rea[j] = list(Reach_Sort)
                     
    Best = 0.0
    Best_Act = [uniform(Inputs[0][0][0], Inputs[0][1][0]),uniform(Inputs[0][0][1], Inputs[0][1][1])] 
    f_obj_min = lambda x , j=j : f_objective(x, f_up_list, f_low_list, f_up_jac, f_low_jac, Index_Vector, Rea, j)[0]

    
    for Input_Set in Inputs:
        
        l_x = Input_Set[0][0]
        u_x = Input_Set[1][0]
        l_y = Input_Set[0][1]
        u_y = Input_Set[1][1]
      
        Volume_Set = (u_x - l_x)*(u_y - l_y)

        Num_Points = max(int(Initial_Num_Points*(float(Volume_Set)/Init_Volume)), Min_Num_Points)
        X, Y = mgrid[l_x:l_y:complex(0,Num_Points), u_x:u_y:complex(0,Num_Points)]
        positions = vstack([X.ravel(), Y.ravel()])      
        loc_best = 0
        Best_Act_Loc =[uniform(l_x, u_x),uniform(l_y, u_y)] 

        for i in range(len(positions)):
            res = minimize(f_obj_min, [positions[0][i],positions[1][i]], method='L-BFGS-B', bounds=[(l_x,u_x),(l_y, u_y)] )
            
            if -res.fun > loc_best:
                loc_best= min(1.0, -res.fun)
                Best_Act_Loc = list([res.x[0], res.x[1]])
                
        if loc_best > Best:
            Best = loc_best
            Best_Act = list(Best_Act_Loc)
                     

     
    z_vector = list(f_objective(Best_Act, f_up_list, f_low_list, f_up_jac, f_low_jac, Index_Vector, Rea, j)[1])                                

    
    return Best, Best_Act, z_vector



def Phi_Synthesis_Max_Continuous_Removal_Parallel(State_Space, Order_A, q1, Reach, Reach_Sort, Inputs, Reachable_Sets, Index_Vector, Optimal_Policy, Automata_size, q0, States_To_Consider, Phi_min, Prob_Per_Mode):
    
    
    start = timeit.default_timer()
    
    res = Parallel(n_jobs=4)(delayed(Parallel_Optimization_Upper)(States_To_Consider[j], Reach_Sort[j], Inputs[States_To_Consider[j]], Index_Vector, j, Reachable_Sets[States_To_Consider[j]/Automata_size][0][0], Reachable_Sets[States_To_Consider[j]/Automata_size][1][0], Reachable_Sets[States_To_Consider[j]/Automata_size][0][1], Reachable_Sets[States_To_Consider[j]/Automata_size][1][1], [ [[State_Space[Reach_Sort[j][k]/Automata_size][0][0], State_Space[Reach_Sort[j][k]/Automata_size][0][1]], [State_Space[Reach_Sort[j][k]/Automata_size][1][0], State_Space[Reach_Sort[j][k]/Automata_size][1][1]]] for k in range(len(Reach_Sort[j]))]) for j in range(len(States_To_Consider)))
    
    for j, res_j in enumerate(res):
        State_Con = States_To_Consider[j]
        Phi_min[State_Con] = [Reach_Sort[j], []]
        for k in res_j[2]:
            Phi_min[State_Con][1].append(k) 
        Index_Vector[State_Con,0] = res_j[0]
        Optimal_Policy[State_Con] = list(res_j[1])
        Prob_Per_Mode[State_Con] = list(res_j[3])
                        
    return Phi_min, Index_Vector, Optimal_Policy, Prob_Per_Mode


def Parallel_Optimization_Upper(States_To_Consider, Reach_Sort, Inputs, Index_Vector, j, Reach_a_x, Reach_b_x, Reach_a_y, Reach_b_y, L):
    

    New_Prob_Per_Mode = list([])

    f_up_list = list([])
    f_low_list = list([])
    f_up_jac = list([])
    f_low_jac = list([])    

                 
    for L_k in L:
        
        a_x = L_k[0][0]
        b_x = L_k[1][0]
        a_y = L_k[0][1]
        b_y = L_k[1][1]         
        
        f_up_list.append(lambda x, a_x=a_x, a_y = a_y, b_x= b_x, b_y=b_y, Reach_a_x = Reach_a_x ,Reach_b_x = Reach_b_x, Reach_a_y = Reach_a_y, Reach_b_y= Reach_b_y: Upper_Bound_Func(x[0],x[1],a_x,b_x,a_y,b_y,Reach_a_x,Reach_b_x,Reach_a_y,Reach_b_y))
        f_low_list.append(lambda x, a_x=a_x, a_y = a_y, b_x= b_x, b_y=b_y, Reach_a_x = Reach_a_x ,Reach_b_x = Reach_b_x, Reach_a_y = Reach_a_y, Reach_b_y= Reach_b_y: Lower_Bound_Func(x[0],x[1],a_x,b_x,a_y,b_y,Reach_a_x,Reach_b_x,Reach_a_y,Reach_b_y))

  
    Rea = [[] for x in range(j+1)]
    Rea[j] = list(Reach_Sort)
    
    Best = 0.0               
    Best_Act = [uniform(Inputs[0][0][0][0], Inputs[0][0][1][0]),uniform(Inputs[0][0][0][1], Inputs[0][0][1][1])] 

    f_obj_min = lambda x , j=j : f_objective(x, f_up_list, f_low_list, f_up_jac, f_low_jac, Index_Vector, Rea, j)[0]


    for k in Inputs:
        for Input_Set in k:
 
            l_x = Input_Set[0][0]
            u_x = Input_Set[1][0]
            l_y = Input_Set[0][1]
            u_y = Input_Set[1][1]
            Volume_Set = (u_x - l_x)*(u_y - l_y)

            Num_Points = max(int(Initial_Num_Points*(float(Volume_Set)/float(Init_Volume))), Min_Num_Points)
            X, Y = mgrid[l_x:l_y:complex(0,Num_Points), u_x:u_y:complex(0,Num_Points)]
            positions = vstack([X.ravel(), Y.ravel()])      
            loc_best = 0
            Best_Act_Loc =[uniform(l_x, u_x),uniform(l_y, u_y)] 

            for i in range(len(positions)):
                res = minimize(f_obj_min, [positions[0][i],positions[1][i]], method='L-BFGS-B', bounds=[(l_x,u_x),(l_y, u_y)] )
            
                if -res.fun > loc_best:
                    loc_best= min(1.0, -res.fun)
                    Best_Act_Loc = list([res.x[0], res.x[1]])

            New_Prob_Per_Mode.append(loc_best)
            
            if loc_best > Best:
                Best = loc_best
                Best_Act = Best_Act_Loc

    
    z_vector = list(f_objective(Best_Act, f_up_list, f_low_list, f_up_jac, f_low_jac, Index_Vector, Rea, j)[1])

    return Best, Best_Act, z_vector, New_Prob_Per_Mode




def Phi_Synthesis_Max_Upper(Lower, Upper, Order_D, q1, Reach, Reach_Sort, Action):
    
    Phi_max = np.zeros((Upper.shape[1]*Upper.shape[0], Upper.shape[1]))
    
    for j in range(Upper.shape[1]):
        
        if j in q1:
            continue
        else:
    
            for k in range(len(Action[j])):

                Up = Upper[Action[j][k]][j][:]
                Low = Lower[Action[j][k]][j][:] 
                Sum_1_D = 0.0
                Sum_2_D = sum(Low[Reach[Action[j][k]][j]])
                Phi_max[j*Upper.shape[0]+Action[j][k]][Reach_Sort[Action[j][k]][j][0]] = min(Low[Reach_Sort[Action[j][k]][j][0]] + 1 - Sum_2_D, Up[Reach_Sort[Action[j][k]][j][0]])  
          
                for i in range(1, len(Reach_Sort[Action[j][k]][j])):
                                 
                    Sum_1_D = Sum_1_D + Phi_max[j*Upper.shape[0]+Action[j][k]][Reach_Sort[Action[j][k]][j][i-1]]
                    if Sum_1_D >= 1:
                        break
                    Sum_2_D = Sum_2_D - Low[Reach_Sort[Action[j][k]][j][i-1]]
                    Phi_max[j*Upper.shape[0]+Action[j][k]][Reach_Sort[Action[j][k]][j][i]] = min(Low[Reach_Sort[Action[j][k]][j][i]] + 1 - (Sum_1_D+Sum_2_D), Up[Reach_Sort[Action[j][k]][j][i]])  
               
    return Phi_max

def Phi_Synthesis_Max_Upper_Parallel(Lower, Upper, Order_D, q1, Reach, Reach_Sort, Action, States_To_Consider, Phi_min, Index_Vector):
    


    List_Values = []
    
    for j, State in enumerate(States_To_Consider):
            List_Values.append([])            
            Phi_min[State] = {}
            for k, Act in enumerate(Action[State]):
                Phi_min[State][Act] = [Reach_Sort[j][k],[]]
                Low = [Lower[Act,State,Reach_Sort[j][k][n]] for n in range(len(Reach_Sort[j][k]))]
                Sum_1_A = 0.0
                Sum_2_A = sum(Low)    
                Values = Upper[Act,State,Reach_Sort[j][k][0]] ###FOR THE TYPE OF ABSTRACTIONS WE CONSTRUCT, I BELIEVE THAT THE UPPER BOUND OF THE FIRST STATE IN THE ORDERING IS ALWAYS ACHIEVABLE

                Phi_min[State][Act][1].append(Values)
                Val = Values*Index_Vector[Reach_Sort[j][k][0],0]
                
                for i, Reach_State in enumerate(Reach_Sort[j][k][1:], 1):                                
                    Sum_1_A += Values
                    if Sum_1_A >= 1:
                        Phi_min[State][Act][1].extend([0]*(len(Reach_Sort[j][k]) - i))
                    Sum_2_A -= Low[i-1]
                    Values = min(Low[i] + 1 - (Sum_1_A+Sum_2_A), Upper[Act,State,Reach_State])                    
                    Phi_min[State][Act][1].append(Values)
                    Val += Values*Index_Vector[Reach_State,0]
    
                List_Values[-1].append(Val)

    return Phi_min, List_Values

def Upp_Parallel(States_To_Consider, Action, Reach_Sort, Upper_Shape0, Upper_Shape1, Up_list, Low_list):
    
    Phi_min_list = []
    
    for k in range(len(Action)):                
        Phi_min = np.zeros(Upper_Shape1)
        Up = np.array(Up_list[k])
        Low = np.array(Low_list[k])                 
        Sum_1_A = 0.0
        Sum_2_A = sum(Low)    
        Phi_min[0] = min( Low[0] + 1 - Sum_2_A, Up[0] )  

        for i in range(1, len(Reach_Sort[k])):                                
            Sum_1_A = Sum_1_A + Phi_min[i-1]
            if Sum_1_A >= 1:
                break
            Sum_2_A = Sum_2_A - Low[i-1]
            Phi_min[i] = min(Low[i] + 1 - (Sum_1_A+Sum_2_A), Up[i])                 
        
        Phi_min_list.append(Phi_min)
    
    return Phi_min_list




def State_Space_Refinement_BMDP(State_Space, Threshold_Uncertainty, Greatest_Suboptimality_Factor, Objective, Potential_Policy, Optimal_Policy, Best_Markov_Chain, Worst_Markov_Chain, first, States_Above_Threshold, Success_Intervals, Product_Intervals, Len_Automata, Lower_Bound_Matrix, Upper_Bound_Matrix, Automata, Automata_Accepting, L_mapping, IA1_u, IA1_l, Is_In_Permanent_Comp, Allowable_Action_Potential, Allowable_Action_Permanent, List_Potential_Winning_Components, Which_Potential_Winning_Component, Bridge_Winning_Components, Is_In_Potential_Winning_Component, List_Potential_Losing_Components, Which_Potential_Losing_Component, Bridge_Losing_Components, Is_In_Potential_Losing_Component, Reachable_States, Is_Bridge_State, Bridge_Transitions, Pre_States, Product_Reachable_States, List_Permanent_Accepting_BSCC, Previous_Accepting_BSCC, Previous_Non_Accepting_BSCC, Allowable_Actions, Discrete_Input_Space, List_Max_Opt, List_Avg_Opt, Fractions_Above, Running_Times, Greatest_Permanent_Winning_Component, Previous_Potential_Winning_Components, Permanent_Losing_Components, Avg_Num_Actions):
    
    Tag = 0 #Counting how many times we went through refinement
    V_Total = 1 #Dummy total volume, might be useful for ranking of states later 
        
    while(Greatest_Suboptimality_Factor > Threshold_Uncertainty):
        
        start = timeit.default_timer()
        
        Tag += 1
        Set_Refinement = []
        
        
        States_Above_Threshold_Product = list(States_Above_Threshold)
        

        Is_in_WC_P = np.zeros(IA1_l['Num_S'])
        Is_in_LC_P = np.zeros(IA1_l['Num_S'])    
        
        Weights = Uncertainty_Ranking_Path_New3(Best_Markov_Chain, Worst_Markov_Chain, States_Above_Threshold_Product, Product_Intervals, IA1_u, IA1_l, Is_In_Potential_Winning_Component, Which_Potential_Winning_Component, Bridge_Winning_Components, Is_In_Potential_Losing_Component, Which_Potential_Losing_Component, Bridge_Losing_Components, Is_in_WC_P, [], [], Is_in_LC_P, [], [], Is_In_Permanent_Comp, State_Space, V_Total, Len_Automata)


        # The lines below convert uncertainty in the product IMC into uncertainty in the original system

        Weights_Original_S = [0.0 for n in range(State_Space.shape[0])]     
        for j in range(len(Weights)):
            Weights_Original_S[j/Len_Automata] += Weights[j]            
        Uncertainty_Ranking_Original_S = list(reversed(sorted(range(len(Weights_Original_S)), key=lambda x: Weights_Original_S[x])))        
        Weights_Original_S = list(reversed(sorted(Weights_Original_S)))

        Index_Stop = -1   
        Cut_off_Percentage = 0.05
        for j in range(len(Weights_Original_S)):
            if (Weights_Original_S[j]/Weights_Original_S[0]) < Cut_off_Percentage:
                Index_Stop = j
                break
            if Weights_Original_S[j] <= 0.0:
                Index_Stop = j
                break

        if Index_Stop == -1: Index_Stop = len(Weights_Original_S)        
        Index_Stop1 = (State_Space.shape[0]) #Parameter that sets the maximum number of states to be refined       
        Num_States = min(len(Uncertainty_Ranking_Original_S), Index_Stop, Index_Stop1)    

        for j in range(Num_States):       
            Set_Refinement.append(Uncertainty_Ranking_Original_S[j])

        Set_Refinement.sort()            
        New_States = Raw_Refinement(Set_Refinement, State_Space)


        Previous_Potential_Winning_Components = [item for sublist in List_Potential_Winning_Components for item in sublist]
        Previous_Potential_Winning_Components.sort()        

        """ Deleting previous state and replacing it with new states, as well as updating index for some lists """

        First_Loop = np.ones(IA1_l['Num_Act'])
        First_Loop = First_Loop.astype(int)
        First_Loop2 = np.ones(IA1_l['Num_Act'])
        First_Loop2 = First_Loop2.astype(int)
        
        Index_Stop = np.zeros((IA1_l['Num_Act'],State_Space.shape[0]+len(Set_Refinement)))
        Index_Stop = Index_Stop.astype(int)
        Index_Stop2 = np.zeros((IA1_l['Num_Act'],State_Space.shape[0]+len(Set_Refinement)))
        Index_Stop2 = Index_Stop2.astype(int) 
        Index_Stop7 = 0
        Index_Stop8 = 0
        Index_Stop9 = 0


        Copy_Greatest_Permanent_Winning_Component = list(Greatest_Permanent_Winning_Component)

        Copy_Permanent_Losing_Components = list(Permanent_Losing_Components)

        Copy_Previous_Potential_Winning_Components = list(Previous_Potential_Winning_Components)
                
        Is_New_State = [0]*(State_Space.shape[0]+len(Set_Refinement))
        Set_Refinement2 = []
        
        start1004 = timeit.default_timer()
        Time_Updating_Matrices = 0
        
        for m, Set in enumerate(Set_Refinement):
                       
            
            State_Space = np.concatenate((State_Space[:Set+1+m], np.asarray([New_States[2*m]]), State_Space[Set+1+m:]))                        
            State_Space = np.concatenate((State_Space[:Set+1+m], np.asarray([New_States[2*m+1]]), State_Space[Set+1+m:]))                        
            State_Space = np.delete(State_Space, Set+m, 0) 

            for z in range(Len_Automata):
                if len(Previous_Potential_Winning_Components) != 0:
                    Allowable_Action_Permanent.insert(Set*Len_Automata+Len_Automata+z+m*Len_Automata, list(Allowable_Action_Permanent[Set*Len_Automata+m*Len_Automata+z]))
                    Allowable_Action_Potential.insert(Set*Len_Automata+Len_Automata+z+m*Len_Automata, list(Allowable_Action_Potential[Set*Len_Automata+m*Len_Automata+z]))
                Allowable_Actions.insert(Set*Len_Automata+Len_Automata+z+m*Len_Automata, list(Allowable_Actions[Set*Len_Automata+m*Len_Automata+z]))
                Optimal_Policy = np.concatenate((Optimal_Policy[:Set*Len_Automata+Len_Automata+z+m*Len_Automata], [np.asarray(Optimal_Policy[Set*Len_Automata+m*Len_Automata+z])],Optimal_Policy[Set*Len_Automata+Len_Automata+z+m*Len_Automata:] ))
                Potential_Policy = np.concatenate((Potential_Policy[:Set*Len_Automata+Len_Automata+z+m*Len_Automata], [np.asarray(Potential_Policy[Set*Len_Automata+m*Len_Automata+z])],Potential_Policy[Set*Len_Automata+Len_Automata+z+m*Len_Automata:] ))
                Is_In_Permanent_Comp = np.concatenate((Is_In_Permanent_Comp[:Set*Len_Automata+Len_Automata+z+m*Len_Automata], [np.asarray(Is_In_Permanent_Comp[Set*Len_Automata+m*Len_Automata+z])],Is_In_Permanent_Comp[Set*Len_Automata+Len_Automata+z+m*Len_Automata:] ))


            start12 = timeit.default_timer()
            
            for z in range(len(Lower_Bound_Matrix)):
                for n in range(len(Lower_Bound_Matrix[z])):
                    Lower_Bound_Matrix[z][n].insert(Set+1+m, 0)
                    Upper_Bound_Matrix[z][n].insert(Set+1+m, 0)
                Lower_Bound_Matrix[z].insert(Set+1+m, blist([0]*len(Lower_Bound_Matrix[z][0])))
                Upper_Bound_Matrix[z].insert(Set+1+m, blist([0]*len(Upper_Bound_Matrix[z][0])))                

            

            
            Time_Updating_Matrices += timeit.default_timer() - start12
            
            L_mapping.insert(Set+m+1, L_mapping[Set+m])
            
            (First_Loop, Reachable_States, Set_Refinement, Index_Stop, m) = Index_Update(First_Loop, Reachable_States, Set_Refinement, Index_Stop, m)
            (First_Loop2, Bridge_Transitions, Set_Refinement, Index_Stop2, m) = Index_Update(First_Loop2, Bridge_Transitions, Set_Refinement, Index_Stop2, m)

            if len(Previous_Potential_Winning_Components) != 0: 
                (Previous_Potential_Winning_Components, Copy_Previous_Potential_Winning_Components, Index_Stop7) = Index_Update_Product_Continuous(Previous_Potential_Winning_Components, Copy_Previous_Potential_Winning_Components, Set_Refinement, Index_Stop7, Len_Automata, m)
            
            (Greatest_Permanent_Winning_Component, Copy_Greatest_Permanent_Winning_Component, Index_Stop8) = Index_Update_Product_Continuous(Greatest_Permanent_Winning_Component, Copy_Greatest_Permanent_Winning_Component, Set_Refinement, Index_Stop8, Len_Automata, m)
            (Permanent_Losing_Components, Copy_Permanent_Losing_Components, Index_Stop9) = Index_Update_Product_Continuous(Permanent_Losing_Components, Copy_Permanent_Losing_Components, Set_Refinement, Index_Stop9, Len_Automata, m)

            Is_New_State[Set + m] = 1
            Is_New_State[Set + m + 1] = 1
            Set_Refinement2.append(Set+m+1)
            Set_Refinement[m] = Set + m           




        """ _______ """

        Is_Bridge_State = np.zeros((len(Bridge_Transitions), len(Bridge_Transitions[0])))
        Is_Bridge_State = Is_Bridge_State.astype(int)
        
        for z in range(len(Bridge_Transitions)):
            for j in range(len(Bridge_Transitions[z])):
                if len(Bridge_Transitions[z][j]) != 0:
                    Is_Bridge_State[z,j] = 1

                    
        Reachable_Sets = Reachable_Sets_Computation_Finite(State_Space, Discrete_Input_Space) 

      
        (Lower_Bound_Matrix, Upper_Bound_Matrix, Reachable_States, Is_Bridge_State, Bridge_Transitions) = BMDP_Computation_Refinement(Reachable_Sets, State_Space, Set_Refinement, Set_Refinement2, Lower_Bound_Matrix, Upper_Bound_Matrix, Is_New_State, Reachable_States, Is_Bridge_State, Bridge_Transitions)
        (IA1_l, IA1_u, Is_Accepting, Is_Non_Accepting, Which_Accepting_Pair, Which_Non_Accepting_Pair, Product_Reachable_States, Product_Is_Bridge_State, Product_Bridge_Transitions, Init, Product_Pre_States) = Build_Product_BMDP_Refinement(Lower_Bound_Matrix, Upper_Bound_Matrix, Automata, L_mapping, Automata_Accepting, Reachable_States, Is_Bridge_State, Bridge_Transitions, Allowable_Actions, Is_In_Permanent_Comp)


        Success_Intervals = [[] for n in range(State_Space.shape[0])]    
        Product_Intervals = [[] for n in range(IA1_l['Num_Act'])]

        
        if Objective == 0:
                         
           if len(Previous_Potential_Winning_Components) != 0: 
               
                IA1_l_Array = np.zeros(((IA1_l['Num_Act']),IA1_l['Num_S'],IA1_l['Num_S']))
                IA1_u_Array = np.zeros(((IA1_l['Num_Act']),IA1_l['Num_S'],IA1_l['Num_S']))
                   
                for k in IA1_l.keys():
                    if isinstance(k,tuple):
                        IA1_l_Array[k[0],k[1],k[2]] = IA1_l[k]
                
                for k in IA1_u.keys():
                    if isinstance(k,tuple):
                        IA1_u_Array[k[0],k[1],k[2]] = IA1_u[k] 
               
                IA1_l_Reduced = IA1_l_Array[:,Previous_Potential_Winning_Components, :]
                IA1_u_Reduced = IA1_u_Array[:,Previous_Potential_Winning_Components, :]
                
                
                #CREATE SINK STATES TO REPRESENT TRANSITIONS OUTSIDE OF THESE PREVIOUS COMPONENTS
                
                IA1_l_Reduced = np.append(IA1_l_Reduced, np.zeros((IA1_l_Reduced.shape[0], IA1_l_Reduced.shape[1], 1)), 2)
                IA1_u_Reduced = np.append(IA1_u_Reduced, np.zeros((IA1_u_Reduced.shape[0], IA1_u_Reduced.shape[1], 1)), 2)
          
              
                
                for i in range(IA1_l_Reduced.shape[0]):
                    for j in range(IA1_l_Reduced.shape[1]):
                        IA1_l_Reduced[i,j,-1] = sum(np.delete(IA1_l_Reduced[i][j][:],Previous_Potential_Winning_Components))
                        IA1_u_Reduced[i,j,-1] = min(sum(np.delete(IA1_u_Reduced[i][j][:],Previous_Potential_Winning_Components)),1.0)
        
                IA1_l_Reduced = IA1_l_Reduced[:,:, list(np.append(Previous_Potential_Winning_Components, IA1_l_Reduced.shape[2]-1))]
                IA1_u_Reduced = IA1_u_Reduced[:,:, list(np.append(Previous_Potential_Winning_Components, IA1_u_Reduced.shape[2]-1))]
        
                        
                IA1_l_Reduced = np.append(IA1_l_Reduced, np.zeros((IA1_l_Reduced.shape[0], 1, IA1_l_Reduced.shape[2])), 1)
                IA1_u_Reduced = np.append(IA1_u_Reduced, np.zeros((IA1_u_Reduced.shape[0], 1, IA1_u_Reduced.shape[2])), 1)
        
        
        
                #SELECTING THE BRIDGE STATES/TRANSITIONS/ACCEPTING/REACHABLE STATES AND NON ACCEPTING STATES OF THE REDUCED PRODUCT
                Product_Reachable_States_Reduced = [[] for x in range(IA1_l_Reduced.shape[0])]
                Product_Is_Bridge_State_Reduced = np.zeros((IA1_l_Reduced.shape[0],IA1_l_Reduced.shape[1]))
                Product_Bridge_Transitions_Reduced = [[] for x in range(IA1_l_Reduced.shape[0])]
                Which_Accepting_Pair_Reduced = [[] for x in range(IA1_l_Reduced.shape[1])]
                Which_Non_Accepting_Pair_Reduced = [[] for x in range(IA1_l_Reduced.shape[1])]
                Is_Accepting_Reduced = np.zeros(IA1_l_Reduced.shape[1])
                Is_Non_Accepting_Reduced = np.zeros(IA1_l_Reduced.shape[1])
                Is_In_Permanent_Comp_Reduced = np.zeros(IA1_l_Reduced.shape[1])
                Previous_Accepting_BSCC = []
                Indices = np.zeros(IA1_l['Num_S'])
                Indices = Indices.astype(int)
    
                
                for i in range(len(Previous_Potential_Winning_Components)):
                    Indices[Previous_Potential_Winning_Components[i]] = i

                
                for i in range(IA1_l_Reduced.shape[0]):
                    IA1_l_Reduced[i][-1][-1] = 1.0
                    IA1_u_Reduced[i][-1][-1] = 1.0 #Sink State transition values
                    
                    for j in range(IA1_l_Reduced.shape[1] - 1):
                        
                        if i == 0:
                            if Is_Accepting[Previous_Potential_Winning_Components[j]] == 1:
                                Is_Accepting_Reduced[j] = 1
                                Which_Accepting_Pair_Reduced[j] = list(Which_Accepting_Pair[Previous_Potential_Winning_Components[j]])
                            if Is_Non_Accepting[Previous_Potential_Winning_Components[j]] == 1:
                                Is_Non_Accepting_Reduced[j] = 1                    
                                Which_Non_Accepting_Pair_Reduced[j] = list(Which_Non_Accepting_Pair[Previous_Potential_Winning_Components[j]])
                        
                                
                        Reach = list([])
                        Bridge = list([])
                        Differential = list(set(Product_Reachable_States[i][Previous_Potential_Winning_Components[j]]) - set(Previous_Potential_Winning_Components))
                        if len(Differential) != 0:
                            Reach.append(IA1_l_Reduced.shape[1] - 1) #Then the sink state is reachable
                            if set(Differential) == (set(Product_Bridge_Transitions[i][Previous_Potential_Winning_Components[j]]) - set(Previous_Potential_Winning_Components)):
                                Bridge.append(IA1_l_Reduced.shape[1] - 1) #If these transitions are bridges, then there is a bridge to the sink state
                        
                        List_Reach = list(set(Product_Reachable_States[i][Previous_Potential_Winning_Components[j]]).intersection(Previous_Potential_Winning_Components))
                        for k in range(len(List_Reach)):
                            Reach.append(Indices[List_Reach[k]])
    
                        List_Bridge = list(set(Product_Bridge_Transitions[i][Previous_Potential_Winning_Components[j]]).intersection(Previous_Potential_Winning_Components))    
                        for k in range(len(List_Bridge)):
                            Bridge.append(Indices[List_Bridge[k]])
                        
                        
                        Reach.sort()
                        Bridge.sort()
                        Product_Reachable_States_Reduced[i].append(Reach)
                        Product_Is_Bridge_State_Reduced[i][j] = Product_Is_Bridge_State[i][Previous_Potential_Winning_Components[j]]
                        Product_Bridge_Transitions_Reduced[i].append(Bridge)
                    
            
            

            
                first = 1;

                Optimal_Policy_Reduced = np.zeros(IA1_l_Reduced.shape[1])
                Optimal_Policy_Reduced = Optimal_Policy_Reduced.astype(int)           
                Potential_Policy_Reduced = np.zeros(IA1_l_Reduced.shape[1]) #Policy to generate the "best" best-case (maximize upper bound)
                Potential_Policy_Reduced = Potential_Policy_Reduced.astype(int)
               
                Permanent_Losing_Components = []
                Allowable_Actions_Reduced = []
                for i in range(IA1_l_Reduced.shape[1] - 1):
                      Allowable_Actions_Reduced.append(list(Allowable_Action_Potential[Previous_Potential_Winning_Components[i]]))
                
                Allowable_Action_Potential_Reduced = []
                Allowable_Action_Permanent_Reduced = []
                
                (Greatest_Potential_Accepting_BSCCs_Reduced, Greatest_Permanent_Accepting_BSCCs_Reduced, Potential_Policy_Reduced, Optimal_Policy_Reduced, Allowable_Action_Potential_Reduced, Allowable_Action_Permanent_Reduced, first, Is_In_Permanent_Comp_Reduced, List_Permanent_Accepting_BSCC_Reduced, List_Potential_Accepting_BSCC_Reduced, Which_Potential_Accepting_BSCC_Reduced, Is_In_Potential_Accepting_BSCC_Reduced, Bridge_Accepting_BSCC_Reduced) = Find_Greatest_Accepting_BSCCs(IA1_l_Reduced, IA1_u_Reduced, Is_Accepting_Reduced, Is_Non_Accepting_Reduced, Which_Accepting_Pair_Reduced, Which_Non_Accepting_Pair_Reduced, Allowable_Action_Potential_Reduced, Allowable_Action_Permanent_Reduced, first, Product_Reachable_States_Reduced, Product_Bridge_Transitions_Reduced, Product_Is_Bridge_State_Reduced, Automata_Accepting, Potential_Policy_Reduced, Optimal_Policy_Reduced, Is_In_Permanent_Comp_Reduced, [], Previous_Accepting_BSCC) # Will return greatest potential accepting bsccs
                
                Greatest_Potential_Accepting_BSCCs = []
                Greatest_Permanent_Accepting_BSCCs = []
                Is_In_Potential_Accepting_BSCC = np.zeros(IA1_l['Num_S'])
                Is_In_Potential_Accepting_BSCC = Is_In_Potential_Accepting_BSCC.astype(int)
                List_Potential_Accepting_BSCC = []
                Bridge_Accepting_BSCC = []
                Which_Potential_Accepting_BSCC = np.zeros(IA1_l['Num_S'])
                Which_Potential_Accepting_BSCC = Which_Potential_Accepting_BSCC.astype(int)
            
                # Adding permanent components from previous partitions
                for i in range(len(Greatest_Permanent_Winning_Component)):
                   Greatest_Potential_Accepting_BSCCs.append(Greatest_Permanent_Winning_Component[i])
                   Greatest_Permanent_Accepting_BSCCs.append(Greatest_Permanent_Winning_Component[i])
    

               #Converting the new components from reduced product to original indices 
                for i in range(len(Greatest_Potential_Accepting_BSCCs_Reduced)):
                   Greatest_Potential_Accepting_BSCCs.append(Previous_Potential_Winning_Components[Greatest_Potential_Accepting_BSCCs_Reduced[i]])
                   Allowable_Action_Potential[Previous_Potential_Winning_Components[Greatest_Potential_Accepting_BSCCs_Reduced[i]]] = list(Allowable_Action_Potential_Reduced[Greatest_Potential_Accepting_BSCCs_Reduced[i]])
    
                for i in range(len(Greatest_Permanent_Accepting_BSCCs_Reduced)):
                   Greatest_Permanent_Accepting_BSCCs.append(Previous_Potential_Winning_Components[Greatest_Permanent_Accepting_BSCCs_Reduced[i]])
                   Is_In_Permanent_Comp[Previous_Potential_Winning_Components[Greatest_Permanent_Accepting_BSCCs_Reduced[i]]] = 1
                   Optimal_Policy[Previous_Potential_Winning_Components[Greatest_Permanent_Accepting_BSCCs_Reduced[i]]] = Optimal_Policy_Reduced[Greatest_Permanent_Accepting_BSCCs_Reduced[i]]
                   Allowable_Actions[Previous_Potential_Winning_Components[Greatest_Permanent_Accepting_BSCCs_Reduced[i]]] = list([Optimal_Policy_Reduced[Greatest_Permanent_Accepting_BSCCs_Reduced[i]]])
               
                
                for i in range(len(List_Potential_Accepting_BSCC_Reduced)):
                   List_Potential_Accepting_BSCC.append(list([]))
                   Bridge_Accepting_BSCC.append(list([]))
                   for x in range(len(List_Potential_Accepting_BSCC_Reduced[i])):
                       Is_In_Potential_Accepting_BSCC[Previous_Potential_Winning_Components[List_Potential_Accepting_BSCC_Reduced[i][x]]] = 1
                       List_Potential_Accepting_BSCC[-1].append(Previous_Potential_Winning_Components[List_Potential_Accepting_BSCC_Reduced[i][x]])
                       Which_Potential_Accepting_BSCC[Previous_Potential_Winning_Components[List_Potential_Accepting_BSCC_Reduced[i][x]]] = i
    
                   for x in range(len(Bridge_Accepting_BSCC_Reduced[i])):
                       Bridge_Accepting_BSCC[-1].append(Previous_Potential_Winning_Components[Bridge_Accepting_BSCC_Reduced[i][x]])
               

                Optimal_Policy_Qual = np.zeros(IA1_l['Num_S'])
                Optimal_Policy_Qual = Optimal_Policy_Qual.astype(int)           
                Potential_Policy_Qual = np.zeros(IA1_l['Num_S']) #Policy to generate the "best" best-case (maximize upper bound)
                Potential_Policy_Qual = Potential_Policy_Qual.astype(int) 
               
                Allowable_Actions_Qual = []
                for i in range(len(Allowable_Actions)):
                      Allowable_Actions_Qual.append(list(Allowable_Actions[i]))
                
                Which_Potential_Accepting_BSCC = Which_Potential_Accepting_BSCC.astype(int)
                Previous_Great = copy.deepcopy(Greatest_Permanent_Winning_Component)

               
                (Greatest_Permanent_Winning_Component, Optimal_Policy_Qual, Is_In_Permanent_Comp, List_Potential_Accepting_BSCC, Is_In_Potential_Winning_Component, Bridge_Accepting_BSCC) = Find_Greatest_Winning_Components(IA1_l_Array, IA1_u_Array, Is_Accepting, Is_Non_Accepting, Which_Accepting_Pair, Which_Non_Accepting_Pair, Allowable_Actions_Qual, Product_Reachable_States, Product_Bridge_Transitions, Product_Is_Bridge_State, Automata_Accepting, Optimal_Policy_Qual, Is_In_Permanent_Comp, List_Permanent_Accepting_BSCC, List_Potential_Accepting_BSCC, Greatest_Permanent_Accepting_BSCCs, Is_In_Potential_Accepting_BSCC, Which_Potential_Accepting_BSCC, Bridge_Accepting_BSCC, len(Automata), Init, Automata_Accepting) # Will return greatest potential and permanent winning components
                List_Iteration = list(set(Greatest_Permanent_Winning_Component)-set(Previous_Great))



                for i in range(len(List_Iteration)):
                   Optimal_Policy[List_Iteration[i]] = Optimal_Policy_Qual[List_Iteration[i]]
                   Potential_Policy[List_Iteration[i]] = Optimal_Policy_Qual[List_Iteration[i]]
                   Allowable_Actions[List_Iteration[i]] = list([Optimal_Policy_Qual[List_Iteration[i]]])

                Greatest_Potential_Winning_Component = list([])
                Potential_BSCCs = [item for sublist in List_Potential_Accepting_BSCC for item in sublist]

                for i in range(len(Greatest_Permanent_Winning_Component)):
                    Greatest_Potential_Winning_Component.append(Greatest_Permanent_Winning_Component[i])
            
                for i in range(len(Potential_BSCCs)):
                    Greatest_Potential_Winning_Component.append(Potential_BSCCs[i])

                List_Potential_Winning_Components = copy.deepcopy(List_Potential_Accepting_BSCC)
                Bridge_Winning_Components = copy.deepcopy(Bridge_Accepting_BSCC)
                Bridge_Winning_Components = [[Bridge_Winning_Components[i]] for i in range(len(Bridge_Winning_Components))]
                Which_Potential_Winning_Component = [[] for i in range(IA1_l['Num_S'])]   
                for i in range(len(Which_Potential_Accepting_BSCC)):
                    if Is_In_Potential_Winning_Component[i] == 1:
                        Which_Potential_Winning_Component[i].append([Which_Potential_Accepting_BSCC[i],0])             
           
           else:
               Is_In_Potential_Winning_Component = [0]*IA1_l['Num_S']
               List_Potential_Winning_Components = []
               Greatest_Potential_Winning_Component = list(Greatest_Permanent_Winning_Component)
                                            
        
           Reach_Allowed_Actions = []
           for y in range(IA1_l['Num_S']): # For all the system states
               if Is_In_Permanent_Comp[y] == 1:
                   Reach_Allowed_Actions.append([Optimal_Policy[y]])                 
               else:
                   Reach_Allowed_Actions.append(Allowable_Actions[y])
            
           (Low_Bound, Low_Bounds_Prod, Worst_Markov_Chain, Optimal_Policy, List_Values_Low) = Maximize_Lower_Bound_Reachability(IA1_l, IA1_u, Greatest_Permanent_Winning_Component, State_Space.shape[0], len(Automata), Product_Reachable_States, Init, Optimal_Policy, Reach_Allowed_Actions, Product_Pre_States, Permanent_Losing_Components) # Maximizes Lower Bound
           (Upp_Bound, Upp_Bounds_Prod, Best_Markov_Chain, Potential_Policy, List_Values_Up) = Maximize_Upper_Bound_Reachability(IA1_l, IA1_u, Greatest_Potential_Winning_Component, State_Space.shape[0], len(Automata), Product_Reachable_States, Init, Potential_Policy, Reach_Allowed_Actions, Product_Pre_States, Permanent_Losing_Components) # Maximizes for winning component
           


        #### The code below takes care of finding the permanent and potential losing components as well as their bridge states for the purpose of refinement 
            
        
        
        
           Is_In_Potential_Losing_Component = np.zeros(IA1_l['Num_S']) #Contains a 1 if the state is in a potential losing component
           Is_In_Potential_Losing_Component = Is_In_Potential_Losing_Component.astype(int)
           Which_Losing_Component = [[] for i in range(IA1_l['Num_S'])] #Tells to you which losing component the state belongs to. For each state, there will be a list of lists [a,b], [a',b'] etc. where a is the BSCC number and b is the number of the component around that BSCC (number 0 is the BSCC by itself)
           Greatest_Permanent_Winning_Component = list([]) 
           Permanent_Losing_Components = []
           Potential_Opposite_Components = [] #Contains the potential losing components of the Worst Case Markov Chain for refinement
           Is_Pot_Comp_Bridge = []#List which tells you whether the component is a Bridge State or not
           for i in range(len(Low_Bounds_Prod)):
               if Low_Bounds_Prod[i] == 0:
                   if Upp_Bounds_Prod[i] > 0:
                       Potential_Opposite_Components.append(i)
#                       Is_In_Potential_Losing_Component[i] = 1
                       
                       if Product_Is_Bridge_State[Optimal_Policy[i]][i] == 1:
                           Is_Pot_Comp_Bridge.append(1)
                       else:
                           Is_Pot_Comp_Bridge.append(0)
                   else:
                       Is_In_Permanent_Comp[i] = 1
                       Permanent_Losing_Components.append(i)
                       Allowable_Actions[i] = list([Optimal_Policy[i]])
               elif Low_Bounds_Prod[i] == 1: #This state is a sink state that is a part of the greatest permanent winning component
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
        
           (List_Potential_Losing_Components, Which_Potential_Losing_Component, Bridge_Losing_Components) = Find_Components_And_Bridge_States(Component_Graph ,Which_Losing_Component, Potential_Opposite_Components, Is_Pot_Comp_Bridge, Absorbing_State)
        

        
        Greatest_Suboptimality_Factor = 0
        Suboptimality_Factors = []
        States_Above_Threshold = []
        Success_Intervals = [[] for n in range(len(State_Space))] #Contains the probability of satisfying the spec in the original system (under the computed policies)
        Product_Intervals = [[] for n in range(IA1_l['Num_S'])] #Contains the probability of satisfying the spec in the product (Under the computed policies)
                                
        for i in range(len(Upp_Bound)):
    
            Success_Intervals[i].append(Low_Bound[i])
            Success_Intervals[i].append(Upp_Bound[i])
        
        for i in range(len(Upp_Bounds_Prod)):
            
            Product_Intervals[i].append(Low_Bounds_Prod[i])
            Product_Intervals[i].append(Upp_Bounds_Prod[i])
        
        Sum_Num_Actions = 0
        
        for i in range(len(Allowable_Actions)):
            Delete_Actions = []
            Optimal = Optimal_Policy[i]
            Index_Optimal = Allowable_Actions[i].index(Optimal)
            Max_Sub_Fac = 0.0 #Keeps track of the suboptimality factor of a given state

            for j in range(len(Allowable_Actions[i])):
                if Allowable_Actions[i][j] != Optimal:                   
                    if List_Values_Up[i][j] <= List_Values_Low[i][Index_Optimal]:
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

        List_Max_Opt.append(Greatest_Suboptimality_Factor)    
        List_Avg_Opt.append(sum(Suboptimality_Factors)/float(len(Suboptimality_Factors)))
        Fractions_Above.append(len(States_Above_Threshold)/float(len(Suboptimality_Factors))) 
        Running_Times.append(timeit.default_timer() - start) 
        Avg_Num_Actions.append(float(Sum_Num_Actions)/float(len(Suboptimality_Factors)))
 

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
#        with open('Best_Markov_Chain.pkl','wb') as f:     
#            pickle.dump(Best_Markov_Chain, f)
#        with open('Worst_Markov_Chain.pkl','wb') as f:      
#            pickle.dump(Worst_Markov_Chain, f)
        with open('Optimal_Policy.pkl','wb') as f:      
            pickle.dump(Optimal_Policy, f)
            
        print 'Greatest Suboptimality Factor'
        print Greatest_Suboptimality_Factor
        print 'Number of Refinement Steps'
        print Tag
        print ''
        if Tag == Num_Ref_Dis:
            break
        
        
    
    return (State_Space, Low_Bound, Upp_Bound, Low_Bounds_Prod, Upp_Bounds_Prod, Potential_Policy, Optimal_Policy, Suboptimality_Factors, Init, L_mapping, Allowable_Actions, List_Max_Opt, List_Avg_Opt, Fractions_Above, Running_Times, Avg_Num_Actions)









def State_Space_Refinement_Continuous(State_Space, Threshold_Uncertainty, Greatest_Optimality_Factor, Objective, Potential_Policy_Continuous, Optimal_Policy_Continuous, Best_Markov_Chain, Worst_Markov_Chain, States_Above_Threshold, Success_Intervals, Product_Intervals, Len_Automata, Automata, Automata_Accepting, L_mapping, Is_In_Permanent_Comp, Input_Qualitative, List_Potential_Winning_Components, Which_Potential_Winning_Component, Bridge_Winning_Components, Is_In_Potential_Winning_Component, List_Potential_Losing_Components, Which_Potential_Losing_Component, Bridge_Losing_Components, Is_In_Potential_Losing_Component, Reachable_States_Cont, Pre_States, Reachable_Quantitative_Product, List_Permanent_Accepting_BSCC, Input_Quantitative_Product, Greatest_Permanent_Winning_Component, Permanent_Losing_Components, List_Max_Opt, List_Avg_Opt, Running_Times, Fraction_Above):
    
    Tag = 0 #Counting how many times we went through refinement
    V_Total = 1 #Dummy total volume, might be useful for ranking of states later 
    
    while(Greatest_Optimality_Factor > Threshold_Uncertainty):
        
        Tag += 1
        Set_Refinement = []
        start = timeit.default_timer()
        
        
        States_Above_Threshold_Product = list(States_Above_Threshold)
        
        Is_in_WC_P = np.zeros(Best_Markov_Chain.shape[1])
        Is_in_LC_P = np.zeros(Best_Markov_Chain.shape[1])  
                

        Weights = Uncertainty_Ranking_Path_New3(Best_Markov_Chain, Worst_Markov_Chain, States_Above_Threshold_Product, Product_Intervals, 0, 0, Is_In_Potential_Winning_Component, Which_Potential_Winning_Component, Bridge_Winning_Components, Is_In_Potential_Losing_Component, Which_Potential_Losing_Component, Bridge_Losing_Components, Is_in_WC_P, [], [], Is_in_LC_P, [], [], Is_In_Permanent_Comp, State_Space, V_Total, Len_Automata)
    
        # The lines below convert uncertainty in the product IMC into uncertainty in the original system

        Weights_Original_S = [0.0 for n in range(State_Space.shape[0])]     
        for j in range(len(Weights)):
            Weights_Original_S[j/Len_Automata] += Weights[j]            
        Uncertainty_Ranking_Original_S = list(reversed(sorted(range(len(Weights_Original_S)), key=lambda x: Weights_Original_S[x])))        
        Weights_Original_S = list(reversed(sorted(Weights_Original_S)))

        Index_Stop = -1   
        Cut_off_Percentage = 0.01
        for j in range(len(Weights_Original_S)):
            if (Weights_Original_S[j]/Weights_Original_S[0]) < Cut_off_Percentage:
                Index_Stop = j
                break
            if Weights_Original_S[j] <= 0.0:
                Index_Stop = j
                break

        if Index_Stop == -1: Index_Stop = len(Weights_Original_S)        
        Index_Stop1 = (State_Space.shape[0]) #Parameter that sets the maximum number of states to be refined       
        Num_States = min(len(Uncertainty_Ranking_Original_S), Index_Stop, Index_Stop1)    

        if Index_Stop == -1: Index_Stop = len(Weights_Original_S)        
        Index_Stop1 = (State_Space.shape[0]) #Parameter that sets the maximum number of states to be refined       
        Num_States = min(len(Uncertainty_Ranking_Original_S), Index_Stop, Index_Stop1)

        for j in range(Num_States):       
            Set_Refinement.append(Uncertainty_Ranking_Original_S[j])

        Set_Refinement.sort()   

        New_States = Raw_Refinement(Set_Refinement, State_Space)
        


        Previous_Potential_Winning_Components = [item for sublist in List_Potential_Winning_Components for item in sublist]
        Previous_Potential_Winning_Components.sort()
        
        #NEED TO UPDATE INPUT QUALITATIVE PRODUCT
        

        
        """ Deleting previous state and replacing it with new states, as well as updating index for some lists """

        First_Loop = 1
        
        Index_Stop = np.zeros(Best_Markov_Chain.shape[0]+len(Set_Refinement))
        Index_Stop = Index_Stop.astype(int) 
        Index_Stop7 = 0
        Index_Stop8 = 0
        Index_Stop9 = 0
        Copy_Greatest_Permanent_Winning_Component = list(Greatest_Permanent_Winning_Component)


        Previous_Reach_Cont = copy.deepcopy(Reachable_States_Cont)

        Copy_Permanent_Losing_Components = list(Permanent_Losing_Components)

        
        Copy_Previous_Potential_Winning_Components = list(Previous_Potential_Winning_Components)

        for m, Set in enumerate(Set_Refinement):
                       

            State_Space = np.concatenate((State_Space[:Set+1+m], np.asarray([New_States[2*m]]), State_Space[Set+1+m:]))                        
            State_Space = np.concatenate((State_Space[:Set+1+m], np.asarray([New_States[2*m+1]]), State_Space[Set+1+m:]))                        
            State_Space = np.delete(State_Space, Set+m, 0)                                     
            
            for z in range(Len_Automata):
                Input_Quantitative_Product.insert(Set*Len_Automata+Len_Automata+z+m*Len_Automata, list(Input_Quantitative_Product[Set*Len_Automata+m*Len_Automata+z]))
                Optimal_Policy_Continuous.insert(Set*Len_Automata+Len_Automata+z+m*Len_Automata, list(Optimal_Policy_Continuous[Set*Len_Automata+m*Len_Automata+z]))
                Potential_Policy_Continuous.insert(Set*Len_Automata+Len_Automata+z+m*Len_Automata, list(Potential_Policy_Continuous[Set*Len_Automata+m*Len_Automata+z]))
                Is_In_Permanent_Comp = np.insert(Is_In_Permanent_Comp, Set*Len_Automata+Len_Automata+z+m*Len_Automata, Is_In_Permanent_Comp[Set*Len_Automata+m*Len_Automata+z])


            L_mapping.insert(Set+m+1, L_mapping[Set+m])
            
            (First_Loop, Reachable_States_Cont, Set_Refinement, Index_Stop, m) = Index_Update_Continuous(First_Loop, Reachable_States_Cont, Set_Refinement, Index_Stop, m)

            for y in range(len(Previous_Reach_Cont)):
                index = 0
                for x in range(len(Previous_Reach_Cont[y])):
                    if Previous_Reach_Cont[y][x+index] < Set+m:
                        continue
                    elif Previous_Reach_Cont[y][x+index] == Set+m:
                        Previous_Reach_Cont[y].insert(x+index+1, Set+1+m)
                        index += 1
                    else: 
                        Previous_Reach_Cont[y][x+index] += 1

            (Previous_Potential_Winning_Components, Copy_Previous_Potential_Winning_Components, Index_Stop7) = Index_Update_Product_Continuous(Previous_Potential_Winning_Components, Copy_Previous_Potential_Winning_Components, Set_Refinement, Index_Stop7, Len_Automata, m)
            (Greatest_Permanent_Winning_Component, Copy_Greatest_Permanent_Winning_Component, Index_Stop8) = Index_Update_Product_Continuous(Greatest_Permanent_Winning_Component, Copy_Greatest_Permanent_Winning_Component, Set_Refinement, Index_Stop8, Len_Automata, m)
            (Permanent_Losing_Components, Copy_Permanent_Losing_Components, Index_Stop9) = Index_Update_Product_Continuous(Permanent_Losing_Components, Copy_Permanent_Losing_Components, Set_Refinement, Index_Stop9, Len_Automata, m)

           


        """ _______ """
        

        
        for m in range(len(Set_Refinement)):
            Previous_Reach_Cont.insert(Set_Refinement[m]+1+m, list(Previous_Reach_Cont[Set_Refinement[m]+m]))

        
        
        Potential_Winning_Components_Original = [] #Tells you which states in the original transition system can generate an accepting BSCC in the product

        for i in range(len(Previous_Potential_Winning_Components)):
            if(Previous_Potential_Winning_Components[i]/Len_Automata) not in Potential_Winning_Components_Original:
                Potential_Winning_Components_Original.append(Previous_Potential_Winning_Components[i]/Len_Automata)


        Reachable_States_Quali = [] #Contains the reachable states of all states which can form an accepting BSCC . All those reachable states are enforced to be able to produce an accepting BSCC as well
        
        for i in range(len(Potential_Winning_Components_Original)):
            Intersect = set(Reachable_States_Cont[Potential_Winning_Components_Original[i]]).intersection(Potential_Winning_Components_Original)
            Reachable_States_Quali.append(list(Intersect))


        Is_New_State = np.zeros(State_Space.shape[0])
        Set_Refinement2 = []
               
        for j in range(len(Set_Refinement)):
            Is_New_State[Set_Refinement[j] + j] = 1
            Is_New_State[Set_Refinement[j] + j + 1] = 1
            Set_Refinement2.append(Set_Refinement[j]+j+1)
            Set_Refinement[j] = Set_Refinement[j] + j

        Inputs = np.zeros((1,State_Space.shape[0]))
        Reachable_Sets = Reachable_Sets_Computation_Finite(State_Space, Inputs)
        Reachable_Sets = list(Reachable_Sets[0])
        
        
        Discrete_Actions = list([])
        Input_Qualitative = [[] for x in range(State_Space.shape[0])]
        #We compute the new set of qualitative inputs by taking the union of all inputs of corresponding states in the product 
        for i in range(len(Input_Qualitative)):
            list_polygons = []
            for k in range(Len_Automata*i, Len_Automata*i+Len_Automata):
                for x in range(len(Input_Quantitative_Product[k])):
                    Polygon_To_Add = Polygon([(Input_Quantitative_Product[k][x][0][0], Input_Quantitative_Product[k][x][0][1]), (Input_Quantitative_Product[k][x][0][0], Input_Quantitative_Product[k][x][1][1]), (Input_Quantitative_Product[k][x][1][0], Input_Quantitative_Product[k][x][1][1]), (Input_Quantitative_Product[k][x][1][0], Input_Quantitative_Product[k][x][0][1])])             

                    list_polygons.append(Polygon_To_Add)
        
            Union = cascaded_union(list_polygons) 
            
            if Union.geom_type == 'MultiPolygon':
                        
                for pol in Union:  # same for multipolygon.geoms
                            
                    list_r = list([[]])
                    poly_r = shapely.geometry.polygon.orient(pol) 
                            
                    for y in range(len(poly_r.exterior.coords)):
                        list_r[0].append(list(poly_r.exterior.coords[y]))
                                    
                                    
                    runtime = get()
                    ctx = runtime.compile('''
                            module.paths.push('%s');
                            var decompose = require('rectangle-decomposition'); 
                            function decompose_region(region){
                            var rectangles = decompose(region)
                            return rectangles;
                            }
                                        
                    ''' % os.path.join(os.path.dirname(__file__),'node_modules'))
                    Input_Qualitative[i] = list(ctx.call("decompose_region",list_r))
        
                                    

                        #PARSE RESPONSE AND STORE IN NEW_OVERLAPS
                        
                            
                        
            elif Union.geom_type == 'Polygon':    
                        
                 list_r = list([[]])
                                
                 Union = shapely.geometry.polygon.orient(Union)
                                    
                 for y in range(len(Union.exterior.coords)):
                     list_r[0].append(list(Union.exterior.coords[y])) #NEED TO STORE VERTICES IN COUNTERCLOCKWISE DIRECTION
                                
                 runtime = get()
                 ctx = runtime.compile('''
                     module.paths.push('%s');
                     var decompose = require('rectangle-decomposition'); 
                     function decompose_region(region){
                     var rectangles = decompose(region)
                     return rectangles;
                     }
                    
                 ''' % os.path.join(os.path.dirname(__file__),'node_modules'))
                
                 Input_Qualitative[i] = list(ctx.call("decompose_region",list_r))

        
        res = Parallel(n_jobs=4)(delayed(Select_Actions)(Input_Qualitative[i], Reachable_Sets[i], State_Space, list(Reachable_States_Cont[i]), Previous_Potential_Winning_Components, Is_New_State[i], Previous_Reach_Cont[i], Potential_Winning_Components_Original, i) for i in range(State_Space.shape[0]))
             
        for j in range(len(res)):
            Discrete_Actions.append(list(res[j][0]))
            Reachable_States_Cont[j] = list(res[j][1])
            
        
        Max_Num_Actions = len(max(Discrete_Actions,key=len)) #HAVE TO DO THIS BECAUSE CURRENT IMPLEMENTATION ASSUMES ALL STATES HAVE SAME NUMBER OF ACTIONS
        Reachable_Sets_Quanti, List_States_Per_Action = Reachable_Sets_Computation_Continuous(State_Space, Discrete_Actions, Max_Num_Actions);          
        (Lower_Bound_Matrix, Upper_Bound_Matrix, Reachable_States, Is_Bridge_State, Bridge_Transitions, Pre_States) = BMDP_Computation_Continuous(Reachable_Sets_Quanti, State_Space, Max_Num_Actions, List_States_Per_Action) 
        (IA1_l, IA1_u, Is_Accepting, Is_Non_Accepting, Which_Accepting_Pair, Which_Non_Accepting_Pair, Product_Reachable_States, Product_Is_Bridge_State, Product_Bridge_Transitions, Init) = Build_Product_BMDP_Continuous(Lower_Bound_Matrix, Upper_Bound_Matrix, Automata, L_mapping, Automata_Accepting, Reachable_States, Is_Bridge_State, Bridge_Transitions, Discrete_Actions)
        
         
        Objective_Prob = 0
        
        #SELECT ONLY STATES THAT WERE PREVIOUSLY IN A POTENTIAL COMPONENT
        if len(Previous_Potential_Winning_Components) != 0:
            
            IA1_l_Array = np.zeros(((IA1_l['Num_Act']),IA1_l['Num_S'],IA1_l['Num_S']))
            IA1_u_Array = np.zeros(((IA1_u['Num_Act']),IA1_u['Num_S'],IA1_u['Num_S']))
                   
            for k in IA1_l.keys():
                if isinstance(k,tuple):
                    IA1_l_Array[k[0],k[1],k[2]] = IA1_l[k]
                
            for k in IA1_u.keys():
                if isinstance(k,tuple):
                    IA1_u_Array[k[0],k[1],k[2]] = IA1_u[k]
        
            IA1_l_Reduced = IA1_l_Array[:,Previous_Potential_Winning_Components, :]
            IA1_u_Reduced = IA1_u_Array[:,Previous_Potential_Winning_Components, :]

#            IA1_l_Reduced = IA1_l[:,Previous_Potential_Winning_Components, :]
#            IA1_u_Reduced = IA1_u[:,Previous_Potential_Winning_Components, :]            
            
            #CREATE SINK STATES TO REPRESENT TRANSITIONS OUTSIDE OF THESE PREVIOUS COMPONENTS
            
            IA1_l_Reduced = np.append(IA1_l_Reduced, np.zeros((IA1_l_Reduced.shape[0], IA1_l_Reduced.shape[1], 1)), 2)
            IA1_u_Reduced = np.append(IA1_u_Reduced, np.zeros((IA1_u_Reduced.shape[0], IA1_u_Reduced.shape[1], 1)), 2)
      
          
            
            for i in range(IA1_l_Reduced.shape[0]):
                for j in range(IA1_l_Reduced.shape[1]):
                    IA1_l_Reduced[i,j,-1] = sum(np.delete(IA1_l_Reduced[i][j][:],Previous_Potential_Winning_Components))
                    IA1_u_Reduced[i,j,-1] = min(sum(np.delete(IA1_u_Reduced[i][j][:],Previous_Potential_Winning_Components)),1.0)
    
            IA1_l_Reduced = IA1_l_Reduced[:,:, list(np.append(Previous_Potential_Winning_Components, IA1_l_Reduced.shape[2]-1))]
            IA1_u_Reduced = IA1_u_Reduced[:,:, list(np.append(Previous_Potential_Winning_Components, IA1_u_Reduced.shape[2]-1))]
    
                    
            IA1_l_Reduced = np.append(IA1_l_Reduced, np.zeros((IA1_l_Reduced.shape[0], 1, IA1_l_Reduced.shape[2])), 1)
            IA1_u_Reduced = np.append(IA1_u_Reduced, np.zeros((IA1_u_Reduced.shape[0], 1, IA1_u_Reduced.shape[2])), 1)
    
    
    
            #SELECTING THE BRIDGE STATES/TRANSITIONS/ACCEPTING/REACHABLE STATES AND NON ACCEPTING STATES OF THE REDUCED PRODUCT
            Product_Reachable_States_Reduced = [[] for x in range(IA1_l_Reduced.shape[0])]
            Product_Is_Bridge_State_Reduced = np.zeros((IA1_l_Reduced.shape[0],IA1_l_Reduced.shape[1]))
            Product_Bridge_Transitions_Reduced = [[] for x in range(IA1_l_Reduced.shape[0])]
            Which_Accepting_Pair_Reduced = [[] for x in range(IA1_l_Reduced.shape[1])]
            Which_Non_Accepting_Pair_Reduced = [[] for x in range(IA1_l_Reduced.shape[1])]
            Is_Accepting_Reduced = np.zeros(IA1_l_Reduced.shape[1])
            Is_Non_Accepting_Reduced = np.zeros(IA1_l_Reduced.shape[1])
            Is_In_Permanent_Comp_Reduced = np.zeros(IA1_l_Reduced.shape[1])
            Previous_Accepting_BSCC = []
            Indices = np.zeros(Len_Automata*len(State_Space))
            Indices = Indices.astype(int)

            
            for i in range(len(Previous_Potential_Winning_Components)):
                Indices[Previous_Potential_Winning_Components[i]] = i
            
            for i in range(IA1_l_Reduced.shape[0]):
                IA1_l_Reduced[i][-1][-1] = 1.0
                IA1_u_Reduced[i][-1][-1] = 1.0 #Sink State transition values
                
                for j in range(IA1_l_Reduced.shape[1] - 1):
                    
                    if i == 0:
                        if Is_Accepting[Previous_Potential_Winning_Components[j]] == 1:
                            Is_Accepting_Reduced[j] = 1
                            Which_Accepting_Pair_Reduced[j] = list(Which_Accepting_Pair[Previous_Potential_Winning_Components[j]])
                        if Is_Non_Accepting[Previous_Potential_Winning_Components[j]] == 1:
                            Is_Non_Accepting_Reduced[j] = 1                    
                            Which_Non_Accepting_Pair_Reduced[j] = list(Which_Non_Accepting_Pair[Previous_Potential_Winning_Components[j]])
                    
                            
                    Reach = list([])
                    Bridge = list([])
                    Differential = list(set(Product_Reachable_States[i][Previous_Potential_Winning_Components[j]]) - set(Previous_Potential_Winning_Components))
                    if len(Differential) != 0:
                        Reach.append(IA1_l_Reduced.shape[1] - 1) #Then the sink state is reachable
                        if set(Differential) == (set(Product_Bridge_Transitions[i][Previous_Potential_Winning_Components[j]]) - set(Previous_Potential_Winning_Components)):
                            Bridge.append(IA1_l_Reduced.shape[1] - 1) #If these transitions are bridges, then there is a bridge to the sink state
                    
                    List_Reach = list(set(Product_Reachable_States[i][Previous_Potential_Winning_Components[j]]).intersection(Previous_Potential_Winning_Components))
                    for k in range(len(List_Reach)):
                        Reach.append(Indices[List_Reach[k]])

                    List_Bridge = list(set(Product_Bridge_Transitions[i][Previous_Potential_Winning_Components[j]]).intersection(Previous_Potential_Winning_Components))    
                    for k in range(len(List_Bridge)):
                        Bridge.append(Indices[List_Bridge[k]])
                    
                    
                    Reach.sort()
                    Bridge.sort()
                    Product_Reachable_States_Reduced[i].append(Reach)
                    Product_Is_Bridge_State_Reduced[i][j] = Product_Is_Bridge_State[i][Previous_Potential_Winning_Components[j]]
                    Product_Bridge_Transitions_Reduced[i].append(Bridge)
                
            
            
            
            
            first = 1;
            Allowable_Action_Potential = list([]) #Actions that could make the state a potential BSCC
            Allowable_Action_Permanent = list([]) #Actions that could make the state a permanent BSCC        
            


        if Objective_Prob == 0:
            
           if len(Previous_Potential_Winning_Components) != 0:
               
               IA1_l_Array = np.zeros(((IA1_l['Num_Act']),IA1_l['Num_S'],IA1_l['Num_S']))
               IA1_u_Array = np.zeros(((IA1_u['Num_Act']),IA1_u['Num_S'],IA1_u['Num_S']))
                       
               for k in IA1_l.keys():
                   if isinstance(k,tuple):
                       IA1_l_Array[k[0],k[1],k[2]] = IA1_l[k]
                    
               for k in IA1_u.keys():
                    if isinstance(k,tuple):
                        IA1_u_Array[k[0],k[1],k[2]] = IA1_u[k]
               
                
               IA1_l = np.array(IA1_l_Array)
               IA1_u = np.array(IA1_u_Array)
                         
               Optimal_Policy = np.zeros(Len_Automata*len(State_Space))
               Optimal_Policy = Optimal_Policy.astype(int)           
               Potential_Policy = np.zeros(Len_Automata*len(State_Space)) #Policy to generate the "best" best-case (maximize upper bound)
               Potential_Policy = Potential_Policy.astype(int)
               
               Allowable_Actions = []
               for i in range(len(Discrete_Actions)):
                  for k in range(len(Automata)):
                      Allowable_Actions.append(range(len(Discrete_Actions[i])))
               
               (Greatest_Potential_Accepting_BSCCs_Reduced, Greatest_Permanent_Accepting_BSCCs_Reduced, Potential_Policy, Optimal_Policy, Allowable_Action_Potential, Allowable_Action_Permanent, first, Is_In_Permanent_Comp_Reduced, List_Permanent_Accepting_BSCC_Reduced, List_Potential_Accepting_BSCC_Reduced, Which_Potential_Accepting_BSCC_Reduced, Is_In_Potential_Accepting_BSCC_Reduced, Bridge_Accepting_BSCC_Reduced) = Find_Greatest_Accepting_BSCCs(IA1_l_Reduced, IA1_u_Reduced, Is_Accepting_Reduced, Is_Non_Accepting_Reduced, Which_Accepting_Pair_Reduced, Which_Non_Accepting_Pair_Reduced, Allowable_Action_Potential, Allowable_Action_Permanent, first, Product_Reachable_States_Reduced, Product_Bridge_Transitions_Reduced, Product_Is_Bridge_State_Reduced, Automata_Accepting, Potential_Policy, Optimal_Policy, Is_In_Permanent_Comp_Reduced, [], Previous_Accepting_BSCC) # Will return greatest potential accepting bsccs



           Greatest_Potential_Accepting_BSCCs = []
           Greatest_Permanent_Accepting_BSCCs = []
           Is_In_Potential_Accepting_BSCC = np.zeros(Len_Automata*len(State_Space))
           Is_In_Potential_Accepting_BSCC = Is_In_Potential_Accepting_BSCC.astype(int)
           List_Potential_Accepting_BSCC = []
           Bridge_Accepting_BSCC = []
           #Which_Potential_Accepting_BSCC = [[] for x in range(IA1_l.shape[1])]
           Which_Potential_Accepting_BSCC = np.zeros(Len_Automata*len(State_Space))
           Which_Potential_Accepting_BSCC = Which_Potential_Accepting_BSCC.astype(int)
            
           # Adding permanent components from previous partitions
           for i in range(len(Greatest_Permanent_Winning_Component)):
               Greatest_Potential_Accepting_BSCCs.append(Greatest_Permanent_Winning_Component[i])
               Greatest_Permanent_Accepting_BSCCs.append(Greatest_Permanent_Winning_Component[i])

           if len(Previous_Potential_Winning_Components) != 0:

               #Converting the new components from reduced product to original indices 
               for i in range(len(Greatest_Potential_Accepting_BSCCs_Reduced)):
                   Greatest_Potential_Accepting_BSCCs.append(Previous_Potential_Winning_Components[Greatest_Potential_Accepting_BSCCs_Reduced[i]])
    
               for i in range(len(Greatest_Permanent_Accepting_BSCCs_Reduced)):
                   Greatest_Permanent_Accepting_BSCCs.append(Previous_Potential_Winning_Components[Greatest_Permanent_Accepting_BSCCs_Reduced[i]])
                   Is_In_Permanent_Comp[Previous_Potential_Winning_Components[Greatest_Permanent_Accepting_BSCCs_Reduced[i]]] = 1
                   Optimal_Policy_Continuous[Previous_Potential_Winning_Components[Greatest_Permanent_Accepting_BSCCs_Reduced[i]]] = list(Discrete_Actions[Greatest_Permanent_Accepting_BSCCs_Reduced[i]/len(Automata)][Optimal_Policy[Greatest_Permanent_Accepting_BSCCs_Reduced[i]]])

               
               for i in range(len(List_Potential_Accepting_BSCC_Reduced)):
                   List_Potential_Accepting_BSCC.append(list([]))
                   Bridge_Accepting_BSCC.append(list([]))
                   for x in range(len(List_Potential_Accepting_BSCC_Reduced[i])):
                       Is_In_Potential_Accepting_BSCC[Previous_Potential_Winning_Components[List_Potential_Accepting_BSCC_Reduced[i][x]]] = 1
                       List_Potential_Accepting_BSCC[-1].append(Previous_Potential_Winning_Components[List_Potential_Accepting_BSCC_Reduced[i][x]])
                       Which_Potential_Accepting_BSCC[Previous_Potential_Winning_Components[List_Potential_Accepting_BSCC_Reduced[i][x]]] = i
    
                   for x in range(len(Bridge_Accepting_BSCC_Reduced[i])):
                       Bridge_Accepting_BSCC[-1].append(Previous_Potential_Winning_Components[Bridge_Accepting_BSCC_Reduced[i][x]])
               
               Optimal_Policy = np.zeros(Len_Automata*len(State_Space))
               Optimal_Policy = Optimal_Policy.astype(int)           
               Potential_Policy = np.zeros(Len_Automata*len(State_Space)) #Policy to generate the "best" best-case (maximize upper bound)
               Potential_Policy = Potential_Policy.astype(int) 
               
               Allowable_Actions = []
               for i in range(len(Discrete_Actions)):
                  for k in range(len(Automata)):
                      Allowable_Actions.append(range(len(Discrete_Actions[i])))
                
               Which_Potential_Accepting_BSCC = Which_Potential_Accepting_BSCC.astype(int)
               Previous_Great = copy.deepcopy(Greatest_Permanent_Winning_Component)
               
               

               
               (Greatest_Permanent_Winning_Component, Optimal_Policy, Is_In_Permanent_Comp, List_Potential_Accepting_BSCC, Is_In_Potential_Winning_Component, Bridge_Accepting_BSCC) = Find_Greatest_Winning_Components(IA1_l, IA1_u, Is_Accepting, Is_Non_Accepting, Which_Accepting_Pair, Which_Non_Accepting_Pair, Allowable_Actions, Product_Reachable_States, Product_Bridge_Transitions, Product_Is_Bridge_State, Automata_Accepting, Optimal_Policy, Is_In_Permanent_Comp, List_Permanent_Accepting_BSCC, List_Potential_Accepting_BSCC, Greatest_Permanent_Accepting_BSCCs, Is_In_Potential_Accepting_BSCC, Which_Potential_Accepting_BSCC, Bridge_Accepting_BSCC, len(Automata), Init, Automata_Accepting) # Will return greatest potential and permanent winning components
               List_Iteration = list(set(Greatest_Permanent_Winning_Component)-set(Previous_Great))


                               

               for i in range(len(List_Iteration)):
                   Optimal_Policy_Continuous[List_Iteration[i]] = list(Discrete_Actions[List_Iteration[i]/len(Automata)][Optimal_Policy[List_Iteration[i]]])

               Greatest_Potential_Winning_Component = list([])
               Potential_BSCCs = [item for sublist in List_Potential_Accepting_BSCC for item in sublist]

               for i in range(len(Greatest_Permanent_Winning_Component)):
                   Greatest_Potential_Winning_Component.append(Greatest_Permanent_Winning_Component[i])
            
               for i in range(len(Potential_BSCCs)):
                   Greatest_Potential_Winning_Component.append(Potential_BSCCs[i])
                                   
    
               Allowable_Action_Permanent = copy.deepcopy(Allowable_Action_Potential) #Some potential actions could be permanent actions for certain states under refinement 
          
            
           
           if len(Previous_Potential_Winning_Components) != 0:
               List_Potential_Winning_Components = copy.deepcopy(List_Potential_Accepting_BSCC)
               Bridge_Winning_Components = copy.deepcopy(Bridge_Accepting_BSCC)
               Bridge_Winning_Components = [[Bridge_Winning_Components[i]] for i in range(len(Bridge_Winning_Components))]
               Which_Potential_Winning_Component = [[] for i in range(Len_Automata*len(State_Space))]   
               for i in range(len(Which_Potential_Accepting_BSCC)):
                   if Is_In_Potential_Winning_Component[i] == 1:
                       Which_Potential_Winning_Component[i].append([Which_Potential_Accepting_BSCC[i],0])             
           
           else:
               Is_In_Potential_Winning_Component = [0]*(Len_Automata*len(State_Space))
               List_Potential_Winning_Components = []
               Greatest_Potential_Winning_Component = list(Greatest_Permanent_Winning_Component)
               
        
           State_Space_Extended = [] #WE HAVE TO MAKE THE BORDER VIRTUALLY LARGER TO ACCOUNT FOR CONTAINEDNESS
           Reachable_Quantitative_Product = [[] for x in range(State_Space.shape[0])*len(Automata)] #Contains the reachable states in the product automaton
           Pre_Quantitative_Product = [set([]) for x in range(State_Space.shape[0])*len(Automata)] #Contains the pre states in the product automaton   
           
           # We create the extended State Space to account for the border extension , and compute the reachable states in the product
           
           for i in range(State_Space.shape[0]):
               
               Copy_list_state = copy.deepcopy(State_Space[i])
               State_Space_Extended.append(list(Copy_list_state))
               
               if State_Space_Extended[-1][0][0] == LOW_1:
                   State_Space_Extended[-1][0][0] = min(LOW_1 + U_MIN_1 + mus[0] - Semi_Width_1, State_Space_Extended[-1][0][0])
               
               elif State_Space_Extended[-1][1][0] == UP_1:
                   State_Space_Extended[-1][1][0] = max(UP_1 + U_MAX_1 + mus[0] + Semi_Width_1, State_Space_Extended[-1][1][0])
                   
               if State_Space_Extended[-1][0][1] == LOW_2:
                   State_Space_Extended[-1][0][1] = min(LOW_2 + U_MIN_2 + mus[1] - Semi_Width_2, State_Space_Extended[-1][0][1])
                         
               elif State_Space_Extended[-1][1][1] == UP_2:  
                   State_Space_Extended[-1][1][1] = max(UP_2 + U_MAX_2 + mus[1] + Semi_Width_2, State_Space_Extended[-1][1][1]) 
                      
                   
           for j in range(State_Space.shape[0]):
               for k in range(len(Reachable_States_Cont[j])):
                   for x in range(len(Automata)):
                       for y in range(len(Automata)):
                           if L_mapping[Reachable_States_Cont[j][k]] in Automata[x][y]:
                               Reachable_Quantitative_Product[j*len(Automata)+x].append(Reachable_States_Cont[j][k]*len(Automata)+y)
                               Pre_Quantitative_Product[Reachable_States_Cont[j][k]*len(Automata)+y].add(j*len(Automata)+x)
                               break
           
                 
           (Low_Bound, Low_Bounds_Prod, Worst_Markov_Chain, Optimal_Policy_Continuous) = Maximize_Lower_Bound_Reachability_Continuous(Greatest_Permanent_Winning_Component, State_Space_Extended, State_Space.shape[0], len(Automata), Reachable_Quantitative_Product, Init, Optimal_Policy_Continuous, Input_Quantitative_Product, Reachable_Sets, Pre_Quantitative_Product, Permanent_Losing_Components) # Maximizes Lower Bound
            
            #The function below takes care of computing the optimality factor as well as removing the suboptimal sets of inputs
            
           (Upp_Bound, Upp_Bounds_Prod, Best_Markov_Chain, Potential_Policy_Continuous, List_Optimality_Factors, Input_Quantitative_Product) = Maximize_Upper_Bound_Reachability_Continuous(Greatest_Potential_Winning_Component, State_Space_Extended, State_Space.shape[0], len(Automata), Reachable_Quantitative_Product, Init, Potential_Policy_Continuous, Input_Quantitative_Product, Reachable_Sets, Low_Bounds_Prod, Pre_Quantitative_Product, Permanent_Losing_Components) # Maximizes Lower Bound
            
           States_Above_Threshold = list([])
           Greatest_Optimality_Factor = 0.0
           for j in range(len(List_Optimality_Factors)):
             Greatest_Optimality_Factor = max(Greatest_Optimality_Factor, List_Optimality_Factors[j])
             if List_Optimality_Factors[j] > Threshold_Uncertainty:
                 States_Above_Threshold.append(j)
           Fraction_Above.append(float(len(States_Above_Threshold))/float(len(List_Optimality_Factors)))
      
           
           
           List_Max_Opt.append(Greatest_Optimality_Factor)
           List_Avg_Opt.append(sum(List_Optimality_Factors)/(len(List_Optimality_Factors)))
           Running_Times.append(timeit.default_timer() - start)
                      
           Is_In_Potential_Losing_Component = np.zeros(len(List_Optimality_Factors)) #Contains a 1 if the state is in a potential losing component
           Is_In_Potential_Losing_Component = Is_In_Potential_Losing_Component.astype(int)
            
           if Greatest_Optimality_Factor > Threshold_Uncertainty and Tag < Num_Ref:
                    #BELOW, WE FIRST COMPUTE ALL PERMANENT AND POTENTIAL LOSING COMPONENTS OF THE WORST CASE MARKOV CHAIN FOR THE PURPOSE OF REFINEMENT
                Greatest_Permanent_Winning_Component = list([]) 
                Potential_Losing = list([])
                Permanent_Losing = list([])
                States_To_Delete = list([])
                Permanent_Losing_Components = []

                Indices = list([]) #Keeps track of the indices of the states in the original Markov Chain
                for i in range(len(Low_Bounds_Prod)):
                    if Low_Bounds_Prod[i] == 0:  #If the upper_bound of a state is non-zero and a lower bound zero, then the state belongs a potential losing component
                        if Upp_Bounds_Prod[i] > 0:   
                            Potential_Losing.append(i)
#                            Is_In_Potential_Losing_Component[i] = 1
                            Indices.append(i)
                        else:
                            Permanent_Losing.append(i)
                            Permanent_Losing_Components.append(i)
                            Is_In_Permanent_Comp[i] = 1
                            Indices.append(i) 
                            #States_To_Delete.append(i)
                            
                    else:
                        if Low_Bounds_Prod[i] == 1: #This state is a sink state that is a part of the greatest permanent winning component
                            Is_In_Permanent_Comp[i] = 1
                            Greatest_Permanent_Winning_Component.append(i)
                        States_To_Delete.append(i)

                if Tag <= Lim_Qualitative_Ref: #When the state space is already very refined, we stop focusing on potential  BSCCs and Sink states, because expensive to compute in current version of implementation
                        

                    Worst_Reduced1 = np.delete(Worst_Markov_Chain.todense(),States_To_Delete,axis=0)
                    Worst_Reduced = np.delete(np.array(Worst_Reduced1),States_To_Delete,axis=1)
                    Worst_Reduced = (Worst_Reduced > 0).astype(int)
                        
                    C,n = SSCC(Worst_Reduced)
                    List_Potential_Losing_Components = list([])
                    Bridge_Losing_Components = list([])
                    Which_Potential_Losing_Component = list([[] for x in range(Worst_Markov_Chain.shape[0])])
                        
                    Number_BSCCs = 0
                    
                
                
                for j in range(len(C)):
                    BSCC = 1
                    
                    if Tag > Lim_Qualitative_Ref: #When the state space is already very refined, we stop focusing on potential  BSCCs and Sink states, because expensive to compute in current version of implementation
                        continue
                    #We only care about sink states towards potential BSCCs in these examples                

                    All_Permanent = 1
                    for k in range(len(C[j])):  
                        if Is_In_Permanent_Comp[Indices[C[j][k]]] == 0:
                            All_Permanent = 0
                        if sum(Worst_Reduced[C[j][k],C[j]]) < sum(Worst_Reduced[C[j][k],:]): #This means, if the state in the SCC leaks
                            BSCC = 0
                            break
                    
                    if All_Permanent == 1:
                        continue
                    
                    if BSCC == 1:
                        if Is_In_Potential_Losing_Component[Indices[C[j][0]]] == 1: #We only consider the  potential losing BSCCs for now, not the sink states towards permanent BSCCs
                            List_Potential_Losing_Components.append(list([]))
                           
                            Bridge_Losing_Components.append(list([]))
                            Bridge_Losing_Components[-1].append(list([]))
                    
                            for i in range(len(C[j])):
                                List_Potential_Losing_Components[-1].append(Indices[C[j][i]])
                                Which_Potential_Losing_Component[Indices[C[j][i]]].append([Number_BSCCs,0])
                                if Is_In_Potential_Winning_Component[Indices[C[j][i]]] == 1:
                                    Bridge_Losing_Components[-1][0].append(Indices[C[j][i]])
                                else:    
                                    for k in range(len(Reachable_Quantitative_Product[Indices[C[j][i]]])):
                                        if Worst_Markov_Chain[Indices[C[j][i]]][Reachable_Quantitative_Product[Indices[C[j][i]]][k]] == 0 and Best_Markov_Chain[Indices[C[j][i]]][Reachable_Quantitative_Product[Indices[C[j][i]]][k]] > 0: #Comparing the worst and best case Markov Chains to figure out the bridge states
                                            Bridge_Losing_Components[-1][0].append(Indices[C[j][i]])
                    
                            
                            C.sort()
                            
                       
                            Gr = igraph.Graph.Adjacency(Worst_Reduced.tolist())
                            R = set([]) #Contains all sink states w.r.t current BSCC
                            for q in range(len(C[j])):                 
                                R = R.union(set(Gr.subcomponent(C[j][q], mode="IN")) - set(C[j]))
                                
                            R = list(R)     
                                
                            
                            
                            R.sort() #COULD MAKE THIS WAY MORE EFFICIENT
                            Check_Disjoint_Graph = Worst_Reduced[R,:] #Create graph to see if sink stakes are disjoint
                            Check_Disjoint_Graph = Check_Disjoint_Graph[:,R]
                            N, label= connected_components(csgraph=csr_matrix(Check_Disjoint_Graph), directed=False, return_labels=True)
                            Dis_Comp = [[] for x in range(N)]
                            for k in range(len(label)):
                              Dis_Comp[label[k]].append(k)
                                         
                            for k in range(len(Dis_Comp)):
                                  
                                Bridge_Losing_Components[-1].append([])
                                for l in range(len(Bridge_Losing_Components[-1][0])): #Adding the bridge states of the BSCC to the bridge states of the component (because those could destroy the component under refinement)
                                    Bridge_Losing_Components[-1][-1].append(Bridge_Losing_Components[-1][0][l])
                                for l in range(len(Dis_Comp[k])):
                                    List_Potential_Losing_Components[-1].append(Indices[R[Dis_Comp[k][l]]])
                                    Which_Potential_Losing_Component[Indices[R[Dis_Comp[k][l]]]].append([Number_BSCCs, k+1])
                                    if Is_In_Potential_Winning_Component[Indices[R[Dis_Comp[k][l]]]] == 1:
                                        Bridge_Losing_Components[-1][-1].append(Indices[R[Dis_Comp[k][l]]])
                                    else:    
                                        for y in range(len(Reachable_Quantitative_Product[Indices[R[Dis_Comp[k][l]]]])):
                                             if Worst_Markov_Chain[Indices[R[Dis_Comp[k][l]]]][Reachable_Quantitative_Product[Indices[R[Dis_Comp[k][l]]]][y]] == 0 and Best_Markov_Chain[Indices[R[Dis_Comp[k][l]]]][Reachable_Quantitative_Product[Indices[R[Dis_Comp[k][l]]]][y]] > 0: #Comparing the worst and best case Markov Chains to figure out the bridge states
                                                 Bridge_Losing_Components[-1][-1].append(Indices[R[Dis_Comp[k][l]]])
                                      
                                          
                    
                            Number_BSCCs += 1
                    
           Success_Intervals = [[] for x in range(len(Low_Bound))]
           Product_Intervals = [[] for x in range(len(Low_Bounds_Prod))]
            
            
                
           for i in range(len(Upp_Bound)):
                    
                Success_Intervals[i].append(Low_Bound[i])
                Success_Intervals[i].append(Upp_Bound[i])
                
                
           for i in range(len(Upp_Bounds_Prod)):
                    
                Product_Intervals[i].append(Low_Bounds_Prod[i])
                Product_Intervals[i].append(Upp_Bounds_Prod[i])        
                

           with open('State_Space.pkl','wb') as f:
            pickle.dump(State_Space, f)
           with open('Running_Times.pkl','wb') as f:           
            pickle.dump(Running_Times, f)
           with open('List_Optimality_Factors.pkl','wb') as f:             
            pickle.dump(List_Optimality_Factors, f)
           with open('Input_Quantitative_Product.pkl','wb') as f:      
            pickle.dump(Input_Quantitative_Product, f)
           with open('List_Max_Opt.pkl','wb') as f:     
            pickle.dump(List_Max_Opt, f)
           with open('List_Avg_Opt.pkl','wb') as f:     
            pickle.dump(List_Avg_Opt, f)
           with open('Fraction_Above.pkl','wb') as f:     
            pickle.dump(Fraction_Above, f)
           with open('Low_Bound.pkl','wb') as f:     
            pickle.dump(Low_Bound, f)
           with open('Upp_Bound.pkl','wb') as f:     
            pickle.dump(Upp_Bound, f)                    
           with open('Optimal_Policy_Continuous.pkl','wb') as f:      
            pickle.dump(Optimal_Policy_Continuous, f)
 
           print 'Greatest Suboptimality Factor'
           print Greatest_Optimality_Factor
           print 'Number of Refinement Steps:'
           print Tag
           print ''

           if Tag == Num_Ref or Greatest_Optimality_Factor <= Threshold_Uncertainty:
                break

    return (State_Space, Low_Bound, Upp_Bound, Low_Bounds_Prod, Upp_Bounds_Prod, Potential_Policy_Continuous, Optimal_Policy_Continuous, List_Optimality_Factors, Init, L_mapping, Input_Quantitative_Product, Greatest_Permanent_Winning_Component, Permanent_Losing_Components, List_Max_Opt, List_Avg_Opt, Worst_Markov_Chain, Best_Markov_Chain, Running_Times, States_Above_Threshold, Fraction_Above)



def Select_Actions(Input_Qualitative, Reachable_Sets, State_Space, Reachable_States_Cont, Previous_Potential_Winning_Components, Is_New_State, Previous_Reach_Cont, Potential_Winning_Components_Original, i):
    
    List_Overlaps = list([])
    On_States_List = list([list([]),list([]), list([]) ]) #Keeps track of all On states in each overlap, we originally have 3 overlaps
    Maybe_States_List = list([list([]),list([]), list([])]) #Keeps Track of all Maybe States in each overlap, we originally have 3 overlaps
    Discrete_Actions = list([])
    
    Break_Tag = 1
    Not_New_Tag = 0
    if i in Potential_Winning_Components_Original:
        List_Iteration = list(Previous_Reach_Cont)
        Reachable_States_Cont = list([])# Resetting the Reachable_States of such states
        if len(Previous_Potential_Winning_Components) == 0: #If no more potential components, no need to go through second qualitative algorithm             
            Break_Tag = 1
        else:
            Break_Tag = 0
        
    elif Is_New_State == 1:
        List_Iteration = list(Previous_Reach_Cont)
        if len(Previous_Potential_Winning_Components) == 0:                
            Break_Tag = 1
        else:
            Break_Tag = 0
    else:
        List_Iteration = list(Previous_Reach_Cont)

        Not_New_Tag = 1
        if len(Previous_Potential_Winning_Components) == 0:                
            Break_Tag = 1
        else:
            Break_Tag = 0
        
    if Not_New_Tag == 1:


        To_Remove = []
        for j in range(len(Reachable_States_Cont)): #This is because some states may not be reachable with the new set of available inputs
            Target = copy.deepcopy(State_Space[Reachable_States_Cont[j]])
            New_Trigger_Regions = list(Compute_Trigger_Regions(Reachable_Sets, Input_Qualitative, Target,j))

            if not(len(New_Trigger_Regions[0]) > 0 or len(New_Trigger_Regions[1]) > 0):
                To_Remove.append(j)
    
        for ele in sorted(To_Remove, reverse = True):  
            del Reachable_States_Cont[ele] 
    
    for j in range(len(List_Iteration)):
        Target = copy.deepcopy(State_Space[List_Iteration[j]])
        New_Trigger_Regions = list(Compute_Trigger_Regions(Reachable_Sets, Input_Qualitative, Target,j))

        if (len(New_Trigger_Regions[0]) > 0 or len(New_Trigger_Regions[1]) > 0) and (List_Iteration[j] not in Reachable_States_Cont):
            Reachable_States_Cont.append(List_Iteration[j])
#We have to compute all the Overlaps below 

        if Break_Tag == 1:
            continue

            
        if j > 0:            
            Original_Length = len(List_Overlaps)
    

            for k in range(Original_Length):

                New_Overlaps = list([list([]),list([]),list([])]) #First Position: OVERLAP WITH ON TRIGGER REGION, #Second position: OVERLAP WITH MAYBE TRIGGER REGION, #Third position: OVERLAP WITH OFF TRIGGER REGION
                New_Maybe_States_List = list(Maybe_States_List[0])
                New_Maybe_States_List.append(List_Iteration[j])

                New_On_States_List = list(On_States_List[0])
                New_On_States_List.append(List_Iteration[j])
        
    
                for x in range(len(List_Overlaps[0])):
            
            
                    Overlap_Polygon = Polygon([(List_Overlaps[0][x][0][0], List_Overlaps[0][x][0][1]), (List_Overlaps[0][x][0][0], List_Overlaps[0][x][1][1]), (List_Overlaps[0][x][1][0], List_Overlaps[0][x][1][1]), (List_Overlaps[0][x][1][0], List_Overlaps[0][x][0][1])])
                    Overlap_Polygon_Subtract = Polygon([(List_Overlaps[0][x][0][0], List_Overlaps[0][x][0][1]), (List_Overlaps[0][x][0][0], List_Overlaps[0][x][1][1]), (List_Overlaps[0][x][1][0], List_Overlaps[0][x][1][1]), (List_Overlaps[0][x][1][0], List_Overlaps[0][x][0][1])])             
                
                    Has_Intersected = 0
                    #BELOW, WE ASSUME THAT OVERLAPS WHICH ONLY SHARE A POINT OR A LINE DO NOT INTERSECT // NOT SURE ABOUT CORRECTNESS BUT SIMPLIFYING ASSUMPTION
                    if Overlap_Polygon_Subtract.is_valid == False:
                        continue
            
                    for y in range(len(New_Trigger_Regions[0])): 
                
                        Trigger_Poly = Polygon([(New_Trigger_Regions[0][y][0][0], New_Trigger_Regions[0][y][0][1]), (New_Trigger_Regions[0][y][0][0], New_Trigger_Regions[0][y][1][1]), (New_Trigger_Regions[0][y][1][0], New_Trigger_Regions[0][y][1][1]), (New_Trigger_Regions[0][y][1][0], New_Trigger_Regions[0][y][0][1])])
                        Intersect = Trigger_Poly.intersection(Overlap_Polygon)
                        if Intersect.is_empty != 1 and Intersect.geom_type == 'Polygon' and Overlap_Polygon_Subtract.is_valid == True:
                            Has_Intersected = 1
                            New_Overlaps[0].append(list([[Intersect.bounds[0], Intersect.bounds[1]],[Intersect.bounds[2], Intersect.bounds[3]]]))
                            Overlap_Polygon_Subtract = Overlap_Polygon_Subtract.difference(Intersect)
                
                    if Overlap_Polygon_Subtract.is_empty == 1 or Overlap_Polygon_Subtract.is_valid == False: #If it is empty, then the entire polygon has been overlapped already
                        continue
                                
                    for y in range(len(New_Trigger_Regions[1])):   
  
                        Trigger_Poly = Polygon([(New_Trigger_Regions[1][y][0][0], New_Trigger_Regions[1][y][0][1]), (New_Trigger_Regions[1][y][0][0], New_Trigger_Regions[1][y][1][1]), (New_Trigger_Regions[1][y][1][0], New_Trigger_Regions[1][y][1][1]), (New_Trigger_Regions[1][y][1][0], New_Trigger_Regions[1][y][0][1])])
                        Intersect = Trigger_Poly.intersection(Overlap_Polygon)
                        if Intersect.is_empty != 1 and Intersect.geom_type == 'Polygon' and Overlap_Polygon_Subtract.is_valid == True:
                            Has_Intersected = 1
                            New_Overlaps[1].append(list([[Intersect.bounds[0], Intersect.bounds[1]],[Intersect.bounds[2], Intersect.bounds[3]]]))
                            Overlap_Polygon_Subtract = Overlap_Polygon_Subtract.difference(Intersect)

                                                          
                    if Overlap_Polygon_Subtract.is_empty == 1 or Overlap_Polygon_Subtract.is_valid == False:
                        continue
       
                    if Has_Intersected == 1:
            
                        if Overlap_Polygon_Subtract.geom_type == 'MultiPolygon':
                    
                            for pol in Overlap_Polygon_Subtract:  # same for multipolygon.geoms
                        
                                list_r = list([[]])
                                poly_r = shapely.geometry.polygon.orient(pol) 
                        
                                for y in range(len(poly_r.exterior.coords)):
                                    list_r[0].append(list(poly_r.exterior.coords[y]))
                                
                                
                                runtime = get()
                                ctx = runtime.compile('''
                                    module.paths.push('%s');
                                    var decompose = require('rectangle-decomposition'); 
                                    function decompose_region(region){
                                    var rectangles = decompose(region)
                                    return rectangles;
                                    }
                                    
                                ''' % os.path.join(os.path.dirname(__file__),'node_modules'))
                                part = ctx.call("decompose_region",list_r)
    
                                
                                for x in range(len(part)):
                                    New_Overlaps[2].append( list([ [ part[x][0][0], part[x][0][1] ] , [ part[x][1][0], part[x][1][1] ] ]) )

                        
                    #PARSE RESPONSE AND STORE IN NEW_OVERLAPS
                    
                        
                    
                        elif Overlap_Polygon_Subtract.geom_type == 'Polygon':    
                    
                            list_r = list([[]])
                            
                            Overlap_Polygon_Subtract = shapely.geometry.polygon.orient(Overlap_Polygon_Subtract)
                                
                            for y in range(len(Overlap_Polygon_Subtract.exterior.coords)):
                                list_r[0].append(list(Overlap_Polygon_Subtract.exterior.coords[y])) #NEED TO STORE VERTICES IN COUNTERCLOCKWISE DIRECTION
                            
                            runtime = get()
                            ctx = runtime.compile('''
                                module.paths.push('%s');
                                var decompose = require('rectangle-decomposition'); 
                                function decompose_region(region){
                                var rectangles = decompose(region)
                                return rectangles;
                                }
                                
                            ''' % os.path.join(os.path.dirname(__file__),'node_modules'))
                            
    
                            part = ctx.call("decompose_region",list_r)
                            
    
                            
                            for x in range(len(part)):
                                New_Overlaps[2].append(list([[part[x][0][0], part[x][0][1]],[part[x][1][0], part[x][1][1]]]))
                        
                    else:
                        New_Overlaps[2].append(list(List_Overlaps[0][x]))
                               
                for x in range(len(New_Overlaps)):
                    if len(New_Overlaps[x]) != 0:
                        List_Overlaps.append(New_Overlaps[x])
                        if x == 0:                            
                            On_States_List.append(New_On_States_List)
                            Maybe_States_List.append(Maybe_States_List[0])
                        elif x == 1:   
                            On_States_List.append(On_States_List[0])
                            Maybe_States_List.append(New_Maybe_States_List) 
                        else:
                            On_States_List.append(On_States_List[0])
                            Maybe_States_List.append(Maybe_States_List[0])
            
                List_Overlaps.pop(0)
                Maybe_States_List.pop(0)
                On_States_List.pop(0)
        
                        
        else:
            
            for k in range(len(New_Trigger_Regions)):
                List_Overlaps.append(New_Trigger_Regions[k])
            
            On_States_List[0].append(List_Iteration[0])
            Maybe_States_List[1].append(List_Iteration[0])
    
    if Break_Tag == 1:
        Discrete_Actions.append([0,0]) #Random Action since we know this state does create a winning component in the product
        return Discrete_Actions, Reachable_States_Cont
    
    Reach_l_x = Reachable_Sets[0][0]
    Reach_u_x = Reachable_Sets[1][0]
    Reach_l_y = Reachable_Sets[0][1]
    Reach_u_y = Reachable_Sets[1][1]


                
    for j in range(len(List_Overlaps)): #Going through all overlaps and picking the approriate input                    

        if len(Maybe_States_List[j]) <= 1: #The overlap contains at most 1 maybe trigger region
            
            Best_Norm = float('inf')
            
            for k in range(len(List_Overlaps[j])):
                
                 Current_Input = list([])
                 for n in range(len(List_Overlaps[j][k][0])):
                     if List_Overlaps[j][k][0][n] > 0:
                        Current_Input.append(List_Overlaps[j][k][0][n])
                     elif List_Overlaps[j][k][1][n] < 0:
                        Current_Input.append(List_Overlaps[j][k][1][n])
                     else:
                        Current_Input.append(0.0) 
                     
                 Cur_Norm = norm(np.asarray(Current_Input))
                 
                 if Cur_Norm < Best_Norm:
                     Chosen_Input = list(Current_Input)
                     Best_Norm = Cur_Norm
                 
                 # Keeping the Input with smallest norm                               
            
            Discrete_Actions.append(Chosen_Input) 
                
        else:
                        
            O = list(On_States_List[j])
            Y = list(Maybe_States_List[j])

            K = list(Maybe_States_List[j])
            K.sort()
            
            if len(O) == 0:
                L = []
                Combinations_List = list(combinations(Y, len(Y)-1))                                
                for k in range(len(Combinations_List)):
                    List = list(Combinations_List[k])
                    List.sort()
                    L.append(List)                                                                                
                
            else:    
                L = list([list(K)])

            
            m = 0
            tag_loop = 0
            Found_Sets = []
            Found_Actions_Overlap = []
            

            
            while(tag_loop == 0): 
                
 
                S = list(L[m])
                for k in range(len(Found_Sets)):
                    if len(set(S) - Found_Sets[k]) == 0:
                        m += 1    
                        if m == len(L): tag_loop = 1 
                        continue
                
                
                S_dif = list(set(Y) - set(S))
                
                #Instead of solving for minimum u, which is too computationally expensive, we try to solve for a u that satisfies the conditions
                
                #List_Functions contains all the upper bound functions
                List_Functions = list([])
                List_Jac = list([])
                
                for k in range(len(O)):
                    
                    a_x = State_Space[O[k]][0][0] #Lower x bound of state O[k]
                    b_x = State_Space[O[k]][1][0] #Upper x bound of state O[k]
                    a_y = State_Space[O[k]][0][1] #Upper y bound of state O[k]
                    b_y = State_Space[O[k]][1][1] #Upper y bound of state O[k]
                    
                    #Modifying the size of states on the boundary to account for maximum and minimum capacity
                    
                    if a_x == LOW_1:
                        a_x = min(LOW_1 + U_MIN_1 + mus[0] - Semi_Width_1, a_x)
                    
                    elif b_x == UP_1:
                        b_x = max(UP_1 + U_MAX_1 + mus[0] + Semi_Width_1, b_x)
                        
                    if a_y == LOW_2:
                        a_y = min(LOW_2 + U_MIN_2 + mus[1] - Semi_Width_2, a_y)
                        
                    elif b_y == UP_2:
                        b_y = max(UP_2 + U_MAX_2 + mus[1] + Semi_Width_2, b_y)

                    List_Functions.append(lambda x, a_x= a_x, b_x = b_x, a_y = a_y , b_y= b_y, Reach_l_x = Reach_l_x, Reach_u_x = Reach_u_x, Reach_l_y = Reach_l_y, Reach_u_y = Reach_u_y: Upper_Bound_Func(x[0], x[1], a_x, b_x, a_y, b_y, Reach_l_x, Reach_u_x,Reach_l_y, Reach_u_y))
                    List_Jac.append(lambda x, a_x= a_x, b_x = b_x, a_y = a_y , b_y= b_y, Reach_l_x = Reach_l_x, Reach_u_x = Reach_u_x, Reach_l_y = Reach_l_y, Reach_u_y = Reach_u_y: list(Upper_Bound_Func_Jac(x[0], x[1], a_x, b_x, a_y, b_y, Reach_l_x, Reach_u_x,Reach_l_y, Reach_u_y)))
                
                for k in range(len(S_dif)):  

                    a_x = State_Space[S_dif[k]][0][0] 
                    b_x = State_Space[S_dif[k]][1][0] 
                    a_y = State_Space[S_dif[k]][0][1] 
                    b_y = State_Space[S_dif[k]][1][1] 
                    
                    
                    
                    #Modifying the size of states on the boundary to account for maximum and minimum capacity
                    
                    if a_x == LOW_1:
                        a_x = min(LOW_1 + U_MIN_1 + mus[0] - Semi_Width_1, a_x)
                    
                    elif b_x == UP_1:
                        b_x = max(UP_1 + U_MAX_1 + mus[0] + Semi_Width_1, b_x)
                        
                    if a_y == LOW_2:
                        a_y = min(LOW_2 + U_MIN_2 + mus[1] - Semi_Width_2, a_y)
                        
                    elif b_y == UP_2:
                        b_y = max(UP_2 + U_MAX_2 + mus[1] + Semi_Width_2, b_y)                    

                    List_Functions.append(lambda x, a_x= a_x, b_x = b_x, a_y = a_y , b_y= b_y, Reach_l_x = Reach_l_x, Reach_u_x = Reach_u_x, Reach_l_y = Reach_l_y, Reach_u_y = Reach_u_y: Upper_Bound_Func(x[0], x[1], a_x, b_x, a_y, b_y, Reach_l_x, Reach_u_x,Reach_l_y, Reach_u_y))
                    List_Jac.append(lambda x, a_x= a_x, b_x = b_x, a_y = a_y , b_y= b_y, Reach_l_x = Reach_l_x, Reach_u_x = Reach_u_x, Reach_l_y = Reach_l_y, Reach_u_y = Reach_u_y: list(Upper_Bound_Func_Jac(x[0], x[1], a_x, b_x, a_y, b_y, Reach_l_x, Reach_u_x,Reach_l_y, Reach_u_y)))
                    
                  
                    
                f = lambda x: -sum(phi(x) for phi in List_Functions) #Remove the minus when not using basinhopping

                Already_Action = 0
                for n in range(len(Found_Actions_Overlap)):                    
                    if f(Found_Actions_Overlap[n]) <= -1.0:
                        Already_Action = 1
                        break
                
                if Already_Action == 1:
                    Found_Sets.append(set(S))
                    m += 1    
                    if m == len(L): tag_loop = 1 
                    continue

#                f_jac = lambda x: np.sum(np.array([phi(x) for phi in List_Jac]), 0)
               
                Feasability = 0
                
                start3 = timeit.default_timer()
                
                for k in range(len(List_Overlaps[j])):
                    
                    
                    Num_Points = 8
                    XX, YY = mgrid[List_Overlaps[j][k][0][0]:List_Overlaps[j][k][0][1]:complex(0,Num_Points), List_Overlaps[j][k][1][0]:List_Overlaps[j][k][1][1]:complex(0,Num_Points)]
                    positions = vstack([XX.ravel(), YY.ravel()])      
                    loc_best = 0
                    Best_Act_Loc =[uniform(List_Overlaps[j][k][0][0], List_Overlaps[j][k][1][0]),uniform(List_Overlaps[j][k][0][1], List_Overlaps[j][k][1][1])] 

                    for www in range(len(positions)):
                        res = minimize(f, [positions[0][www],positions[1][www]], method='L-BFGS-B', bounds=[(List_Overlaps[j][k][0][0],List_Overlaps[j][k][1][0]),(List_Overlaps[j][k][0][1], List_Overlaps[j][k][1][1])] )
                    
                        if -res.fun > loc_best:
                            loc_best= -res.fun
                            Best_Act_Loc = list([res.x[0], res.x[1]])
                   
                    
                    if loc_best >= 1.0:
                        Feasability = 1   
                        Discrete_Actions.append(list(Best_Act_Loc))
                        Found_Actions_Overlap.append(list(Best_Act_Loc))
                        Found_Sets.append(set(S))
                        break       
                                        
                if Feasability == 0:     
                    
                    if len(S) >= 1: #NO NEED TO HAVE MORE COMBINATIONS WHEN THERE IS ONLY ONE STATE IN S
                        Combinations_List = list(combinations(S, len(S)-1))
                        
                        for k in Combinations_List:
                            List = list(k)
                            List.sort()
                            if List not in L:
                                L.append(List)
                                

                m += 1    
                if m == len(L): tag_loop = 1 
    
    return Discrete_Actions, Reachable_States_Cont

def f_objective(x, f_up, f_low, f_up_jac, f_low_jac, Index, Reach, j):
    

    up = [f_u(x) for f_u in f_up]
    low = [f_l(x) for f_l in f_low]    
    
    sum_low = sum(low[1:])
    sum_up = 0.0
    z = [min(up[0], 1 - sum_low)]

    
    sum_term = -Index[Reach[j][0], 0]*z[0]
    
    for k in range(1, len(f_up)):
        sum_low -= low[k]
        sum_up += z[k-1]
        z.append(min(up[k],1 - sum_up - sum_low ))
            
        sum_term -= z[k]*Index[Reach[j][k], 0]
    
    sum_jac = 0.0 #Dummy jacobian
    
    return sum_term, z, sum_jac

def f_objective_jac(x, f_up, f_low, f_up_jac, f_low_jac, Index, Reach, j):
    
    sum_term = 0.0
    sum_jac = [0.0, 0.0]
    up = [f_up[i](x) for i in range(len(f_up))]
    low = [f_low[i](x) for i in range(len(f_low))]
    low_jac = [f_low_jac[i](x) for i in range(len(f_low_jac))]
    
    sum_low = sum(low[1:])
    sum_low_jac = [sum(n) for n in zip(*low_jac[1:])]
    sum_up = 0.0
    sum_up_jac = [0.0, 0.0]

    if up[0] < 1 - sum_low:
        z = [up[0]]
        z_jac = [f_up_jac[0](x)]      
    else:
        z = [1 - sum_low]
        z_jac = [[- n for n in sum_low_jac]]
    z = [min(up[0], 1 - sum_low)]
    
    sum_jac = [Index[Reach[j][0], 0]*n for n in z_jac[0]]
    sum_term = Index[Reach[j][0], 0]*z[0]
    
    for k in range(1, len(f_up)):
        sum_low = sum_low - low[k]
        sum_low_jac = map(sub, sum_low_jac , low_jac[k])
        sum_up = sum_up + z[k-1]
        sum_up_jac = map(add, sum_up_jac, z_jac[k-1])
        if up[k] < 1 - sum_up - sum_low:
            z.append(up[k])
            z_jac.append(f_up_jac[k](x))
        else:
            z.append(1 - sum_up - sum_low)
            z_jac.append([-n for n in map(add,sum_low_jac, sum_up_jac)])
            
        z.append(min(up[k], 1 - sum_up - sum_low))
        sum_term = sum_term + z[k]*Index[Reach[j][k], 0]
        sum_jac = map(add, sum_jac , [Index[Reach[j][k], 0]*n for n in z_jac[k]])
    
    sum_term = -sum_term
    sum_jac = np.array([-n for n in sum_jac])
    
    return sum_term, z, sum_jac




def csr_zero_rows(csr, rows_to_zero):
    rows, cols = csr.shape
    mask = np.ones((rows,), dtype=np.bool)
    mask[rows_to_zero] = False
    nnz_per_row = np.diff(csr.indptr)

    mask = np.repeat(mask, nnz_per_row)
    nnz_per_row[rows_to_zero] = 0
    csr.data = csr.data[mask]
    csr.indices = csr.indices[mask]
    csr.indptr[1:] = np.cumsum(nnz_per_row)
    
    return csr

def zero_rows(M, rows):
    diag = scipy.sparse.eye(M.shape[0]).tolil()
    for r in rows:
        diag[r, r] = 0
    return diag.dot(M)

def Uncertainty_Ranking_Path_New3(Upper_csr, Lower_csr, Maybe, Success_Int, IA_u, IA_l, Is_in_WC_L, Which_WC_L, Bridge_WC_L, Is_in_LC_L, Which_LC_L, Bridge_LC_L, Is_in_WC_P, Which_WC_P, Bridge_WC_P, Is_in_LC_P, Which_LC_P, Bridge_LC_P, Is_in_P, S_Space, V_Total, Len_A):
    
    #Computes an Uncertainty Ranking based on the states satisfiability intervals
    
    Upper_shape = Upper_csr.shape[0]
    Difference_Success = [0]*Upper_shape
    Reachable_States = [[] for x in range(Upper_shape)] #List all _ vertices from a given vertex
    Pre_States = [[] for x in range(Upper_shape)] #List all _ vertices from a given vertex   
    
    Score = [0]*Upper_shape #Individual uncertainty score of each state, which is the difference between 
    Check_Score = [0]*Upper_shape
    Absorbing = [0]*Upper_shape

    

    for i, Success in enumerate(Success_Int):
        Difference_Success[i] = Success[1] - Success[0]

        if Is_in_WC_L[i] == 1 or Is_in_LC_L[i] == 1:
            if Upper_csr.indptr[i] - Upper_csr.indptr[i+1] != 0:
                Upper_csr  = csr_zero_rows(Upper_csr, i)
            Upper_csr[i,i] = 1 
            Absorbing[i] = 1
            
        elif Is_in_P[i]  == 1:
            if Upper_csr.indptr[i] - Upper_csr.indptr[i+1] != 0:
                Upper_csr  = csr_zero_rows(Upper_csr, i) 
            
            Absorbing[i] = 1
            
 
        
            
    ## First, find the states which the states in Maybe can reach, and their pre-states
    N_components, component_list = sparse.csgraph.connected_components(Upper_csr, connection='strong')


    
    All_Components = [[] for x in range(N_components)]
    Reduced_Graph = {k: set([]) for k in range(N_components)}
    Is_Visited = [0]*N_components
    Visited_Set = []
    Reach_Reduced = [set([]) for k in range(N_components)]
    Pre_Reduced = [set([]) for k in range(N_components)]
    
    conversion = [0]*N_components
    
    

    for j, comp in enumerate(component_list):
        All_Components[comp].append(j)
        if Absorbing[j] == 0:
            Reachable_Edges = list(Upper_csr[j, :].nonzero()[1]) 
            for k in Reachable_Edges:
                if comp != component_list[k]: #To avoid loops in reduced graph
                    Reduced_Graph[comp].add(component_list[k])
            

    
    for j in Maybe:
        
        if Is_Visited[component_list[j]] == 1:
            continue
        
        Visited_Edges = []
        Path = [component_list[j]]
        Reach_Reduced[Path[-1]].add(Path[-1])
        Pre_Reduced[Path[-1]].add(Path[-1])

        
        if conversion[Path[-1]] == 0:
            conversion[Path[-1]] = 1
            Reduced_Graph[Path[-1]] = list(Reduced_Graph[Path[-1]])
               
        while len(Path) != 0:
            
            if Is_Visited[Path[-1]] == 1:
                for k in Reach_Reduced[Path[-1]]:
                    for n in Path[:-1]:
                        Pre_Reduced[k].add(n)
                        Reach_Reduced[n].add(k)
                
                Path.pop()
                continue
                    
                    
            if len(Visited_Edges) < len(Path):
                Visited_Edges.append([])
            

            if len(Visited_Edges[-1]) == len(Reduced_Graph[Path[-1]]):
                Is_Visited[Path[-1]] = 1
                Visited_Set.append(Path[-1])
                Path.pop()
                Visited_Edges.pop()
                continue


            if Is_Visited[Reduced_Graph[Path[-1]][len(Visited_Edges[-1])]] == 0:
                for k in Path:
                    Reach_Reduced[k].add(Reduced_Graph[Path[-1]][len(Visited_Edges[-1])])
                    Pre_Reduced[Reduced_Graph[Path[-1]][len(Visited_Edges[-1])]].add(k)
                Pre_Reduced[Reduced_Graph[Path[-1]][len(Visited_Edges[-1])]].add(Reduced_Graph[Path[-1]][len(Visited_Edges[-1])])
                Reach_Reduced[Reduced_Graph[Path[-1]][len(Visited_Edges[-1])]].add(Reduced_Graph[Path[-1]][len(Visited_Edges[-1])])

            Path.append(Reduced_Graph[Path[-1]][len(Visited_Edges[-1])])
            Visited_Edges[-1].append(Reduced_Graph[Path[-2]][len(Visited_Edges[-1])])
            

            if conversion[Path[-1]] == 0:
                conversion[Path[-1]] = 1
                Reduced_Graph[Path[-1]] = list(Reduced_Graph[Path[-1]])


    
    Reach_Subset = []
        
    

    for k in Visited_Set:
        Reach_Subset.extend(All_Components[k])
        for j in All_Components[k]:
            for h in Reach_Reduced[k]:
                    for m in All_Components[h]:
                        if j != m:
                            Reachable_States[j].append(m)
        
        for h in Pre_Reduced[k]:           
            for j in All_Components[k]:
                    for m in All_Components[h]:
                        if j != m:
                            Pre_States[j].append(m)


    
    Reachability_Matrix = {}
    

                      


    #COULD REDUCE TIME BY NOT SLICING SPARSE MATRIX IF HAVE ENOUGH MEMORY. INSTEAD, DIRECTLY CONVERT UPPER_CSR TO DENSE
    
    for j in Reach_Subset:
        if Is_in_P[j] == 0:
            Pre = list(Pre_States[j])
            len_Pre = len(Pre)
            if len_Pre == 1:
                Reachability_Matrix[Pre[0], j] = float(Upper_csr[Pre[0],j])/(1.0-Upper_csr[Pre[0],Pre[0]])
            elif len_Pre > 1:
                Slice = Upper_csr[Pre,:]
                Sub_Matrix = Slice[:, Pre]
                b = Slice[:,j]                 
                Prob_Reach = spsolve(identity(len_Pre) - Sub_Matrix, b) 
                for l in range(Prob_Reach.shape[0]):
                    Reachability_Matrix[Pre[l], j] = Prob_Reach[l]
                
        
        
    Weights = [0]*Upper_shape
    V_Weight = 1.0
    
    
    for j in Maybe:
        
        if Check_Score[j] == 0:
            Score[j] = norm(Upper_csr[j, :].todense() - Lower_csr[j, :].todense())
            Check_Score[j] = 1
            
        
        Weights[j] += Score[j]
        

        
        for k in Reachable_States[j]:

            if Is_in_P[k] == 0:
                        
                if Is_in_WC_L[k] == 1:
                    Current_A = Which_WC_L[k]
                    Max_Probability_Gain = V_Weight*Difference_Success[k]*Reachability_Matrix[j, k]
                    for q in range(len(Current_A)):
                        for i in range(len(Bridge_WC_L[Current_A[q][0]][Current_A[q][1]])):                   
                                Weights[Bridge_WC_L[Current_A[q][0]][Current_A[q][1]][i]] += Max_Probability_Gain                      
                    continue

        
                if Is_in_WC_P[k] == 1:
                    Current_A = Which_WC_P[k]
                    Max_Probability_Gain = V_Weight*Difference_Success[k]*Reachability_Matrix[j, k]
                    for q in range(len(Current_A)):
                        for i in range(len(Bridge_WC_P[Current_A[q][0]][Current_A[q][1]])):
                                Weights[Bridge_WC_P[Current_A[q][0]][Current_A[q][1]][i]] += Max_Probability_Gain                      
                    continue

                
                if Is_in_LC_L[k] == 1:
                    Current_N_A = Which_LC_L[k]
                    Max_Probability_Gain = V_Weight*Difference_Success[k]*Reachability_Matrix[j, k]
                    for q in range(len(Current_N_A)):
                        for i in range(len(Bridge_LC_L[Current_N_A[q][0]][Current_N_A[q][1]])):
                                Weights[Bridge_LC_L[Current_N_A[q][0]][Current_N_A[q][1]][i]] += Max_Probability_Gain                                      
                    continue

                
                if Is_in_LC_P[k] == 1:
                    Current_N_A = Which_LC_P[k]
                    Max_Probability_Gain = V_Weight*Difference_Success[k]*Reachability_Matrix[j, k]
                    for q in range(len(Current_N_A)):
                        for i in range(len(Bridge_LC_P[Current_N_A[q][0]][Current_N_A[q][1]])):
                                Weights[Bridge_LC_P[Current_N_A[q][0]][Current_N_A[q][1]][i]] += Max_Probability_Gain                      
                    continue
                
                if Check_Score[k] == 0:
                    Score[k] = norm(Upper_csr[k, :].todense() - Lower_csr[k, :].todense())
                    Check_Score[k] = 1

                
                Weights[k] += Reachability_Matrix[j, k]* Score[k]

     
                  
    return Weights



def Raw_Refinement(State, Space): 
       
    New_St = []       
    for i in State:

       a1 = Space[i][1][0] - Space[i][0][0]
       a2 = Space[i][1][1] - Space[i][0][1]
    
       if a1 > a2:
                               
           New_St.append([(Space[i][0][0],Space[i][0][1]),((Space[i][1][0] + Space[i][0][0])/2.0,Space[i][1][1])])
           New_St.append([((Space[i][1][0] + Space[i][0][0])/2.0 , Space[i][0][1]),(Space[i][1][0],Space[i][1][1])])
  
       else:
           
           New_St.append([(Space[i][0][0] , (Space[i][1][1]+Space[i][0][1])/2.0),(Space[i][1][0],Space[i][1][1])])
           New_St.append([(Space[i][0][0] , Space[i][0][1]),(Space[i][1][0],(Space[i][1][1]+Space[i][0][1])/2.0)])
   
    return New_St





def Find_Components_And_Bridge_States(Graph, Which, Indices, Is_State_Bridge, Absorbing_Tag):
    
    # The variable Absorbing Tag tells you whether an absorbing state was added at the end of the graph for all unconnected components 
    Comp_Graph = csr_matrix(Graph)
    Num_Comp, labels =  connected_components(csgraph=Comp_Graph, directed=False, return_labels=True) #Compute all connected components of the graph
    Bridges = [] #Bridges contains a list of lists, [[],[],...,[]] where the outer list corresponds to the BSCC Number and the inner lists correspond to the components of the BSCC 
    List_Comp = []
    
    C = [[] for x in range(Num_Comp)]
    
    for i in range(len(labels)):
        C[labels[i]].append(i)
    

    #For all connected components, we compute the SCCs of their subgraph
    Number_BSCCs = 0
    
    for i in range(len(C)):

        Comp = C[i]
        Comp.sort()
        Sub_Graph = Graph[Comp, :]
        Sub_Graph = Sub_Graph[:, Comp]   
        SCCs, n = SSCC(Sub_Graph)
        
        for j in range(len(SCCs)):            
            SCC = list(SCCs[j])
            BSCC = 1
            for k in range(len(SCC)):            
                if sum(Sub_Graph[SCC[k],SCC]) < sum(Sub_Graph[SCC[k],:]): #This means, if the state in the SCC leaks
                    BSCC = 0
                    break
            
            if BSCC == 1: #It means that this SCC is a BSCC of this sub-graph.               

               Bridges.append([]) 
               List_Comp.append([])
               
               
               if Absorbing_Tag == 1 and Comp[SCC[0]] == Graph.shape[0] - 1: #If the BSCC is the absorbing states 
                    
                    
                    Bridges[-1].append([])
                    List_Comp[-1].append([]) #append Dummy component
                    Gr = igraph.Graph.Adjacency(Sub_Graph.tolist())                    
                    R = Gr.subcomponent(SCC[0], mode="IN") #Sink states for the absorbing state                    
                    R.sort()
                    R.pop()
                    Check_Disjoint_Graph = Sub_Graph[R,:] #Create graph to see if sink stakes are disjoint
                    Check_Disjoint_Graph = Check_Disjoint_Graph[:,R]
                    N, label= connected_components(csgraph=csr_matrix(Check_Disjoint_Graph), directed=False, return_labels=True)
                    Dis_Comp = [[] for x in range(N)]
                    for k in range(len(label)):
                        Dis_Comp[label[k]].append(k)
                     

    
                    for k in range(len(Dis_Comp)):
                         List_Comp[-1].append([])
                         Bridges[-1].append([])
                         for l in range(len(Dis_Comp[k])): 
                            List_Comp[-1][-1].append(Indices[Comp[R[Dis_Comp[k][l]]]])
                            Which[Indices[Comp[R[Dis_Comp[k][l]]]]].append([Number_BSCCs, k+1])
                            if Is_State_Bridge[Comp[R[Dis_Comp[k][l]]]] == 1:
                                Bridges[-1][-1].append(Indices[Comp[R[Dis_Comp[k][l]]]])
                   
                    
                    
               else:                   
                    Bridges[-1].append([])
                    List_Comp[-1].append([])
                    for k in range(len(SCC)):
                        List_Comp[-1][-1].append(Indices[Comp[SCC[k]]])
                        if Is_State_Bridge[Comp[SCC[k]]] == 1:
                            Bridges[-1][-1].append(Indices[Comp[SCC[k]]])
                    Gr = igraph.Graph.Adjacency(Sub_Graph.tolist())
                    R = [] #Contains all sink states w.r.t current BSCC
                    for q in range(len(SCC)):                 
                        Res = Gr.subcomponent(SCC[q], mode="IN")
                        R2 = [x for x in Res if x not in R]
                        R.extend(R2)
                    R = list(set(R) - set(SCC)) #Removing the BSCC states from the set of states which can reach the BSCC    
                    R.sort()
                    Check_Disjoint_Graph = Sub_Graph[R,:] #Create graph to see if sink stakes are disjoint
                    Check_Disjoint_Graph = Check_Disjoint_Graph[:,R]
                    N, label= connected_components(csgraph=csr_matrix(Check_Disjoint_Graph), directed=False, return_labels=True)
                    Dis_Comp = [[] for x in range(N)]
                    for k in range(len(label)):
                        Dis_Comp[label[k]].append(k)
                     
                    for k in range(len(Dis_Comp)):
                         List_Comp[-1].append([])
                         Bridges[-1].append([])
                         for l in range(len(Bridges[-1][0])): #Adding the bridge states of the BSCC to the bridge states of the component (because those could destroy the component under refinement)
                             Bridges[-1][-1].append(Bridges[-1][0][l])
                         for l in range(len(Dis_Comp[k])):
                            List_Comp[-1][-1].append(Indices[Comp[R[Dis_Comp[k][l]]]])
                            Which[Indices[Comp[R[Dis_Comp[k][l]]]]].append([Number_BSCCs, k+1])
                            if Is_State_Bridge[Comp[R[Dis_Comp[k][l]]]] == 1:
                                Bridges[-1][-1].append(Indices[Comp[R[Dis_Comp[k][l]]]])
                                

        
               Number_BSCCs += 1
    
    return List_Comp, Which, Bridges
    




def SSCC(graph):
    
    #Search for all Strongly Connected Components in a Graph

    #set of visited vertices
    used = set()
    
    #call first depth-first search
    list_vector = [] #vertices in topological sorted order
    for vertex in range(len(graph)):
       if vertex not in used:
          (list_vector,used) = first_dfs(vertex, graph, used, list_vector)              
    list_vector.reverse()
    
    #preparation for calling second depth-first search
    graph_t = reverse_graph(graph)
    used = set()
    
    #call second depth-first search
    components= []
    list_components = [] #strong-connected components
    scc_quantity = 0 #quantity of strong-connected components 
    for vertex in list_vector:
        if vertex not in used:
            scc_quantity += 1
            list_components = []
            (list_components, used) = second_dfs(vertex, graph_t, list_components, list_vector, used)
            components.append(list_components)
            

    
    return components, scc_quantity

def first_dfs(vertex, graph, used, list_vector):
    used.add(vertex)
    for v in range(len(graph)):   
        if graph[vertex][v] == 1 and v not in used:   
            (list_vector, used) = first_dfs(v, graph, used, list_vector)
    list_vector.append(vertex)
    return(list_vector, used)

def second_dfs(vertex, graph_t, list_components, list_vector, used):
    used.add(vertex)
    for v in list_vector:   
        if graph_t[vertex][v] == 1 and v not in used:   
            (list_components, used) = second_dfs(v, graph_t, list_components, list_vector, used)
    list_components.append(vertex)
    return(list_components, used)

def reverse_graph(graph):
    graph_t = list(zip(*graph))
    return graph_t










def Index_Update(First_Loop, Reachable_States, Set_Refinement, Index_Stop, m):

    for z in range(len(Reachable_States)):
        
        if First_Loop[z] == 1:
            First_Loop[z] = 0
            y = 0
            Count = 0
            Checked_All_Set = 0
            while(y+Count != len(Reachable_States[z])):
                
                if Checked_All_Set == 0:
                    if y+Count == (Set_Refinement[Count])+Count:
                        Reachable_States[z][y+Count] = []
                        Reachable_States[z].insert(y+Count,[])
                        Index_Stop[z,y+Count] = 0
                        Index_Stop[z,y+Count+1] = 0
                        Count += 1
                        if Count == len(Set_Refinement):
                            Checked_All_Set = 1
                    else:
                        End_List = 1
                        for k in range(len(Reachable_States[z][y+Count])):
                            if(Reachable_States[z][y+Count][k] == Set_Refinement[0]):
                                del Reachable_States[z][y+Count][k]
                                Index_Stop[z,y+Count] = k
                                End_List = 0
                                break
                            elif(Reachable_States[z][y+Count][k] > Set_Refinement[0]):                                
                                Index_Stop[z,y+Count] = k
                                End_List = 0
                                break
                                                    
                        if End_List == 1:
                            Index_Stop[z,y+Count] = len(Reachable_States[z][y+Count]) 
                            
                        
                else:
                    End_List = 1
                    for k in range(len(Reachable_States[z][y+Count])):
                        if(Reachable_States[z][y+Count][k] == Set_Refinement[0]):
                            del Reachable_States[z][y+Count][k]
                            Index_Stop[z,y+Count] = k
                            End_List = 0
                            break
                        elif(Reachable_States[z][y+Count][k] > Set_Refinement[0]):                                
                            Index_Stop[z,y+Count] = k
                            End_List = 0
                            break
    
                    if End_List == 1:
                        Index_Stop[z,y+Count] = len(Reachable_States[z][y+Count])                    
                
                y += 1
                            
            if (len(Set_Refinement)== 1):
                for k in range(len(Reachable_States[z])):
                    for n in range(Index_Stop[z,k], len(Reachable_States[z][k])):
                        Reachable_States[z][k][n] += 1
                         
        else:
            
            for k in range(len(Reachable_States[z])):
                deleted_state = 0
                for n in range(Index_Stop[z,k], len(Reachable_States[z][k])):                                                  
                    if (Reachable_States[z][k][n-deleted_state] == Set_Refinement[m]):
                        if (m < len(Set_Refinement) - 1): 
                            del Reachable_States[z][k][n]
                            Index_Stop[z,k] = n
                            deleted_state = 1
                            break
                        else:
                            del Reachable_States[z][k][n]
                            deleted_state = 1
                                                          
                    elif (Reachable_States[z][k][n-deleted_state]  > Set_Refinement[m]):
                        if (m < len(Set_Refinement) - 1):
                            Index_Stop[z,k] = n
                            break
                        else:
                                Reachable_States[z][k][n-deleted_state] += m + 1 
                    else:
                        Reachable_States[z][k][n] += m
                        if n == len(Reachable_States[z][k]) - 1:
                            Index_Stop[z,k] = len(Reachable_States[z][k])    

    return First_Loop, Reachable_States, Set_Refinement, Index_Stop, m



def Index_Update_Continuous(First_Loop, Reachable_States, Set_Refinement, Index_Stop, m):

    if First_Loop == 1:
        First_Loop = 0
        y = 0
        Count = 0
        Checked_All_Set = 0
        while(y+Count != len(Reachable_States)):
            
            if Checked_All_Set == 0:
                if y+Count == (Set_Refinement[Count])+Count:
                    Reachable_States[y+Count] = []
                    Reachable_States.insert(y+Count,[])
                    Index_Stop[y+Count] = 0
                    Index_Stop[y+Count+1] = 0
                    Count += 1
                    if Count == len(Set_Refinement):
                        Checked_All_Set = 1
                else:
                    End_List = 1
                    for k in range(len(Reachable_States[y+Count])):
                        if(Reachable_States[y+Count][k] == Set_Refinement[0]):
                            del Reachable_States[y+Count][k]
                            Index_Stop[y+Count] = k
                            End_List = 0
                            break
                        elif(Reachable_States[y+Count][k] > Set_Refinement[0]):                                
                            Index_Stop[y+Count] = k
                            End_List = 0
                            break
                                                
                    if End_List == 1:
                        Index_Stop[y+Count] = len(Reachable_States[y+Count]) 
                        
                    
            else:
                End_List = 1
                for k in range(len(Reachable_States[y+Count])):
                    if(Reachable_States[y+Count][k] == Set_Refinement[0]):
                        del Reachable_States[y+Count][k]
                        Index_Stop[y+Count] = k
                        End_List = 0
                        break
                    elif(Reachable_States[y+Count][k] > Set_Refinement[0]):                                
                        Index_Stop[y+Count] = k
                        End_List = 0
                        break

                if End_List == 1:
                    Index_Stop[y+Count] = len(Reachable_States[y+Count])                    
            
            y += 1
                        
        if (len(Set_Refinement)== 1):
            for k in range(len(Reachable_States)):
                for n in range(Index_Stop[k], len(Reachable_States[k])):
                    Reachable_States[k][n] += 1
                     
    else:
        
        for k in range(len(Reachable_States)):
            deleted_state = 0
            for n in range(Index_Stop[k], len(Reachable_States[k])):                                                  
                if (Reachable_States[k][n-deleted_state] == Set_Refinement[m]):
                    if (m < len(Set_Refinement) - 1): 
                        del Reachable_States[k][n]
                        Index_Stop[k] = n
                        deleted_state = 1
                        break
                    else:
                        del Reachable_States[k][n]
                        deleted_state = 1
                                                      
                elif (Reachable_States[k][n-deleted_state]  > Set_Refinement[m]):
                    if (m < len(Set_Refinement) - 1):
                        Index_Stop[k] = n
                        break
                    else:
                            Reachable_States[k][n-deleted_state] += m + 1 
                else:

                    Reachable_States[k][n-deleted_state] += m
#                    Reachable_States[k][n] += m
                    if n == len(Reachable_States[k]) - 1:
                        Index_Stop[k] = len(Reachable_States[k])    

    return First_Loop, Reachable_States, Set_Refinement, Index_Stop, m





def BMDP_Computation_Refinement(R_Set, Target_Set, Set_Ref1, Set_Ref2, L_Bound_Matrix, U_Bound_Matrix, Is_New, Reachable_States, Is_Bridge_State, Bridge_Transitions):
 
    
    
    Sigma1 = sigma1
    Sigma2 = sigma2
    semi_Width_1 = Semi_Width_1
    semi_Width_2 = Semi_Width_2
    Mu1 = mu1
    Mu2 = mu2
    sqr = sqrt(2)
    
    
    Z1 = (erf(semi_Width_1/Sigma1)/sqr) - (erf(-semi_Width_1/Sigma1)/sqr)
    Z2 = (erf(semi_Width_2/Sigma2)/sqr) - (erf(-semi_Width_2/Sigma2)/sqr)
    low_1 = LOW_1
    up_1 = UP_1
    low_2 = LOW_2
    up_2 = UP_2
    tt1 = Mu1 - semi_Width_1
    ll1 = Mu1 + semi_Width_1
    tt2 = Mu2 - semi_Width_2
    ll2 = Mu2 + semi_Width_2
    
    
    for z, R in enumerate(R_Set):
        
        for j, Rz in enumerate(R):
                                    
            r0 = Rz[0][0]
            r1 = Rz[1][0]
            r2 = Rz[0][1]
            r3 = Rz[1][1]
         

            rr0 = r0 + tt1
            rr1 = r1 + ll1
            rr2 = r2 + tt2
            rr3 = r3 + ll2
            
                
            if Is_New[j] == 1:

                
                 for h, Tar in enumerate(Target_Set):
                     
                    q0 = Tar[0][0]
                    q1 = Tar[1][0]
                    q2 = Tar[0][1]
                    q3 = Tar[1][1]
                   
                    
                    if q0 == low_1 and rr0 < low_1:
                        q0 = rr0
                        
                    if q1 == up_1 and rr1 > up_1:
                        q1 = rr1
                        
                    if q2 == low_2 and rr2 < low_2:
                        q2 = rr2
                        
                    if q3 == up_2 and rr3 > up_2:
                        q3 = rr3



                    
                    if (rr0 >= q1 ) or (rr1 <= q0 ) or (rr2 >= q3) or (rr3 <= q2):
                        L_Bound_Matrix[z][j][h] = 0
                        U_Bound_Matrix[z][j][h] = 0                        
                        continue

    
    
                    bisect.insort(Reachable_States[z][j], h)
                                                                                                                   
                    
                    a1_Opt = ((q0 + q1)/2.0) - Mu1
                    a2_Opt = ((q2 + q3)/2.0) - Mu2

                                    
                    
                    if (r1 < a1_Opt): 
                        a1_Max = r1
                        a1_Min = r0
                    elif(r0 > a1_Opt): 
                        a1_Max = r0
                        a1_Min = r1
                    else: 
                        a1_Max = a1_Opt       
                        if (a1_Opt <= (r1+r0)/2.0):
                            a1_Min = r1
                        else:
                            a1_Min = r0
                                                            
                    if (r2 > a2_Opt): 
                        a2_Max = r2
                        a2_Min = r3
                    elif(r3 < a2_Opt): 
                        a2_Max = r3
                        a2_Min = r2
                    else: 
                        a2_Max = a2_Opt
                        if (a2_Opt <= (r2+r3)/2.0):
                            a2_Min = r3
                        else:
                            a2_Min = r2
                                
                    
                                                                                               
                
                    if (a1_Max + Mu1 - semi_Width_1) > q0 and (a1_Max + Mu1 + semi_Width_1) < q1 and (a2_Max + Mu2 - semi_Width_2) > q2 and (a2_Max + Mu2 + semi_Width_2) < q3:
                        H = 1
                    else:
                        
                        b0 = max(a1_Max + Mu1 - semi_Width_1, q0)
                        b1 = min(a1_Max + Mu1 + semi_Width_1, q1)
                        b2 = max(a2_Max + Mu2 - semi_Width_2, q2)
                        b3 = min(a2_Max + Mu2 + semi_Width_2, q3)
                        
                           
                        
                        H = ( ( (erf((b1 - a1_Max - Mu1)/Sigma1)/sqr) - (erf((b0 - a1_Max - Mu1)/Sigma1)/sqr) ) / (Z1)) * (( (erf((b3 - a2_Max - Mu2)/Sigma2)/sqr) - (erf((b2 - a2_Max - Mu2)/Sigma2)/sqr) ) / ( Z2 )) 
                
                        if H > 1:
                            H = 1
                            
                        
                        
                    if (a1_Min + Mu1 + semi_Width_1 <= q0) or (a1_Min + Mu1 - semi_Width_1 >= q1) or (a2_Min + Mu2 + semi_Width_2 <= q2) or (a2_Min + Mu2 - semi_Width_2 >= q3) :
                        Is_Bridge_State[z][j] = 1
                        bisect.insort(Bridge_Transitions[z][j],h)
                        L_Bound_Matrix[z][j][h] = 0
                        U_Bound_Matrix[z][j][h] = H
                        continue                
                    
                    
                    else:
                        
                        b0 = max(a1_Min + Mu1 - semi_Width_1, q0)
                        b1 = min(a1_Min + Mu1 + semi_Width_1, q1)
                        b2 = max(a2_Min + Mu2 - semi_Width_2, q2)
                        b3 = min(a2_Min + Mu2 + semi_Width_2, q3)
                        

                    
                    
              # Perform integration for lower bound probability of transition, based on Gaussian overlap      
                    L = ( ( (erf((b1 - a1_Min - Mu1)/Sigma1)/sqr) - (erf((b0 - a1_Min - Mu1)/Sigma1)/sqr) ) / (Z1)) * (( (erf((b3 - a2_Min - Mu2)/Sigma2)/sqr) - (erf((b2 - a2_Min - Mu2)/Sigma2)/sqr) ) / ( Z2 )) 
            
            
                    if L < 0:
                        L = 0
                        
                    L_Bound_Matrix[z][j][h] = L
                    U_Bound_Matrix[z][j][h] = H                        
    
            else:
                                
                for h in Set_Ref1:
     
                    q0 = Target_Set[h][0][0]
                    q1 = Target_Set[h][1][0]
                    q2 = Target_Set[h][0][1]
                    q3 = Target_Set[h][1][1]

    
                    if q0 == low_1 and rr0 < low_1:
                        q0 = rr0
                        
                    if q1 == up_1 and rr1 > up_1:
                        q1 = rr1
                        
                    if q2 == low_2 and rr2 < low_2:
                        q2 = rr2
                        
                    if q3 == up_2 and rr3 > up_2:
                        q3 = rr3                      

                   
                    
                    if (rr0 >= q1) or (rr1 <= q0) or (rr2 >= q3) or (rr3 <= q2):
                        L_Bound_Matrix[z][j][h] = 0
                        U_Bound_Matrix[z][j][h] = 0
                        continue 
    
                    bisect.insort(Reachable_States[z][j],h)                                                                                               
                    
                    a1_Opt = ((q0 + q1)/2.0) - Mu1
                    a2_Opt = ((q2 + q3)/2.0) - Mu2

                                    
                    
                    if (r1 < a1_Opt): 
                        a1_Max = r1
                        a1_Min = r0
                    elif(r0 > a1_Opt): 
                        a1_Max = r0
                        a1_Min = r1
                    else: 
                        a1_Max = a1_Opt       
                        if (a1_Opt <= (r1+r0)/2.0):
                            a1_Min = r1
                        else:
                            a1_Min = r0
                                                            
                    if (r2 > a2_Opt): 
                        a2_Max = r2
                        a2_Min = r3
                    elif(r3 < a2_Opt): 
                        a2_Max = r3
                        a2_Min = r2
                    else: 
                        a2_Max = a2_Opt
                        if (a2_Opt <= (r2+r3)/2.0):
                            a2_Min = r3
                        else:
                            a2_Min = r2
     
                                                                                               
                
                    if a1_Max + Mu1 - semi_Width_1  > q0 and a1_Max + Mu1 + semi_Width_1 < q1 and a2_Max + Mu2 - semi_Width_2 > q2 and a2_Max + Mu2 + semi_Width_2 < q3 :
                        H = 1
                    else:
                        
                        b0 = max(a1_Max + Mu1 - semi_Width_1, q0)
                        b1 = min(a1_Max + Mu1 + semi_Width_1, q1)
                        b2 = max(a2_Max + Mu2 - semi_Width_2, q2)
                        b3 = min(a2_Max + Mu2 + semi_Width_2, q3)
                                                    
                        
                        H = ( ( (erf((b1 - a1_Max - Mu1)/Sigma1)/sqr) - (erf((b0 - a1_Max - Mu1)/Sigma1)/sqr) ) / (Z1)) * (( (erf((b3 - a2_Max - Mu2)/Sigma2)/sqr) - (erf((b2 - a2_Max - Mu2)/Sigma2)/sqr) ) / ( Z2 )) 
                
                        if H > 1:
                            H = 1
                            
                        
                        
                    if (a1_Min + Mu1 + semi_Width_1 <= q0) or (a1_Min + Mu1 - semi_Width_1 >= q1) or (a2_Min + Mu2 + semi_Width_2 <= q2) or (a2_Min + Mu2 - semi_Width_2 >= q3) :
                        Is_Bridge_State[z][j] = 1
                        bisect.insort(Bridge_Transitions[z][j],h)                   
                        L_Bound_Matrix[z][j][h] = 0
                        U_Bound_Matrix[z][j][h] = H
                        continue
    
                    else:
                        
                        b0 = max(a1_Min + Mu1 - semi_Width_1, q0)
                        b1 = min(a1_Min + Mu1 + semi_Width_1, q1)
                        b2 = max(a2_Min + Mu2 - semi_Width_2, q2)
                        b3 = min(a2_Min + Mu2 + semi_Width_2, q3)
                        
                        
                    
                    
              # Perform integration for lower bound probability of transition, based on Gaussian overlap      
                    L = ( ( (erf((b1 - a1_Min - Mu1)/Sigma1)/sqr) - (erf((b0 - a1_Min - Mu1)/Sigma1)/sqr) ) / (Z1)) * (( (erf((b3 - a2_Min - Mu2)/Sigma2)/sqr) - (erf((b2 - a2_Min - Mu2)/Sigma2)/sqr) ) / ( Z2 )) 
            
                
                    if L < 0:
                        L = 0
                        
                        
                    L_Bound_Matrix[z][j][h] = L
                    U_Bound_Matrix[z][j][h] = H
                    
                    
                    
                    
                for h in Set_Ref2:
                    
                    q0 = Target_Set[h][0][0]
                    q1 = Target_Set[h][1][0]
                    q2 = Target_Set[h][0][1]
                    q3 = Target_Set[h][1][1]

                    
    
                    if q0 == low_1 and rr0 < low_1:
                        q0 = rr0
                        
                    if q1 == up_1 and rr1 > up_1:
                        q1 = rr1
                        
                    if q2 == low_2 and rr2 < low_2:
                        q2 = rr2
                        
                    if q3 == up_2 and rr3  > up_2:
                        q3 = rr3                      

                
                    
                    if (rr0 >= q1) or (rr1 <= q0) or (rr2 >= q3) or (rr3 <= q2) :
                        L_Bound_Matrix[z][j][h] = 0
                        U_Bound_Matrix[z][j][h] = 0
                        continue 
    
                    bisect.insort(Reachable_States[z][j],h)                                                                                               
                    
                    a1_Opt = ((q0 + q1)/2.0) - Mu1
                    a2_Opt = ((q2 + q3)/2.0) - Mu2

                                    
                    
                    if (r1 < a1_Opt): 
                        a1_Max = r1
                        a1_Min = r0
                    elif(r0 > a1_Opt): 
                        a1_Max = r0
                        a1_Min = r1
                    else: 
                        a1_Max = a1_Opt       
                        if (a1_Opt <= (r1+r0)/2.0):
                            a1_Min = r1
                        else:
                            a1_Min = r0
                                                            
                    if (r2 > a2_Opt): 
                        a2_Max = r2
                        a2_Min = r3
                    elif(r3 < a2_Opt): 
                        a2_Max = r3
                        a2_Min = r2
                    else: 
                        a2_Max = a2_Opt
                        if (a2_Opt <= (r2+r3)/2.0):
                            a2_Min = r3
                        else:
                            a2_Min = r2
                            
       
                                                                                               
                
                    if a1_Max + Mu1 - semi_Width_1  > q0 and a1_Max + Mu1 + semi_Width_1 < q1 and a2_Max + Mu2 - semi_Width_2 > q2 and a2_Max + Mu2 + semi_Width_2 < q3 :
                        H = 1
                    else:
                        
                        b0 = max(a1_Max + Mu1 - semi_Width_1, q0)
                        b1 = min(a1_Max + Mu1 + semi_Width_1, q1)
                        b2 = max(a2_Max + Mu2 - semi_Width_2, q2)
                        b3 = min(a2_Max + Mu2 + semi_Width_2, q3)
                        
                                                         
                        
                        H = ( ( (erf((b1 - a1_Max - Mu1)/Sigma1)/sqr) - (erf((b0 - a1_Max - Mu1)/Sigma1)/sqr) ) / (Z1)) * (( (erf((b3 - a2_Max - Mu2)/Sigma2)/sqr) - (erf((b2 - a2_Max - Mu2)/Sigma2)/sqr) ) / ( Z2 )) 
                
                        if H > 1:
                            H = 1
                            
                        
                        
                    if (a1_Min + Mu1 + semi_Width_1 <= q0) or (a1_Min + Mu1 - semi_Width_1 >= q1) or (a2_Min + Mu2 + semi_Width_2 <= q2) or (a2_Min + Mu2 - semi_Width_2 >= q3) :
                        Is_Bridge_State[z][j] = 1
                        bisect.insort(Bridge_Transitions[z][j],h)                   
                        L_Bound_Matrix[z][j][h] = 0
                        U_Bound_Matrix[z][j][h] = H
                        continue
    
                    else:
                        
                        b0 = max(a1_Min + Mu1 - semi_Width_1, q0)
                        b1 = min(a1_Min + Mu1 + semi_Width_1, q1)
                        b2 = max(a2_Min + Mu2 - semi_Width_2, q2)
                        b3 = min(a2_Min + Mu2 + semi_Width_2, q3)
                        
                        
                                            
                    
              # Perform integration for lower bound probability of transition, based on Gaussian overlap      
                    L = ( ( (erf((b1 - a1_Min - Mu1)/Sigma1)/sqr) - (erf((b0 - a1_Min - Mu1)/Sigma1)/sqr) ) / (Z1)) * (( (erf((b3 - a2_Min - Mu2)/Sigma2)/sqr) - (erf((b2 - a2_Min - Mu2)/Sigma2)/sqr) ) / ( Z2 )) 
            
                
                    if L < 0:
                        L = 0
                        
                        
                    L_Bound_Matrix[z][j][h] = L
                    U_Bound_Matrix[z][j][h] = H
                    
                    
                              
                
    
    return (L_Bound_Matrix, U_Bound_Matrix, Reachable_States, Is_Bridge_State, Bridge_Transitions)





def Index_Update_Product(Leaky_States_P_Accepting, Copy_Leaky_States_P_Accepting, Set_Refinement, Index_Stop3, Len_Automata, m):

    for l in range(len(Leaky_States_P_Accepting)):
        Number_Added = 0
        Check = 0
        for k in range(Index_Stop3[l], len(Leaky_States_P_Accepting[l])):
            if Copy_Leaky_States_P_Accepting[l][k+Number_Added]/Len_Automata > Set_Refinement[m]:
                if Check == 0:
                    Index_Stop3[l] = k
                    Check = 1
                Leaky_States_P_Accepting[l][k+Number_Added] += Len_Automata
            elif Copy_Leaky_States_P_Accepting[l][k+Number_Added]/Len_Automata == Set_Refinement[m]:
                if Check == 0:
                    Index_Stop3[l] = k
                    Check = 1 
                    
                Copy_Leaky_States_P_Accepting[l].insert(k+Number_Added+1,Copy_Leaky_States_P_Accepting[l][k+Number_Added])                           
                Leaky_States_P_Accepting[l].insert(k+Number_Added+1, Leaky_States_P_Accepting[l][k+Number_Added] + Len_Automata)
                Number_Added += 1


    return Leaky_States_P_Accepting, Copy_Leaky_States_P_Accepting, Index_Stop3


def Psi(xx):
    return (0.5)*(1.0+erf(xx/sqrt(2.0)))


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
        y_low = ((a2 - y_max_s) - m2 )/ s2       
    
        
    return ((((0.5)*(1.0+erf(x_up/sqr)) - ((0.5)*(1.0+erf(alpha_x/sqr))))/((0.5)*(1.0+erf(beta_x/sqr)) - (0.5)*(1.0+erf(alpha_x/sqr)))) - (((0.5)*(1.0+erf(x_low/sqr)) - (0.5)*(1.0+erf(alpha_x/sqr)))/((0.5)*(1.0+erf(beta_x/sqr)) - (0.5)*(1.0+erf(alpha_x/sqr))))) * ((((0.5)*(1.0+erf(y_up/sqr)) - ((0.5)*(1.0+erf(alpha_y/sqr))))/((0.5)*(1.0+erf(beta_y/sqr)) - (0.5)*(1.0+erf(alpha_y/sqr)))) - (((0.5)*(1.0+erf(y_low/sqr)) - ((0.5)*(1.0+erf(alpha_y/sqr))))/((0.5)*(1.0+erf(beta_y/sqr)) - (0.5)*(1.0+erf(alpha_y/sqr)))))



def Upper_Bound_Func_Jac(inpx,inpy, a1,b1,a2,b2,l1,u1,l2,u2):
    
    #Computing the upper bound probability of transition for a given input

    m1 = mu1
    m2 = mu2
    s1 = sigma1
    s2 = sigma2
    sqr = sqrt(2.0)
    jac1 = 3.0 #random number
    jac2 = 3.0 #random number
         
    if (( ((a1+b1)/2.0) - m1) < u1+inpx) and (( ((a1+b1)/2.0) - m1) > l1+inpx):
        jac1 = 0.0
    if (( ((a2+b2)/2.0) - m2) < u2+inpy) and (( ((a2+b2)/2.0) - m2) > l2+inpy):
        jac2 = 0.0
        
    if int(jac1) == 0 and int(jac2) == 0:
        return [jac1, jac2]
    
    x_max_s = float(max(l1+inpx, min(u1+inpx, ( ((a1+b1)/2.0) - m1) ) )) #Computing the maximizing shifts
    y_max_s = float(max(l2+inpy, min(u2+inpy, ( ((a2+b2)/2.0) - m2) ) ))
 
    alpha_x = float((w1_low - m1))/s1
    beta_x = float((w1_up - m1))/s1 
    alpha_y = float((w2_low - m2))/s2
    beta_y = float((w2_up - m2))/s2       
        
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
        y_low = ((a2 - y_max_s) - m2 )/ s2

      
    t1 = (0.5)*(1.0+erf(alpha_x/sqr))
    t2 = (0.5)*(1.0+erf(alpha_y/sqr))
        
    Z1 = (0.5)*(1.0+erf(beta_x/sqr)) - t1
    Z2 = (0.5)*(1.0+erf(beta_y/sqr)) - t2          
     
    if int(jac1) > 0:
               
        term_1 = ( -(1.0/(sqrt(2.0*pi)))*exp(-0.5*((x_up)**2))  + (1.0/(sqrt(2.0*pi)))*exp(-0.5*((x_low)**2)) )/(s1*Z1)
        term_2 = (((0.5)*(1.0+erf(y_up/sqr)) - t2)/Z2) - (((0.5)*(1.0+erf(y_low/sqr)) - t2)/Z2)
                      
        jac1 = float(term_1*term_2)
        

    if int(jac2) > 0:
               
        term_1 = (((0.5)*(1.0+erf(x_up/sqr)) - t1)/Z1) - (((0.5)*(1.0+erf(x_low/sqr)) - t1)/Z1) 
        term_2 = ( -(1.0/(sqrt(2.0*pi)))*exp(-0.5*((y_up)**2))  + (1.0/(sqrt(2.0*pi)))*exp(-0.5*((y_low)**2)) )/(s2*Z2)
                      
        jac2 = float(term_1*term_2 )   
        
             
        
    return [jac1, jac2]


#NEED TO SET UP THE LOWER BOUND FUNCTION

def Lower_Bound_Func(inpx,inpy, a1,b1,a2,b2,l1,u1,l2,u2):
    
    #Computing the lower bound probability of transition for a given input

    m1 = mu1
    m2 = mu2
    s1 = sigma1
    s2 = sigma2
    sqr = sqrt(2.0)
    
    x_max_s = float(max(l1+inpx, min(u1+inpx, ( ((a1+b1)/2.0) - m1) ) )) #Computing the maximizing shifts
    y_max_s = float(max(l2+inpy, min(u2+inpy, ( ((a2+b2)/2.0) - m2) ) ))
    
    if x_max_s < ( ((l1+u1)/2.0) + inpx):
        x_min_s = u1+inpx
    else:
        x_min_s = l1+inpx
        
    if y_max_s < ( ((l2+u2)/2.0) + inpy):
        y_min_s = u2+inpy
    else:
        y_min_s = l2+inpy        
        
       
    alpha_x = (w1_low - m1)/s1
    beta_x = (w1_up - m1)/s1
    alpha_y = (w2_low - m2)/s2
    beta_y = (w2_up - m2)/s2
    
    if (b1 - x_min_s) < w1_low:
        x_up = alpha_x        
    elif(b1 - x_min_s) > w1_up:
        x_up = beta_x
    else:
        x_up = ((b1 - x_min_s) - m1)/s1
        
    if (a1 - x_min_s) < w1_low:
        x_low = alpha_x        
    elif(a1 - x_min_s) > w1_up:
        x_low = beta_x
    else:
        x_low = ((a1 - x_min_s) - m1)/s1

    if (b2 - y_min_s) < w2_low:
        y_up = alpha_y       
    elif(b2 - y_min_s) > w2_up:
        y_up = beta_y
    else:
        y_up = ((b2 - y_min_s)- m2)/s2
        
    if (a2 - y_min_s) < w2_low:
        y_low = alpha_y        
    elif(a2 - y_min_s) > w2_up:
        y_low = beta_y
    else:
        y_low = ((a2 - y_min_s) - m2 )/ s2       
    
    
    t1 = (0.5)*(1.0+erf(alpha_x/sqr))
    t2 = (0.5)*(1.0+erf(alpha_y/sqr))
    
    Z1 = (0.5)*(1.0+erf(beta_x/sqr)) - (0.5)*(1.0+erf(alpha_x/sqr))
    Z2 = (0.5)*(1.0+erf(beta_y/sqr)) - (0.5)*(1.0+erf(alpha_y/sqr))

    term_1 = (((0.5)*(1.0+erf(x_up/sqr)) - t1)/Z1) - (((0.5)*(1.0+erf(x_low/sqr)) - t1)/Z1)
    term_2 = (((0.5)*(1.0+erf(y_up/sqr)) - t2)/Z2) - (((0.5)*(1.0+erf(y_low/sqr)) - t2)/Z2)    
      
    return term_1*term_2


def Lower_Bound_Func_Jac(inpx,inpy, a1,b1,a2,b2,l1,u1,l2,u2):
    
    #Computing the upper bound probability of transition for a given input

    m1 = mu1
    m2 = mu2
    s1 = sigma1
    s2 = sigma2
    sqr = sqrt(2.0)
    jac1 = 3.0 #random number
    jac2= 3.0 #random number
         
    x_max_s = float(max(l1+inpx, min(u1+inpx, ( ((a1+b1)/2.0) - m1) ) )) #Computing the maximizing shifts
    y_max_s = float(max(l2+inpy, min(u2+inpy, ( ((a2+b2)/2.0) - m2) ) ))
    
    if x_max_s < ( ((l1+u1)/2.0) + inpx):
        x_min_s = u1+inpx
    else:
        x_min_s = l1+inpx
        
    if y_max_s < ( ((l2+u2)/2.0) + inpy):
        y_min_s = u2+inpy
    else:
        y_min_s = l2+inpy        
        
       
    alpha_x = (w1_low - m1)/s1
    beta_x = (w1_up - m1)/s1
    alpha_y = (w2_low - m2)/s2
    beta_y = (w2_up - m2)/s2
    
    if (b1 - x_min_s) < w1_low:
        x_up = alpha_x        
    elif(b1 - x_min_s) > w1_up:
        x_up = beta_x
    else:
        x_up = ((b1 - x_min_s) - m1)/s1
        
    if (a1 - x_min_s) < w1_low:
        x_low = alpha_x        
    elif(a1 - x_min_s) > w1_up:
        x_low = beta_x
    else:
        x_low = ((a1 - x_min_s) - m1)/s1

    if (b2 - y_min_s) < w2_low:
        y_up = alpha_y       
    elif(b2 - y_min_s) > w2_up:
        y_up = beta_y
    else:
        y_up = ((b2 - y_min_s)- m2)/s2
        
    if (a2 - y_min_s) < w2_low:
        y_low = alpha_y        
    elif(a2 - y_min_s) > w2_up:
        y_low = beta_y
    else:
        y_low = ((a2 - y_min_s) - m2 )/ s2      

      
    t1 = (0.5)*(1.0+erf(alpha_x/sqr))
    t2 = (0.5)*(1.0+erf(alpha_y/sqr))
        
    Z1 = (0.5)*(1.0+erf(beta_x/sqr)) - t1
    Z2 = (0.5)*(1.0+erf(beta_y/sqr)) - t2          
     
    if int(jac1) > 0:
               
        term_1 = ( -(1.0/(sqrt(2.0*pi)))*exp(-0.5*((x_up)**2))  + (1.0/(sqrt(2.0*pi)))*exp(-0.5*((x_low)**2)) )/(s1*Z1)
        term_2 = (((0.5)*(1.0+erf(y_up/sqr)) - t2)/Z2) - (((0.5)*(1.0+erf(y_low/sqr)) - t2)/Z2)
                      
        jac1 = term_1*term_2
        

    if int(jac2) > 0:
               
        term_1 = (((0.5)*(1.0+erf(x_up/sqr)) - t1)/Z1) - (((0.5)*(1.0+erf(x_low/sqr)) - t1)/Z1) 
        term_2 = ( -(1.0/(sqrt(2.0*pi)))*exp(-0.5*((y_up)**2))  + (1.0/(sqrt(2.0*pi)))*exp(-0.5*((y_low)**2)) )/(s2*Z2)
                      
        jac2 = term_1*term_2    

     
        
        
    return [jac1, jac2]


def Index_Update_Product_Continuous(Leaky_States_P_Accepting, Copy_Leaky_States_P_Accepting, Set_Refinement, Index_Stop3, Len_Automata, m):


        Number_Added = 0
        Check = 0
        for k in range(Index_Stop3, len(Leaky_States_P_Accepting)):
            if Copy_Leaky_States_P_Accepting[k+Number_Added]/Len_Automata > Set_Refinement[m]:
                if Check == 0:
                    Index_Stop3 = k
                    Check = 1
                Leaky_States_P_Accepting[k+Number_Added] += Len_Automata
            elif Copy_Leaky_States_P_Accepting[k+Number_Added]/Len_Automata == Set_Refinement[m]:
                if Check == 0:
                    Index_Stop3 = k
                    Check = 1 
                    
                Copy_Leaky_States_P_Accepting.insert(k+Number_Added+1,Copy_Leaky_States_P_Accepting[k+Number_Added])                           
                Leaky_States_P_Accepting.insert(k+Number_Added+1, Leaky_States_P_Accepting[k+Number_Added] + Len_Automata)
                Number_Added += 1


        return Leaky_States_P_Accepting, Copy_Leaky_States_P_Accepting, Index_Stop3

def dummy_function(yy):
    return 0.0

def dummy_jac(hhh):
    return [0.0,0.0]

def norm_u(u):
    return sqrt((u[0]**2) + (u[1]**2))

class MyBounds(object):
     def __init__(self, xmax=[1.1,1.1], xmin=[-1.1,-1.1] ):
         self.xmax = np.array(xmax)
         self.xmin = np.array(xmin)
     def __call__(self, **kwargs):
         x = kwargs["x_new"]
         tmax = bool(np.all(x <= self.xmax))
         tmin = bool(np.all(x >= self.xmin))
         return tmax and tmin
     
def State_Space_Plot(Space):
    
    
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    

    fig = plt.figure('Final Refined Partition')
    plt.title(r'Final Partition for $\phi_{1}$ Synthesis (Finite Inputs)', fontsize=18)

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
        
def dummy_hess(lll):
    return [[0.0,0.0],[0.0,0.0]]

