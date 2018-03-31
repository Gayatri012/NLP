# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 17:50:12 2018

@author: Gayatri
"""

import sys
import pandas as pd

# Viterbi Algorithm to return the best hidden sequence
def viterbi(inputSequence, A, B, states):
    inputList = list(inputSequence)
    Tlen = len(inputList)
    v = {}
    
    #Initialization
    for s in states:
        v[s, 0] =  A.loc['Start', s] * B.loc[s, inputList[0]]
    
    #Recursion
    for t in range(1, Tlen):
        for s in states:
            v[s, t] = max(v['R', t-1] * A.loc['R', s] * B.loc[s, inputList[t]], v['S', t-1] * A.loc['S',s] * B.loc[s, inputList[t]]) 
       
    output = ''   
    #Final observation 
    if v['R', Tlen - 1] > v['S', Tlen - 1]:
        output = output + 'R'
    else:
        output = output + 'S'
        
    for i in reversed(range(1, Tlen)):
        if output[Tlen - i - 1] == 'R':
            if v['R',i-1] * A.loc['R','R'] > v['S',i-1] * A.loc['S','R']:
                output = output + 'R'
            else:
                output = output + 'S'
        else:
            if v['R',i-1] * A.loc['R','S'] > v['S',i-1] * A.loc['S','S']:
                output = output + 'R'
            else:
                output = output + 'S'
    
    return output
    

#A
transitionProbability = pd.DataFrame([[0.7, 0.3], [0.6, 0.4], [0.3, 0.7]], index=['Start','R', 'S'], columns=['R', 'S'])

#B
observationLikelihood = pd.DataFrame([[0.3, 0.35, 0.35], [0.5, 0.4, 0.1]], index=['R', 'S'], columns=['W', 'S', 'C'])

observation = sys.argv[1]

if (len(observation) < 1 or len(observation) > 10) :
    print("Input length exceeds the limit. Please enter input with length between 1-10")
else:
    weather = viterbi(list(sys.argv[1]), transitionProbability, observationLikelihood, ['R','S'])
    print("\nObservation (Input) : ", observation)
    print("W-Walk, C-Clean, S-Shop\n")
    print("Weather (Output) : ", weather[::-1]) #Reversing the weather to get output in correct sequence 
    print("R-Rainy, S-Sunny")
