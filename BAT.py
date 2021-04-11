# -*- coding: utf-8 -*-
"""
author: Matheus S. Quessada
email: matheus.quessada@unesp.br
"""
import math
import numpy
import random
import time

def euclidiana(y,z):
    return (numpy.sqrt(sum((y - z) ** 2))) 


def F10(x):
    dim = len(x)
    o = (
        -20 * numpy.exp(-0.2 * numpy.sqrt(numpy.sum(x ** 2) / dim))
        - numpy.exp(numpy.sum(numpy.cos(2 * math.pi * x)) / dim)
        + 20
        + numpy.exp(1)
    )
    return o

def bat_solution(sw, Serv_List, MaxIter):
    Fog_nro = sw
    dim = 2
    n = dim

    Nro_serv = len(Serv_List)

    A = 0.5
    # Loudness  (constant or decreasing)
    r = 0.5
    # Pulse rate (constant or decreasing)

    Qmin = 0  # Frequency minimum
    Qmax = 2  # Frequency maximum

    d = dim  # Number of dimensions

    # Initializing arrays
    Q = numpy.zeros(Nro_serv)  # Frequency
    v = numpy.zeros((Nro_serv, dim))  # Velocities
    Convergence_curve = []

    # Initialize the population/solutions
    Sol = numpy.zeros((Nro_serv, dim))
    for i in range (0, dim):
        for j in range(0, Nro_serv):
            y = numpy.array([Serv_List[j][i+1]])
            z = numpy.array(sw)
            Sol[j, i] = euclidiana(y,z)
    S = numpy.zeros((Nro_serv, dim))

    S = numpy.copy(Sol)

    Fitness = numpy.zeros(Nro_serv)

    # Initialize timer for the experiment
    timerStart = time.time()


    # Evaluate initial random solutions
    for i in range(0, Nro_serv):
        Fitness[i] = F10(Sol[i, :])

    # Find the initial best solution and minimum fitness
    I = numpy.argmin(Fitness)
    best = Sol[I, :]
    fmin = min(Fitness)
    Best = I

    # Main loop
    for t in range(0, MaxIter): #loop de 0 até numero de iterações
        # Loop over all bats(solutions)
        for i in range(0, Nro_serv): #loop de 0 até número de população
            Q[i] = Qmin + (Qmin - Qmax) * random.random()
            v[i, :] = v[i, :] + (Sol[i, :] - best) * Q[i]
            S[i, :] = Sol[i, :] + v[i, :]

            # Pulse rate
            if random.random() > r:
                S[i, :] = best + 0.001 * numpy.random.randn(d)

            # Evaluate new solutions
            Fnew = F10(S[i, :])

            # Update if the solution improves
            if (Fnew <= Fitness[i]) and (random.random() < A):
                Sol[i, :] = numpy.copy(S[i, :])
                Fitness[i] = Fnew
                Best = i              

            # Update the current best solution
            if Fnew <= fmin:
                best = numpy.copy(S[i, :])
                fmin = Fnew
                Best = i         
                #print("Best = ", Best)     

    print("Best antes do return: ", Best)
    return Best





def bat(W, T):
    MaxIter = len(T)
    vi = 0
    bloqueio = 0
    g = 0
    gain_mors = 0
    alocadas = 0
    nao_alocadas = 0
    cont_peso = 0    
    resto = []
    for p in range (0,len(T)):
        T[p][0] = g
        g += 1
    
    VCC=[]
    for j,k in W.items():
        VCC.append(k)

    t1=[]
    g=0
    VCC = sorted(VCC, reverse=True)    
    aloc=[]
    for x in range(0,len(VCC)):
        vi = VCC[x]
        for r in range (0,len(T)):
            T[r][0] = g
            g += 1
        for n in range(0,len(T)):
            t1.append(T[n])
        while (len(t1) > 0):
            if (vi > 0):
                g=0
                for q in range (0,len(t1)):
                    t1[q][0] = g
                    g += 1
                p = bat_solution(vi,t1, MaxIter)
                if ((vi - t1[p][1]) >= 0):
                    t = t1[p]
                    gain_mors += t[2]
                    cont_peso += t[1]
                    alocadas += 1
                    vi -= t[1]
                    aloc = aloc + [t]
                    
                else:
                    bloqueio += 1
                del t1[p]
            else:
                for x in t1:
                    del t1[0]
            
        for x in range(0,len(aloc)):
            for y in range (0,len(T)):
                if (aloc[0][1]==T[y][1] and aloc[0][2]==T[y][2]):
                    del T[y]
                    del aloc[0]
                    break
    return (gain_mors, alocadas, cont_peso)
