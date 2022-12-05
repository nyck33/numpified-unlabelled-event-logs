#!/usr/bin/env python
# coding: utf-8

# In[83]:

import sys
import math
from operator import itemgetter
import numpy as np
from numba import jit


# In[96]:
def printArray(arr, name='arr'):
    print(name, '\n')
    print(np.array2string(arr, separator=","))

## M, symbols, ints, symbols: ints
# printmodel(M, D, newD, Ddict)
def printmodel(T, D, newD, Ddict):
    print(' '.ljust(5), end=' ') # tab in 5 spaces from left
    for a in D: #print top column names
        print(a.ljust(5), end=' ')
    print()
    for a in D:
        print(a.ljust(5), end=' ') #print row names
        for b in D: #print cols of row, - if 0.0, else the probability
            aIdx = Ddict[a]
            bIdx = Ddict[b]
            if T[aIdx,bIdx] == 0.0:
                print('-'.ljust(5), end=' ')
            else:
                print('{0:.2f}'.format(T[aIdx,bIdx]).ljust(5), end=' ')
        print()


# In[8]:


# general routine for converting a sequence to string
@jit
def seq2str(seq):
	'''
	seq is a list whose elements are concatenated into
	string
	'''
	string = ''
	for elem in seq:
		string += str(elem)
	return string

# general routine for sorting a dictionary by values
def sortbyvalue(d):
	'''

	'''
	return sorted(iter(d.items()), key=itemgetter(1), reverse=True)


# In[9]:


def initializeD(x, BEGIN, END):
	'''
	D = [] # the set of symbols in x

	'''
	D = [BEGIN] + sorted(set(x)) + [END]

	return D


# In[10]:


def translateXToInts(x, Ddict):
    intsX = [0 for i in range(len(x))]
    for i in range(len(x)):
        intsX[i] = Ddict[x[i]]
    
    return intsX


# In[11]:


def initializeGM(D):
	numSymbols = len(D)
	gM = np.zeros((numSymbols, numSymbols), dtype=np.float32)
	return gM


# In[12]:


def buildGM(x, gM, N):
    '''
    immmutable newGM
    #adjust probs in nested dict counting occurrences 
    # in symbol sequence x
    gM is zeros array
    N is len x

    '''
    print('x: ', x)
    print('N: ', N)

    newGM = np.zeros(np.shape(gM))

    for n in range(0, N-1):
        a = x[n]
        b = x[n+1]
        gM[a,b] += 1.0
        #newGM[a,b] = gM[a,b] + 1.0 # for checking

    printArray(gM, 'newGM')
    return gM
    #return newGM


# In[100]:


def normalize(d):
	'''
	D = [] # the set of symbols in x
	d is a dict whose values get normalized
	{char: double}
	'''
	rowsum = 0.0
	for k in d.keys():
		rowsum = rowsum + d[k]
	if rowsum > 0.0:
		for k in d.keys():
			d[k] = d[k] / rowsum

	return d


# In[107]:


def normalizeGM(gM):
    '''
    newGM[key] = {char: double} normalized
    '''
    newGM = np.zeros(np.shape(gM))
    #sum cols
    rowsum = (np.sum(gM, axis=1)[np.newaxis]) #.T
    #print(f'rowsum: {rowsum}')
    _,cols = np.shape(rowsum)
    #print(f'cols:{cols}')
    
    #rowsumStack = np.column_stack([rowsum for i in range(np.shape(rowsum)[1])])
    for i in range(cols):
        if(rowsum[0,i] > 0):
            newGM[i] = gM[i] / rowsum[0,i]
    
    #np.place(newGM, rowsumStack>0, gM/rowsumStack)

    printArray(newGM, "normGM")
    

    return newGM


# # Estimate

# In[117]:


# expectation-maximization procedure to estimate s and M iteratively (algorithm 2 in the paper)
def estimate(x, s, gM, M, D, y, N, BEGIN, END):
    print('start estimate\n')
    print('x: ', x)
    print('s: ', s)
    print('gM: ', gM)
    print('N: ', M)
    print('D: ', D)
    print('y: ', y)
    print('N: ', N)
    print('BEGIN', BEGIN)
    print('END: ', END)

    

    ###########################################################################################
    prevsseqs = []
    print('Initializing source sequence...')

    #gM = param T
    s, y = estsources(x, D, N, gM, BEGIN, END) # start with an estimate of s computed from the global model gM
    its = 0
    while s not in prevsseqs:
        its += 1
        print('#{0}: Estimating parameters...'.format(its))
        M = estparams(D, y, BEGIN, END) # update transition matrix M
        prevsseqs.append(s[:])
        print('#{0}: Computing source sequence...'.format(its))
        s, y = estsources(x, D, N, M, BEGIN, END) # use current M to re-estimate s
    print('done estimate\n')
    print(f's: \n{s}\n')
    print(f'y: \n{y}')
   
    return len(set(s)), M, y


# In[122]:


def estsources(x, D, N, T, BEGIN, END):
    '''
    x = list of sequence symbols
    D = list of sequence symbols
    D needs to be the int version of symbols for T indexing
    N = len(x)
    T = M a len(D) * len(D) nparray
    
    '''
    s = []
    # list of lists
    y = []
    #y.append([])
    # set but now list use if x in active
    active = []
    for n in range(0,N):
        #print(f'top leny: {len(y)}\n')

        xn = x[n]
        pmax = 0.0
        sn = -1
        for k in active:
            if xn in y[k]:
                continue
            a = y[k][-1]
            b = xn
            p = T[a,b]
            if p > pmax:
                sn = k
                pmax = p
        if sn == -1 or T[BEGIN, xn] > pmax:
            # making y longer by one subarray
            if len(y) == 0:
                sn = len(y) + 1
            else:
                sn = len(y)
            
            if sn not in active:
                active.append(sn)
            if sn == 1:
                y.append([])
                y.append([])
            else:
                y.append([])
        #print(f'bottom leny: {len(y)}\n')
        #print(f'sn: {sn}\n')

        s.append(sn)
        #print(f'y:\n{y}\n')
        y[sn].append(xn)
        pnext = 0.0
        bnext = BEGIN
        # need to enumerate D
        for b in D:
            Txnb = T[xn,b]
            if Txnb > pnext:
                pnext = T[xn][b]
                bnext = b
        if bnext == END:
            active = [x for x in active if x != sn]
        print(f's: {s}\n')
        print(f'y: {y}\n')

    #take out the empty list in y
    y = [x for x in y if x != []]
        
    return s, y
            
    


# In[91]:


def estparams(D,y, BEGIN, END):
    M = np.zeros((len(D), len(D)), dtype=np.float32)
    #iter sublists
    for k in range(len(y)):
        a = BEGIN
        b = y[k][0]
        M[a,b] += 1.0
        for r in range(0, len(y[k])-1):
            a = y[k][r]
            b = y[k][r+1]
            M[a,b] += 1.0
        a = y[k][-1]
        b = END
        M[a,b] += 1.0
    
    normM = normalizeGM(M)
    print(f'M: {normM}')

    return normM


# In[ ]:





# In[ ]:





# In[ ]:





# In[120]:


def seqprobs(y):
    print(f'y:{y}')
    probs = dict()
    #probs = []
    #probsKeys = []
    '''
    for k in range(len(y)):
        z = seqstr(y[k])
        if z not in probsKeys:
            probsKeys.append(z)
    '''
    for k in range(len(y)):
        z = seq2str(np.array(y[k]))
        #print(f'z:{z}')
        if z in probs:
            probs[z] += 1.0
        else:
            probs[z] = 1.0
    #print(f'raw probs: {probs}')

    normalize(probs)
    return probs



# In[99]:


# checks that it is possible to recover the symbol sequence x from the separate sequences y (sanity check)
def checkmodel(s, x):
    x2 = []
    pos = dict()
    for k in y:
        pos[k] = -1
    for n in range(len(s)):
        sn = s[n]
        pos[sn] += 1
        xn = y[sn][pos[sn]]
        x2.append(xn)
    return x2 == x


# In[123]:


if __name__=="__main__":
    
    BEGIN = 'o'
    END = 'x'
    # read symbol sequence x from stdin, with one symbol per line
    x = []

    #filename = "mocksequence.txt"
    filename = "sequence.txt"

    with open(filename, mode="r") as f:
        lines = f.readlines()
        for line in lines:
            symbol = line.strip()
            if len(symbol) > 0:
                x += [symbol]
    

    # print the sequence as string
    print('raw x: ', x)
    print("Symbol sequence: ", seq2str(x))

    print("({0} symbols)".format(len(x)))

    N = len(x)
    #################################################################
    # call init funcs
    # list
    D = initializeD(x, BEGIN, END)
    print(f'D: {D}')
    #dictionary of alpha: integer index into array
    Ddict = {D[i]:i for i in range(len(D))}
    revDdict = {i:D[i] for i in range(len(D))}
    #translate x sequence into integers
    newX = translateXToInts(x, Ddict)
    print(f'newX: {newX}')
    
    #translate D to newD ints
    newD = translateXToInts(D, Ddict)
    print(f'newD: {newD}')

    print(f'Ddict: {Ddict}')
    # only uses len of D so okay
    gM = initializeGM(D)
    print(f'gM:\n {gM}')
    gM = buildGM(newX,gM, len(x))
    print(f'gM:\n {gM}')
    gM = normalizeGM(gM)
    print(f'gM:\n {gM}')
    
    M = np.zeros(np.shape(gM))
    s = [] # the source sequence s (to be determined)
    
    # max number of symbols in any value list is len(D)
    #{int: [unique symbols]}, use list of lists
    #y = dict() # the separate source sequences (y^{(k)} in the paper)
    y = []
    # estimate model, with all above member variables
    
    BEGIN = Ddict['o']
    END = Ddict['x']

    # x is ints, D is ints, y is ints, gM and M keys are ints
    # 

    print("Calling Estimate:\n")
    print(f'X: {x}\n')
    print(f'newX: {newX}\n')
    print(f's: {s}\n')
    print(f'gM: {gM}\n')
    print(f'M:{M}\n')
    print(f'D: {D}\n')
    print(f'newD:{newD}\n')
    print(f'y:{y}\n')
    print(f'N:{N}\n')

    K, M, y = estimate(newX, s, gM, M, newD, y, N, BEGIN, END)
    
    #translate y back to chars
    newY = []
    for i in range(len(y)):
        subarr = y[i]
        newsub = []
        for num in subarr:
            newsub.append(revDdict[num])
        newY.append(newsub)

    #modelCorrect = m.checkmodel()

    #print(f'model is ok: {modelCorrect}')

    # print model
    # M, symbols, ints, symbols: ints
    printmodel(M, D, newD, Ddict)

    # show the probability distribution of the different sequences in the model
    probs = seqprobs(newY)
    print(f'probs:\n{probs}\n')
    pz = sortbyvalue(probs)
    for z, p in pz:
        print('{0:.3f} : {1}'.format(p, z))

    print('Total number of sources: {0}'.format(K))

    #####################################


# # Test Cell

# In[60]:





