from Numpified import *
from testVars import *

import numpy as np
import pandas as pd

def dictToNPArray(d):
        df = pd.DataFrame.from_dict(d)
        npArr = df.to_numpy().astype('float32')

        return npArr

def dictToListofLists(d):
    arr = []
    for k,v in d.items():
        arr.append(v)

    return arr

def sublistsEqual(a, b):
    return all(l for l in b if l in a)


#class TestClass:   

def test_initializeD():

    begin = 0
    end = len(D) -1

    BEGIN = 'o'
    END = 'x'

    assert initializeD(startIntsX , 0, 9) == startIntsD

def test_translateXToInts():

    assert translateXToInts(startX, Ddict) == startIntsX

def test_initializeGM():
    '''
    returned GM is a dictionary
    '''

    initGmD = initializeGM(startIntsD)
    print('testGM:\n ', initGmD) # dict
    resArr = dictToNPArray(initGmD) #npArr
    print('resArr: \n', resArr) 
    
    assert np.array_equal(resArr, zerosGMArr)
    #np.testing.assert_equal(testGM, initGM)


def test_buildGM():
    solArr = np.array(
        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,5.,0.,8.,1.,1.,4.,1.,0.,0.],
        [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
        [0.,2.,0.,4.,9.,1.,3.,0.,0.,0.],
        [0.,3.,0.,4.,4.,4.,4.,0.,0.,0.],
        [0.,2.,0.,1.,2.,1.,4.,0.,0.,0.],
        [0.,6.,1.,1.,3.,2.,1.,1.,1.,0.],
        [0.,0.,0.,1.,0.,0.,0.,0.,1.,0.],
        [0.,1.,0.,0.,0.,1.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]
    )

    gMQuery = buildGM(startIntsX, zerosGMArr, len(startX))

    print('gmQuery:\n ', gMQuery)

    assert np.array_equal(solArr, gMQuery)

    normGM = normalizeGM(gMQuery)

    print(normGM)

    normSol = np.array(
        [[0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
        0.        ,0.        ,0.        ,0.        ],
        [0.        ,0.25      ,0.        ,0.4       ,0.05      ,0.05      ,
        0.2       ,0.05      ,0.        ,0.        ],
        [0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
        1.        ,0.        ,0.        ,0.        ],
        [0.        ,0.10526316,0.        ,0.21052632,0.47368421,0.05263158,
        0.15789474,0.        ,0.        ,0.        ],
        [0.        ,0.15789474,0.        ,0.21052632,0.21052632,0.21052632,
        0.21052632,0.        ,0.        ,0.        ],
        [0.        ,0.2       ,0.        ,0.1       ,0.2       ,0.1       ,
        0.4       ,0.        ,0.        ,0.        ],
        [0.        ,0.375     ,0.0625    ,0.0625    ,0.1875    ,0.125     ,
        0.0625    ,0.0625    ,0.0625    ,0.        ],
        [0.        ,0.        ,0.        ,0.5       ,0.        ,0.        ,
        0.        ,0.        ,0.5       ,0.        ],
        [0.        ,0.5       ,0.        ,0.        ,0.        ,0.5       ,
        0.        ,0.        ,0.        ,0.        ],
        [0.        ,0.        ,0.        ,0.        ,0.        ,0.        ,
        0.        ,0.        ,0.        ,0.        ]])
    #gmNormSol = dictToNPArray(normalizeGMDict)
    
    assert np.allclose(normGM, normSol)


def test_estimate_start():
    '''
    call everything in main and check inputs to estimate
    '''
        
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
    '''
    most number of entries in y is going to be the number of permutations of the set of symbols D
    https://math.stackexchange.com/a/1193956/635903
    2^n - 1

    '''
    numSymbols = len(D)
    numSetsFromD = (2**numSymbols) - 1
    print('num sets from symbols: ', numSetsFromD)

    #y = []
    y = np.full((numSetsFromD, len(x)), -9, dtype=np.int)

    # estimate model, with all above member variables
    
    BEGIN = Ddict['o']
    END = Ddict['x']

    # x is ints, D is ints, y is ints, gM and M keys are ints
    # 
    M = np.zeros(np.shape(gM))

    '''
    gM, M, D, x = estimate(newX, s, gM, M, newD, y, N, BEGIN, END)

    
    assert np.allclose(gM, gmStartArr)
    assert np.shape(gM) == np.shape(gmStartArr)
    
    assert np.shape(M) == np.shape(gmStartArr)

    printArray(M, "M")

    assert np.allclose(M, MStart, atol=1e-03)
    assert D == startIntsD
    assert x == startIntsX
    '''
    

def test_estsources():


    s, y, yColIdxs = estsources(startIntsX, startIntsD, N, gmStartArr, 0, 9)
    print("s:\n", s)
    assert np.allclose(s, sArr1)

    print("y:\n", y)
    assert np.allclose(y, yArr1)






def test_estimate():
    
    #K = 23, 
    BEGIN = 'o'
    END = 'x'
    s = []

    y =[]

    ################################################
    KSolution = 23
    YSolution = YSolutionArr
    MSol = MSolArr
    
    #def estimate(x, s, gM, M, D, y, N, BEGIN, END):
    K,M,y, ylastIdxArr = estimate(startIntsX, s, gmStartArr, gmStartArr, 
                                startIntsD, y, N, 0, 9)

    
    #assert K == KSolution

    print('M:\n')
    print(np.array2string(M, separator=","))

    assert np.allclose(M, MSolArr)

    assert sublistsEqual(YSolution, y)







