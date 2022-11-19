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
    

    

    K,M,y = estimate(startIntsX, s, gmStartArr, gmStartArr, 
                                startIntsD, y, N, 0, 9)

    
    assert K == KSolution
    assert np.allclose(M, MSolArr)

    assert sublistsEqual(YSolution, y)







