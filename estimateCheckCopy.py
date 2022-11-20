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

