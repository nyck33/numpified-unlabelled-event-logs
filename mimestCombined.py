import sys
import math
from operator import itemgetter

def findDiff(d1, d2, path=""):
    numKeysD1 = len(d1.keys())
    numKeysD2 = len(d2.keys())

    assert numKeysD1 == numKeysD2, f"d1: {numKeysD1} != d2: {numKeysD2}"
    wrongCount = 0
    for k in d1:
        if k in d2:
            if type(d1[k]) is dict:
                findDiff(d1[k],d2[k], "%s -> %s" % (path, k) if path else k)
            if d1[k] != d2[k]:
                result = [ "%s: " % path, " - %s : %s" % (k, d1[k]) , " + %s : %s" % (k, d2[k])]
                print("\n".join(result))
        else:
            print ("%s%s as key not in d2\n" % ("%s: " % path if path else "", k))
            wrongCount += 1
    
    return wrongCount

BEGIN = 'o'
	
END = 'x'

# general routine for normalizing probability distributions
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

# general routine for converting a sequence to string
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

# routine for computing the G-metric between two MIM models
def gmetric(m1, m2):
	pz = m1.seqprobs()
	qz = m2.seqprobs()
	g = 0.0
	for z in pz.keys():
		if z in qz:
			g += math.sqrt(pz[z]*qz[z])
	return g
	
#################################################################################
#################################################################################
#################################################################################
# instead of class Model call these in order
def initializeD(x):
	'''
	D = [] # the set of symbols in x

	'''
	D = [BEGIN] + sorted(set(x)) + [END]

	return D

def initializeGM(D):
	gM = dict()
	for a in D:
		gM[a] = dict()
		for b in D:
			gM[a][b] = 0.0
	
	return gM

def buildGM(x, gM, N):
	'''
	immmutable newGM
	#adjust probs in nested dict counting occurrences 
	# in symbol sequence x

	'''
	newGM = dict()

	for i in range(0, N-1):
		newGM[i] = dict()

	for n in range(0, N-1):
		a = x[n]
		b = x[n+1]
		gM[a][b] += 1.0 # for checking
		#newGM[a][b] = gM[a][b] + 1.0

	#diffCount = findDiff(gM, newGM)
	#assert diffCount == 0, f"diffCount: {diffCount}"
	#return newGM
	return gM

def normalizeGM(D, gM):
	'''
	newGM[key] = {char: double} normalized
	'''
	newGM = dict()
	for a in D:
		newGM[a] = normalize(gM[a])

	return newGM
	
#################################################################################
#################################################################################
#################################################################################
# print a given transition matrix T
def printmodel(T, D):
	print(' '.ljust(5), end=' ') # tab in 5 spaces from left
	for a in D: #print top column names
		print(a.ljust(5), end=' ')
	print()
	for a in D:
		print(a.ljust(5), end=' ') #print row names
		for b in D: #print cols of row, - if 0.0, else the probability
			if T[a][b] == 0.0:
				print('-'.ljust(5), end=' ')
			else:
				print('{0:.2f}'.format(T[a][b]).ljust(5), end=' ')
		print()


# expectation-maximization procedure to estimate s and M iteratively (algorithm 2 in the paper)
def estimate(x, s, gM, M, D, y, N):
	
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

	prevsseqs = []
	print('Initializing source sequence...')
    #gM = param T
	s, y = estsources(x, D, N, gM) # start with an estimate of s computed from the global model gM
	its = 0
	while s not in prevsseqs:
		its += 1
		print('#{0}: Estimating parameters...'.format(its))
		M = estparams(D, y) # update transition matrix M
		prevsseqs.append(s[:])
		print('#{0}: Computing source sequence...'.format(its))
		s, y = estsources(x, D, N, M) # use current M to re-estimate s
	print('done estimate\n')
	print(f's: \n{s}\n')
	print(f'y: \n{y}')

	return len(set(s)), M

# estimate the source sequence s from a given transition matrix T (algorithm 1 in the paper)
def estsources(x, D, N, T):
	'''
	T=M
	x = [] # the symbol sequence

	'''
	s = []
	y = dict()
	active = set()
	for n in range(0,N):
		# xn is char
		xn = x[n]
		pmax = 0.0
		sn = -1
		# iterate empty set?
		for k in active:
			#print(f'k : {k}')
			#print(f'y[k] : {y[k]}')
			if xn in y[k]:
				continue
			#print(f'y[k] : {y[k]}')
			a = y[k][-1]
			b = xn
			p = T[a][b]
			if p > pmax:
				sn = k
				pmax = p
		if sn == -1 or T[BEGIN][xn] > pmax:
			sn = len(y) + 1
			active.add(sn)
			y[sn] = []
		s.append(sn)
		y[sn].append(xn)
		pnext = 0.0
		bnext = BEGIN
		for b in D:
			Txnb = T[xn][b]
			if Txnb > pnext:
				pnext = T[xn][b]
				bnext = b
		if bnext == END:
			active.remove(sn)
		print(f's: {s}\n')
		print(f'y: {y}\n')
	return s, y

# update the transition matrix M based on the current separate source sequences y
def estparams(D, y):
	'''
	new M normalized returned
	'''
	M = dict()
	for a in D:
		M[a] = dict()
		for b in D:
			M[a][b] = 0.0
	for k in y.keys():
		a = BEGIN
		b = y[k][0]
		M[a][b] += 1.0
		for r in range(0,len(y[k])-1):
			a = y[k][r]
			b = y[k][r+1]
			M[a][b] += 1.0
		a = y[k][-1]
		b = END
		M[a][b] += 1.0
	
	# immutable
	normM = dict()

	for a in D:
		normM[a] = normalize(M[a])

	return normM




# computes the probability distribution for the different sequences produced by this model (p(z) or q(z) in the paper)
def seqprobs(y):
	probs = dict()
	for k in y.keys():
		z = seq2str(y[k])
		if z in probs:
			probs[z] += 1.0
		else:
			probs[z] = 1.0
	normalize(probs)
	return probs

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


###############################################################################


if __name__=="__main__":
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
	print("Symbol sequence: ", seq2str(x))

	print("({0} symbols)".format(len(x)))

	N = len(x)
	D = [] # the set of symbols in x
	# nested {state: {nextstate: prob, nextstate2: prob2,...}}
	# char: {char : double}}
	gM = dict() # the global model used to initialize M (M^{+} in the paper)

	M = dict() # the transition matrix M
	
	s = [] # the source sequence s (to be determined)
	
	y = dict() # the separate source sequences (y^{(k)} in the paper)

	#################################################################
	# call init funcs
	D = initializeD(x)
	gM = initializeGM(D)
	gM = buildGM(x, gM, N)
	gM = normalizeGM(D, gM)

	#every function needs a return value
	# estimate model, with all above member variables

	K, M = estimate(x, s, gM, M, D, y, N)

	#modelCorrect = m.checkmodel()

	#print(f'model is ok: {modelCorrect}')

	# print model
	printmodel(M, D)

	# show the probability distribution of the different sequences in the model
	pz = sortbyvalue(seqprobs(y))
	for z, p in pz:
		print('{0:.3f} : {1}'.format(p, z))

	print('Total number of sources: {0}'.format(K))

	#####################################



