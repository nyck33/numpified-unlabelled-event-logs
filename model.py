import sys
import math
from operator import itemgetter
from mimestCombined import *

class model:

	BEGIN = 'o'
	
	END = 'x'

	x = [] # the symbol sequence
	
	N = 0 # the length of x
	
	D = [] # the set of symbols in x

	gM = dict() # the global model used to initialize M (M^{+} in the paper)

	M = dict() # the transition matrix M
	
	s = [] # the source sequence s (to be determined)
	
	y = dict() # the separate source sequences (y^{(k)} in the paper)

	# class constructor initializes the global model gM
	def __init__(self, x):
		self.x = x
		self.N = len(self.x)
		# build D
		self.D = [self.BEGIN] + sorted(set(self.x)) + [self.END]
		# init gM
		for a in self.D:
			self.gM[a] = dict()
			for b in self.D:
				self.gM[a][b] = 0.0
		# build gM
		for n in range(0,self.N-1):
			a = self.x[n]
			b = self.x[n+1]
			self.gM[a][b] += 1.0
		# normalize nested dict of gM
		for a in self.D:
			#normalize the nested dict's probabilities
			normalize(self.gM[a])


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

if __name__=="__main__":
	# read symbol sequence x from stdin, with one symbol per line
	x = []

	filename = "mocksequence.txt"
	#filename = "sequence.txt"

	with open(filename, mode="r") as f:
		lines = f.readlines()
		for line in lines:
			symbol = line.strip()
			if len(symbol) > 0:
				x += [symbol]

	N = len(x)

	'''
	for line in sys.stdin:
		symbol = line.strip()
		if len(symbol) > 0:
			x += [symbol]
	'''
	# print the sequence as string
	print("Symbol sequence: ", seq2str(x))

	print("({0} symbols)".format(len(x)))

	#################################################################
	# call init funcs
	D = initializeD(x)
	gM = initializeGM(D)
	gM = buildGM(x, gM, N)
	gM = normalizeGM(D, gM)

	#every function needs a return value
	# estimate model

	s = []
	M = dict()
	y = dict()

	K = estimate(s, gM, M, D, y, N)

	#modelCorrect = m.checkmodel()

	#print(f'model is ok: {modelCorrect}')

	# print model
	printmodel(M)

	# show the probability distribution of the different sequences in the model
	pz = sortbyvalue(seqprobs(y))
	for z, p in pz:
		print('{0:.3f} : {1}'.format(p, z))

	print('Total number of sources: {0}'.format(K))

	#####################################



