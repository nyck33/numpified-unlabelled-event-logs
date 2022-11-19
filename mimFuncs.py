import math

# general routine for normalizing probability distributions
def normalize(d):
	rowsum = 0.0
	for k in d.keys():
		rowsum = rowsum + d[k]
	if rowsum > 0.0:
		for k in d.keys():
			d[k] = d[k] / rowsum

# general routine for converting a sequence to string
def seq2str(seq):
	string = ''
	for elem in seq:
		string += str(elem)
	return string

# general routine for sorting a dictionary by values
from operator import itemgetter
def sortbyvalue(d):
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
	
class model:

	BEGIN = 'o'
	
	END = 'x'

	x = [] # the symbol sequence
	
	N = 0 # the length of x
	
	D = [] # the set of symbols in x
	# nested {state: {nextstate: prob, nextstate2: prob2,...}}
	# char: {char : double}}
	gM = dict() # the global model used to initialize M (M^{+} in the paper)

	M = dict() # the transition matrix M
	
	s = [] # the source sequence s (to be determined)
	
	y = dict() # the separate source sequences (y^{(k)} in the paper)

	# class constructor initializes the global model gM
	def __init__(self, x):
		# symbol sequence one per line alpha
		self.x = x
		self.N = len(self.x)
		# sort and put the begin and end symbols on both ends
		self.D = [self.BEGIN] + sorted(set(self.x)) + [self.END]
		# for symbol in symbol sequence
		# make a nested dictionary of probabilities of transitioning from 
		# the key state to each other state, init to 0.0
		for a in self.D:
			self.gM[a] = dict()
			for b in self.D:
				self.gM[a][b] = 0.0
		#adjust probs in nested dict
		for n in range(0,self.N-1):
			a = self.x[n]
			b = self.x[n+1]
			self.gM[a][b] += 1.0
		#normalize probs in nested dict
		for a in self.D:
			normalize(self.gM[a])

	# print a given transition matrix T
	def printmodel(self, T):
		print(' '.ljust(5), end=' ')
		for a in self.D:
			print(a.ljust(5), end=' ')
		print()
		for a in self.D:
			print(a.ljust(5), end=' ')
			for b in self.D:
				if T[a][b] == 0.0:
					print('-'.ljust(5), end=' ')
				else:
					print('{0:.2f}'.format(T[a][b]).ljust(5), end=' ')
			print()

	# estimate the source sequence s from a given transition matrix T (algorithm 1 in the paper)
	def estsources(self, T):
		self.s = []
		#separate source sequence s (TBD)
		self.y = dict()
		active = set()
		for n in range(0,self.N):
			xn = self.x[n]
			pmax = 0.0
			sn = -1
			for k in active:
				if xn in self.y[k]:
					continue
				# self.y is a dict of symbols to lists of something
				a = self.y[k][-1]
				b = xn
				p = T[a][b]
				if p > pmax:
					sn = k
					pmax = p
			if sn == -1 or T[self.BEGIN][xn] > pmax:
				sn = len(self.y) + 1
				active.add(sn)
				self.y[sn] = []
			self.s.append(sn)
			self.y[sn].append(xn)
			pnext = 0.0
			bnext = self.BEGIN
			for b in self.D:
				if T[xn][b] > pnext:
					pnext = T[xn][b]
					bnext = b
			if bnext == self.END:
				active.remove(sn)

	# update the transition matrix M based on the current separate source sequences y
	def estparams(self):
		self.M = dict()
		for a in self.D:
			self.M[a] = dict()
			for b in self.D:
				self.M[a][b] = 0.0
		for k in self.y.keys():
			a = self.BEGIN
			b = self.y[k][0]
			self.M[a][b] += 1.0
			for r in range(0,len(self.y[k])-1):
				a = self.y[k][r]
				b = self.y[k][r+1]
				self.M[a][b] += 1.0
			a = self.y[k][-1]
			b = self.END
			self.M[a][b] += 1.0
		for a in self.D:
			normalize(self.M[a])

	# expectation-maximization procedure to estimate s and M iteratively (algorithm 2 in the paper)
	def estimate(self):
		prevsseqs = []
		print('Initializing source sequence...')
		self.estsources(self.gM) # start with an estimate of s computed from the global model gM
		its = 0
		while self.s not in prevsseqs:
			its += 1
			print('#{0}: Estimating parameters...'.format(its))
			self.estparams() # update transition matrix M
			prevsseqs.append(self.s[:])
			print('#{0}: Computing source sequence...'.format(its))
			self.estsources(self.M) # use current M to re-estimate s
		return len(set(self.s))

	# computes the probability distribution for the different sequences produced by this model (p(z) or q(z) in the paper)
	def seqprobs(self):
		probs = dict()
		for k in self.y.keys():
			z = seq2str(self.y[k])
			if z in probs:
				probs[z] += 1.0
			else:
				probs[z] = 1.0
		normalize(probs)
		return probs

	# checks that it is possible to recover the symbol sequence x from the separate sequences y (sanity check)
	def checkmodel(self):
		x2 = []
		pos = dict()
		for k in self.y:
			pos[k] = -1
		for n in range(len(self.s)):
			sn = self.s[n]
			pos[sn] += 1
			xn = self.y[sn][pos[sn]]
			x2.append(xn)
		return x2 == self.x

###########################################################################
#mimest.py < sequence.txt
import sys
import mim

# read symbol sequence x from stdin, with one symbol per line
x = []
'''
with open("sequence.txt", mode="r") as file:
	for line in file.readline():
		symbol = line.strip()
		if len(symbol) > 0:
			x += [symbol]

'''
for line in sys.stdin:
	symbol = line.strip()
	if len(symbol) > 0:
		x += [symbol]

# print the sequence as string
print("Symbol sequence: ", mim.seq2str(x))

print("({0} symbols)".format(len(x)))

# create to be estimated from sequence x
m = mim.model(x)

# estimate model
K = m.estimate()

# print model
m.printmodel(m.M)

# show the probability distribution of the different sequences in the model
pz = mim.sortbyvalue(m.seqprobs())
for z, p in pz:
	print('{0:.3f} : {1}'.format(p, z))

print('Total number of sources: {0}'.format(K))
