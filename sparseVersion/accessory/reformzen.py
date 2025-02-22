import numpy as np
import pylab as pl
import zen
import networkx as nx
import datetime


def zenGraph(edg):
    gg = zen.DiGraph()
    for ii in edg:
        gg.add_edge(ii[0], ii[1])
    return gg
    
def idPermute(gg):	
	gg = zen.to_networkx(gg)
	perm = np.arange(len(gg), dtype='int')
	nV = len(perm)
	np.random.shuffle(perm)
	mappg = {i:perm[i] for i in range(nV)}
	hh = nx.relabel_nodes(gg, mappg)
	hh = zen.from_networkx(hh)
	return hh


def orderMMEdge(gg):
	nV = len(gg)
	Edg = gg.edges()
	nEdg = len(Edg)
	jj=0
	# nDD = []
	DD = []
	while nEdg>0:
		'ordering ', jj
		jj+=1
		print datetime.datetime.now().replace(microsecond=0)
		MM = zen.maximum_matching(gg)	
		print len(MM)	
		DD = DD + MM
		# nDD.append(len(MM))
		for ii in MM:
			src, tgt = ii
			gg.rm_edge(src,tgt)
		Edg = gg.edges()
		gg = zenGraph(Edg) # ----
	 	nEdg = len(Edg)
		print '----', jj, 'and remained edges', nEdg 
	# nDD = nV - np.array(nDD)
	DD = np.array(DD)
	return DD#nDD
		
			
def randPr(gg, dEE): # excluding the original nCN
	nN = len(gg)
	Edg = gg.edges()
	np.random.shuffle(Edg)
	nEdg = len(Edg)
	DD=[]#np.zeros(200)
	jj=0
	while nEdg>dEE:
		print 'rnd ----', jj
		print datetime.datetime.now().replace(microsecond=0)
		
		Edg = Edg[dEE:]
		gg = zen.DiGraph()
		for ii in Edg:
			gg.add_edge(ii[0], ii[1])
		DD.append(len(zen.maximum_matching(gg)))
 		nEdg = len(Edg)
		jj+=1
	MM = np.array(DD)
	return nN-MM

def outPr(gg, dEE): # excluding the original nCN
	nN = len(gg)
	Edg = gg.edges()	
	nEdg = len(Edg)
	DD=[]#np.zeros(200)
	jj=0
	while nEdg>dEE:
		print 'out ----', jj, ' and remained: ', nEdg
		print datetime.datetime.now().replace(microsecond=0)
		Edg = Edg[dEE:]
		gg = zen.DiGraph()
		for ii in Edg:
			gg.add_edge(ii[0], ii[1])
		DD.append(len(zen.maximum_matching(gg)))
 		nEdg = len(Edg)
		jj+=1
	MM = np.array(DD)
	return nN-MM

def ordPr(gg, dEE): # excluding the original nCN
	nN = len(gg)
	Edg = orderMMEdge(gg.copy())		
	nEdg = len(Edg)
	DD=[]#np.zeros(200)
	jj=0
	while nEdg>dEE:
		print 'ord ----', jj
		print datetime.datetime.now().replace(microsecond=0)
		rmmv = Edg[:dEE]
		for ii in rmmv:
			print ii
			src, tgt = ii
			gg.rm_edge(src,tgt)
		DD.append(len(zen.maximum_matching(gg)))
		Edg = Edg[dEE:]
		gg = zen.DiGraph() # ----
		for ii in Edg:
			gg.add_edge(ii[0], ii[1])
 		nEdg = len(Edg)
		jj+=1
	MM = np.array(DD)
	return nN-MM



		







