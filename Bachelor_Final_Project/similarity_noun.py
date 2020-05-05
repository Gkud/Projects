from sklearn.cluster import AgglomerativeClustering as ag
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
import nltk
from nltk.stem.porter import *
from textblob import TextBlob
import gzip
import numpy as np
import ast
import string

class Cluster:

	#Function to parse a file
	def parse(self, filename): 
		f = open(filename, 'r') 
		for l in f: 
			l = l.strip()
			if l:
				entry = ast.literal_eval(l)
				yield entry
		f.close()

	#Reads the file and creates a dictionary
	def read(self, filename):
		product = {}
		for e in self.parse(filename):
			if e:
				
				#remove punctuation
				desc = e['product/description']			
				nouns = self.find_nouns(desc)

				#Stems the noun phrases and gets tags
				tags = self.stem_word(nouns)
			
				#Stemmed tags are added to dictionary for each product
				product[e['product/productId']] = tags 
		return product

	#Finds nouns in text
	def find_nouns(self, desc):
		punc_free_text = desc.translate(string.maketrans("",""), string.punctuation)
		text = TextBlob(punc_free_text)

		#Finds noun in the description
		nouns = []
		tagged = text.tags
		for word, pos in tagged:
			if pos == "NN" or pos == "NNS" or pos == "NNP" or pos == "NNPS":
				nouns.append(word)

		return nouns
			
	#Function to stem words
	def stem_word(self, ls):
	    stemmed_words = []
		
	    #Creates a Porter Stemmer object
	    stemmer = PorterStemmer()
	    for word in ls:

		#Converts each word to lower case
		w = word.lower()

	        stemmed_words.append(stemmer.stem(w))
	    return stemmed_words

	#Calculates Jaccard Index for two sets 
	def jaccard_index(self, list1, list2):
	    set1 = set(list1)
	    set2 = set(list2)
	
	    intersect = len(list(set1.intersection(set2)))
	    union = len(list(set1.union(set2)))
	
	    if union == 0:
		return 0.0

	    ratio = float(intersect) / union
	    f = open('s1.txt','a+')
	    f.write(str(ratio)+'\n')
	    f.close()
	    return round(ratio, 3)

	#Function to find similarity using Jaccard index
	def similarity(self, desc1, desc2):
		D_sim = self.jaccard_index(desc1, desc2)
		return D_sim

	def similarity1(self, desc1, desc2, tag_count, f4):
		set1 = set(desc1)
		set2 = set(desc2)
	
		intersect = len(list(set1.intersection(set2)))
		f4.write('intersection = '+str(intersect))
		if tag_count == 0:
			return 0

		else:
			ratio = float(intersect)/tag_count
			f4.write('ratio = ' + str(ratio))
			return round(ratio, 3)


	#Creates a similarity matrix
	def compute_matrix(self, product):
		pd = product.keys()
		n = len(pd)
		mat = np.zeros((n, n))
	
		#Creates an upper triangular matrix
		for i in range(n):
			ls1 = product[pd[i]]
			for j in range(i+1, n):
				ls2 = product[pd[j]]
				mat[i][j] = self.similarity(ls1, ls2)
	
		return mat

	#Creates clusters of products based on description
	def cluster(self, m, no_of_clusters, product):
		prd = product.keys()

		clusters = []
		c = {}
		f4 = open('clust2.txt', 'a+')
		#Converts similarity matrix to a distance matrix
		m1 = 1-m

		#Applies hierarchical clustering to find clusters
		clustering = ag(linkage='average', n_clusters=no_of_clusters, affinity='precomputed')
		#labels =   AffinityPropagation(affinity='precomputed').fit_predict(np.triu(m))
 
		#Finds the clusters to which each object belongs
	        labels = clustering.fit_predict(m1)
		
		i = 0
		for k in np.unique(labels):
			members = np.where(labels == k)[0]
			c[i] = list(members)
			clusters.append({i:list(members), 'pd':[], 'tag':[]})
			i = i + 1

		for i in range(len(clusters)):
			f4.write(str(i)+'='+str(clusters[i][i])+'='+str(clusters[i]['pd'])+'\n')
		f4.write('\n\n\n')
		
		f4.close()

		#Finds clusters with their respective members
		tag_count = []
		l = 0
		for i in range(len(clusters)):
			tg = []
			for j in clusters[i][l]:
				clusters[i]['pd'].append(prd[j])
				tg = tg + product[prd[j]]
			t = set(tg)
			clusters[i]['tag'] = list(t)
			tag_count.append(len(list(t)))
			l = l + 1
	
		return (clusters, prd, list(labels), tag_count)

	#Function to get cluster members for a test product
	def test(self, clusters, tag_count, test_pd):
		sim = []
	        f4 = open('clust2.txt', 'a+')
		text = test_pd['product/description']
		n = self.find_nouns(text)
		pd_tags = self.stem_word(n)
		
		count = max(tag_count)
		f4.write('max_tag count = '+str(count))
		f4.write('min_tag count = '+str(min(tag_count)))
		for i in range(len(clusters)):
			ans = self.similarity1(pd_tags, clusters[i]['tag'], count, f4)
			sim.append(ans)
		
		f4.write('\n')
		f4.write(str(sim))
		max_sim = max(sim)
		index = sim.index(max_sim)
		f4.write('\n')
		f4.write(str(max_sim))
		f4.write('\n')
		f4.write(str(index))
		f4.write('\n')
		f4.write(str(clusters[index]['pd']))
		f4.write('\n\n')
		f4.close()
		return clusters[index]['pd']

	#Runs the different function in required order
	def test_run(self, filename, test_pd, no_of_clusters):
		
		product = self.read(filename)
		matrix = self.compute_matrix(product)
		clusters = self.cluster(matrix, no_of_clusters, product)
		a = self.test(clusters[0], clusters[3], test_pd)
		return a
	
	

#z = {"product/description": "Phone can be used during charging. Intelligent IC chips prevent overcharging.", "product/productId": #"B0009F8PPQ"}
#test_run('simtest.txt', z, 4)
		

	

				
		




	

