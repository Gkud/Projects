from sklearn.metrics import confusion_matrix
import numpy as np
import math
from math import sqrt
from collections import OrderedDict
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from similarity_noun import Cluster
import ast
np.seterr(divide='ignore', invalid='ignore')

class Recommend:

	#Function to parse a file
	def parse(self, filename): 
		f = open(filename, 'r') 
		for l in f: 
			l = l.strip()
			if l:
				entry = ast.literal_eval(l)
				yield entry
		f.close()

	#Function to read data and create a dictionary of each product with their user ratings
	def read(self, filename):
		users = {}
		u = []
		for e in self.parse(filename):
			if e and e['review/score']:
				val = e['review/userId']
				if e['product/productId'] in users.keys():
					key = e['product/productId']
			
				else:
					key = e['product/productId']
					users[key] = {}

				users[key][val] = float(e['review/score'])
	
				if val not in u:
					u.append(val)

		return (users, u)

	#Function to create a rating matrix and find a unrated product
	def score(self, users, val, clusters, labels, prd):
		m = len(users.keys())
		n = len(val)
		mat = np.zeros((n, m))
		key = users.keys()
		
		#Creates a user-product matrix
		#An entry (i, j) of the matrix contains the rating given by user(i) to the product(j)
		p = 0
		for j in prd:
			q = 0
			for i in val:
				if i in users[j].keys():
					mat[q][p] = users[j][i]
				else:
					mat[q][p] = 0
				q = q+1
			p = p+1	
	
		#Finds an unrated product and recommends a rating
		for i in range(len(val)):
			index = np.where(mat[i, :] == 0)[0]
			v = val[i]
			if index.any():
			        o = self.recommend(index, clusters, labels,users, prd, v, mat, val)
				return o
		
        	
	#Function to get the cluster members of the unrated product
	def recommend(self, index, clusters, labels,users, prd, v, mat, u):
		for i in list(index):

			#Finds the cluster label of unrated product
			lab = labels[i]

			#Finds the product id of unrated product
			p = prd[i]

			#Finds the cluster members of unrated product
			members = clusters[lab]['pd']

			#Finds the rating similarity of unrated product with cluster members
			f = self.rating_sim(members, users, p, v, mat, u)
			return f
			

	#Function to find the rating and enhanced rating similarity 
	def rating_sim(self, members, users, p, v, mat, u):
		ratingSim = {}
		enhancedSim = {}
		normalized_rating = {}
		neighbours = {}

		#Finds rating rating similarity if there are atleast 3 members
		if len(members) >= 3:

			#Finds the users who have rated target product
			u1 = users[p].keys()

			#Calculates similarity with members
			for i in members:
				v1 = []
				v2 = []
				
				if i != p:
					
					#Finds the users who have rated member product
					u2 = users[i].keys()

					#Finds users which have rated both member and target product
					c = set(u1).intersection(set(u2))
					
					#Gets ratings of co-rated products
					for m in c:
						v1.append(users[p][m])
						v2.append(users[i][m])	

					#Finds pearson correlation coefficient of co-rated products			
					cor = pearsonr(v1, v2)[0]
					if math.isnan(cor):
						ratingSim[i] = 0
					else:
						ratingSim[i] = cor

					#Finds enhanced rating similarity of co-rated products
					e = (2*len(c))/float((len(u1)+len(u2)))
					enhancedSim[i] = ratingSim[i]
			
			#Sort the members in decreasing order of similarity
			sortedSim = OrderedDict(sorted(enhancedSim.items(), key=lambda kv: kv[1], reverse=True))

			count = 1

			#Selects k neighbors
			for i in sortedSim.keys():
				if count <=13:
					activeUserRate = users[i][v]
					neighbours[i] = sortedSim[i]

					#normalizes ratings to [-1, 1] scale
					ans = ((2*(activeUserRate - 1)) - 4) / 4.0

					normalized_rating[i] = ans
					count = count + 1
		else:
			neighbours = {}
	
		return self.predict_rating(neighbours, normalized_rating, users, p, v, mat, u)

	#Function to predict a rating 
	def predict_rating(self, neighbours, n_rate, users, p, v, mat, user_labels):
		num = 0
		den = 0
		flag = 0
	
		#Finds mean user rating
		index = user_labels.index(v)
		all_ratings = mat[index, :]
		all_ratings = filter(lambda a: a != 0, all_ratings)
		mu = np.mean(all_ratings)
				
		#If there are neighbors, finds weighted sum
		#Else returns mean user rating
		if any(neighbours):

			#Finds weighted sum of neighbors
			for i in neighbours.keys():
				num = num + (neighbours[i] * n_rate[i])
				den = den + abs(neighbours[i])
	
			#Finds the predicted rating
 			normalized_rating = float(num) / den

			#Denormalizes the rating and returns along with user and product id
			original_rating = (((normalized_rating + 1) * 4) * 0.5) + 1

			return [original_rating, p, v]
		else:
			return [mu, p, v]
	
		




	

