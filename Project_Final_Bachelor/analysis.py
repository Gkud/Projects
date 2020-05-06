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
from similarity import Cluster
import ast
import time

#Function to parse a file
def parse(filename): 
		f = open(filename, 'r') 
		for l in f: 
			l = l.strip()
			if l:
				entry = ast.literal_eval(l)
				yield entry
		f.close()

#Function to read data and create a dictionary of each product with their user ratings
def read(filename):
	
	users = {}
	u = []
	for e in parse(filename):
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


#Function to create a user-product matrix
#An entry (i, j) of the matrix contains the rating given by user(i) to the product(j)
def score(users, val, clusters, labels, prd):
	m = len(users.keys())
	n = len(val)
	mat = np.zeros((n, m))
	p = 0
	key = users.keys()
	for j in prd:
		#print j
		q = 0
		for i in val:
			if i in users[j].keys():
				#print i
				mat[q][p] = users[j][i]
			else:
				mat[q][p] = 0
				#print mat[q][p]
			q = q+1
		p = p+1
	
	predict(mat, n, m, users, val, clusters, labels, prd)

#Function to find predicted ratings and test for rated products
def predict(mat, no_of_users, no_of_products, users, u, clusters, labels, prd):
	actual = []
	predicted = []
	
	for j in range(no_of_users):
		for i in range(no_of_products):
			if mat[j][i]: 
				rate = mat[j][i]
				actual.append(rate)
				user = u[j]
				pred = recommend(i, clusters, labels,users, prd, user, mat, u, rate)
				predicted.append(round(pred))
					
	measures(actual, predicted)

#Function to get the cluster members of the target product
def recommend(index, clusters, labels,users, prd, v, mat, u, rate):
	#Finds the cluster label of target product
	lab = labels[index]

        #Finds the product id of target product
	p = prd[index]

	#Finds the cluster members of target product
	members = clusters[lab]['pd']
	
	#Returns rating similarity of target product with cluster members
	return rating_sim(members, users, p, v, mat, u, rate)
	
#Function to find the rating and enhanced rating similarity 
def rating_sim(members, users, p, v, mat, user_labels, rate):
	ratingSim = {}
	enhancedSim = {}
	normalized_rating = {}
	neighbours = {}

	#Finds rating rating similarity if there are atleast 3 members
	if len(members) >= 3:

		#Finds the users who have rated target product
		u1 = users[p].keys()

		#Remove the active user from the rated users list
		u1.remove(v)

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
				#ratedUsers = users[i].keys()
				activeUserRate = users[i][v]
				neighbours[i] = sortedSim[i]

				#normalizes ratings to [-1, 1] scale
				ans = ((2*(activeUserRate - 1)) - 4) / 4.0

				normalized_rating[i] = ans
				count = count + 1
	else:
		neighbours = {}


	return predict_rating(neighbours, normalized_rating, users, p, v, rate, mat, user_labels)

#Function to predict a rating
def predict_rating(neighbours, n_rate, users, p, v, rate, mat, user_labels):
	num = 0
	den = 0
	flag = 0
	
	#Finds mean user rating
	index = user_labels.index(v)
	all_ratings = mat[index, :]
	all_ratings = filter(lambda a: a != 0, all_ratings)

	#Removes active user rating from rating list
	all_ratings.remove(rate)

	#Finds the mean 
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

		#Denormalizes the rating to [1,5] scale
		original_rating = (((normalized_rating + 1) * 4) * 0.5) + 1
		
		return original_rating
	else:
		return mu
	
#Function to find MAE 
def measures(actual, predicted):
	print 'MAE = '+str(mean_absolute_error(actual, predicted))	

#Function to run the tests in order
def test(filename, no_of_clusters):
	start_time = time.clock()
	cl = Cluster()	
	product = cl.read(filename)
	matrix = cl.compute_matrix(product)
	c = cl.cluster(matrix, no_of_clusters, product)
	u = read('userGrp6.txt')
	score(u[0], u[1], c[0], c[2], c[1])
	print time.clock() - start_time, "seconds"

test('descr1.txt', 15)	
