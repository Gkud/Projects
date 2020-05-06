from test_predict import Recommend
from similarity import Cluster

#Function to predict a rating
def test_run2(filename, no_of_clusters):

	#Creates Recommend and Cluster objects
	rec = Recommend()
	cl = Cluster()
	
	#Reads the product description data file
	product = cl.read(filename)
	
	#Finds the similarity matrix of product descriptions
	matrix = cl.compute_matrix(product)

	#Forms clusters
	c = cl.cluster(matrix, no_of_clusters, product)

	#Reads the co-rated product file
	u = rec.read('Testing.txt')

	#Finds the predicted rating
	predicted_rating = rec.score(u[0], u[1], c[0], c[2], c[1])

	#Prints the user id
	print 'User: ' + predicted_rating[2]

	#Prints the product id
	print 'Product: ' + predicted_rating[1]
	
	#Prints the predicted rating
	print 'Predicted Rating for the product by the user: ' + str(format(predicted_rating[0], '.2f'))

	#If the predicted rating is greater than 2.5, the product is recommended to the user
	#Else it is not recommended
	if predicted_rating[0] > 2.5:
		print 'Recommended'
	else:
		print 'Not recommended'


test_run2('descr1.txt', 15)



		


