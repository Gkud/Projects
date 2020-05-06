import gzip  
import sys 
import ast
from collections import OrderedDict
from vaderSentiment.vaderSentiment import sentiment
from pytldr.summarize.relevance import RelevanceSummarizer
from nltk import sent_tokenize
import os
import matplotlib.pyplot as plt
import numpy as np
#reload(sys)
#sys.setdefaultencoding("ISO8859-1")

class Converter(object):

	#Function to parse a file
	def parse(self, filename): 
		f = open(filename, 'r') 
		for l in f: 
			l = l.strip()
			if l:
				entry = ast.literal_eval(l)
				yield entry
		f.close()


	#Parses the file and creates a dictionary
	#Counts the number of 1 to 5 star ratings for each product
	#Creates files for each product containing all reviews of that product
	def read(self, filename, title, category, cl):
		n = 1
		
		for e in self.parse(filename):
			if e and (e['product/productId'] in cl):
				if e['product/productId'] in title.keys():
					t = e['product/productId']
				
				else:
					fn = 'p'+str(n)+'.txt'
					t = e['product/productId']
					title[t] = []
					title[t].append(fn)
					ratings = {1:0, 2:0, 3:0, 4:0, 5:0}
					title[t].append(ratings)
					title[t].append(e['product/title'])
					count = 0
					title[t].append(count)
						
					if t in category.keys():
						title[t].append(category[t])
					else:
						title[t].append('')
					
					n = n + 1
				
				#Creates a file to combine the reviews for each product
				data = open(title[t][0], 'a+')
				data.write(e['review/text']+" ") 

				#Obtains the calculated rating
				if e['review/text']:
					cal_rating = self.convert_to_rating(e['review/text'])
				else:
					cal_rating = 0.0
				#Obtains the rating given by the user
				if e['review/score']:
					given_rating = float(e['review/score'])
				else:
					given_rating = 0.0

				#Obtains the average of calculated and given ratings
				new_rating = (cal_rating + given_rating) / 2.0
				new = int(round(new_rating))
		
				#Counts the number of each rating
				if new != 0:				
					r = title[t][1][new]
					title[t][1][new] = r + 1
				
				#Counts the number of reviews
				title[t][3] = title[t][3] + 1
				data.close()
	

	#Converts the review text to a rating
	def convert_to_rating(self, text): 
    
		sentences = sent_tokenize(text)
        	pos = 0
        	neg = 0
        	neu = 0

		#Finds number of positive, negative and neutral sentences
        	for sentence in sentences:
        		vs = sentiment(sentence)
       
        		if vs['neg'] > 0.0 and vs['compound'] < 0.0:
            			neg = neg + 1
        		elif vs['neg'] > 0.0 and vs['compound'] > 0.0:
            			if vs['pos'] > 0.0 and vs['neu'] > 0.0 and vs['compound'] > 0.5:
                			pos = pos + 1
            			elif vs['pos'] == 0.0 and vs['neu'] > 0.0:
                			neu = neu + 1
        		elif vs['neg'] == 0.0 and vs['compound'] > 0.0:
            			if abs(vs['neu'] - vs['pos']) <= 0.3:
                			pos = pos + 1
            			else:
                			neu = neu + 1
        		else:
            			neu = neu + 1

		#Obtains total number of positive and negative sentences
       		total = pos+neg
        	if total == 0:
        		total = 1
			
		#Formula to convert text to rating on a scale of [1:5]
    		rating = ((float(pos) / (total)) * 4.0) + 1
    
    		return rating
    
   
	#Function to summarize the reviews of each product
	def summarize(self, fname, fp):
		summarizer = RelevanceSummarizer()
		fn = open(fname, 'r')
		content = fn.read()
		#summ = summarizer.summarize(content, length=1)
		summ = summarizer.summarize(content, length=2, binary_matrix=True) 
		fp.write('Summary:')
		if summ:
			fp.write(''.join(summ))
		else:
			fp.write(content)
		fp.write('\n\n\n')
		fn.close()
		#os.remove(fname)

	#Function to find the average rating of each product
	def average(self, rate):
		total = 0
		for i in range(1,6):
			total = total + rate[i]*i
		den = sum(rate.values())
		return total/float(den)

	#Function to find the percentage of each rating for a given product
	def rating_percent(self, scores):
		ans = []
		count = 0
		for i in scores:
			
			count = count + 1
			v = i.values()
			total = sum(v)
			factor = 100.0/total
			v = [x*factor for x in v]
			ans.append(v)
			if count == 3:
				break		
		return ans

	def variance(self, rate, count):
		ls = []
		if count > 20:
			for i in rate.keys():
				w = []
				w.append(i)
				w = w * rate[i]
				ls = ls + w
	
			var = np.var(ls)
			
			if var >= 1:
				return self.average(rate)
			else:
				max_rate = max(rate.values())
				for i in rate.keys():
					if rate[i] == max_rate:
						return i
		
		else:
			return 0.0


	#Plots the rating percentage of first three highest rated products
	def plot(self, percent, title):
		N = 5
		ind = np.arange(N)  
		width = 0.23  
		color = ['red', 'green', 'blue']
		r = []
		rec = ()
		pd = ()

		fig = plt.figure()
		ax = fig.add_subplot(111)

		for i in range(len(percent)):
			uvals = percent[i]
			rects = ax.bar(ind+(width*i), uvals, width, color=color[i])
			self.autolabel(rects, ax)
			rec = rec + (rects[0], )
		 	name = 'Product' + str(i+1)
			pd = pd + (name, )
			

		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

		ax.set_ylabel('Percentage Ratings')
		ax.set_xticks(ind*1.12+width-0.2)
		
		ax.set_xticklabels( ('*', '**', '***', '****', '*****') )
		ax.set_xlabel('Star Ratings') 
		ax.set_title(title)
		hleg = ax.legend( rec, pd, loc='upper right', bbox_to_anchor=(1.4, 0.5))
		plt.show()

	# Attaches text labels
	def autolabel(self, rects, ax):
		for rect in rects:
        		height = rect.get_height()
        		ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height), ha='center', va='bottom')

	def select_competitors(self, d):
		for name in d.keys():
			overall_rate = self.variance(d[name][1], d[name][3])
			d[name].append(overall_rate)

	def output(self, sort_product, fp, title):
		count = 0
		flag = 0
		scores = []
		for key in sort_product.keys():
			if sort_product[key][5] == 0.0:
				continue
			count = count + 1
			if count <= 5: 
				fp.write('Product: '+ sort_product[key][2] +'\n')
				fp.write('Average Ratings: ' + format(sort_product[key][5], '.2f') + ' out of 5\n')
				self.summarize(sort_product[key][0],  fp)
				scores.append(sort_product[key][1])
				flag = 1	
			else:
				break

		if flag == 0:
			fp.write("No competitors were found for the product in the data");

		fp.write('\n\n')
			
		percent = self.rating_percent(scores)
    		self.plot(percent, title) 

	#Runs the different functions in required order
	def run(self, fname, cl, category, targetCategory, sumFile):
		title = {} 
		similarCategory = {}
		self.read(fname, title, category, cl)
		
		for i in title.keys():
			if targetCategory in title[i][4]:
				similarCategory[i] = title[i]
		
		fp = open(sumFile, 'a+')

		self.select_competitors(title)
		sort_product = OrderedDict(sorted(title.items(), key=lambda kv: kv[1][5], reverse=True))
		fp.write('Mixed Products'+'\n')
		self.output(sort_product, fp, 'Mixed Products')

		self.select_competitors(similarCategory)
		sort_product = OrderedDict(sorted(similarCategory.items(), key=lambda kv: kv[1][5], reverse=True))
		f = open('s1.txt', 'a+')
		f.write(str(sort_product.keys()))
		fp.write('Products by Category'+'\n')
		self.output(sort_product, fp, 'Products by Category')

		fp.close()
		
		for name in title.keys():
			os.remove(title[name][0])

	

