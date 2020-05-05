from new_parse import Converter
from similarity_noun import Cluster
from tagger import Tagger
import time

def test_run1(filename, test_product, no_of_clusters):
	conv = Converter()
	cl = Cluster()
	tg = Tagger()

	targetCategory = test_product['category']
	result = cl.test_run(filename, test_product, no_of_clusters)
	category = tg.readCategory(result)
	conv.run('r200.txt', result, category, targetCategory, 'newSum1.txt')


"""prd = {"product/description": "Phone can be used during charging. Intelligent IC chips prevent overcharging.", "product/productId": "B0009F8PPQ"}
#test_run1('d150.txt', prd, 10)



prd = {"category": "Industrial & Scientific", "product/description": "Precision seamless drawn 1/2 to 3/4 hard brass Telescoping Tubing is the answer to many problems in the development of experimental models and prototypes. Telescoping Tubing can be easily soldered, brazed and flared to accept fittings. Advantageous when a reduction in size, weight or flow is desired. Each telescopes into the next larger size. General purpose 1/2 to 3/4 hard brass. Approximate composition: 65-70 percent copper and 30-35 percent zinc", "product/productId": "B000FN434G"}

prd = {"category": "Jewelry", "product/description": "Deluxe Light Oak Wood Finish Jewelry Armoire This is a brand new deluxe Jewelry Armoire. Item is designed in an light oak finish with seven drawers, two side doors for hanging jewelry and a mirror underneath the top opening cover. Items may require simple assembly. Dimensions Measure: 19W 13.25D 37.5H", "product/productId": "B000GBUWJW"}"""

prd = {"category": "Jewelry", "product/description": "Enjoy the beauty of this silver-plated and crystal Expressively Yours bracelet. The Sister, Friend, and Forever Words are on both sides of the silver bead. It is finished off with a heart charm and a toggle clasp. Included with this bracelet is this inspirational saying: Sisters care and sisters share, Sometimes they disagree, A sister is a special friend, Forever family. It comes packaged in a beautiful gift box with a ribbon-ready to give as a gift. Ages 13 and up, 8 long.", "product/productId": "B000NZWU00"}

start_time = time.clock()
test_run1('d200.txt', prd, 10)
print time.clock() - start_time, "seconds"


