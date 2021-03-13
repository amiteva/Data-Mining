import csv
import math
import sys
import copy

def read_file(file_name):
    f = open(file_name, "rt", encoding="utf8")

    listOfCountries=[] #list of the names of all countries
    for l in csv.reader(f):
        if l[2]=="From country": #we of course need to skip the first line of the .csv
            continue
        if l[2]=="F.Y.R. Macedonia": #I will consider North Macedonia and F.Y.R. Macedonia as the same
            continue

        fromCountry=l[2]
        if fromCountry not in listOfCountries:
            listOfCountries.append(fromCountry)

    sum = {} #dictionary that represents the number of points that each one country gave to all the other countries.
             #I took the average points of all years summed up together
    counter={} #dictionary that represents how many times one country has voted for all the other countries
               #Or we can explain it as the number of times the pair (c1,c2) appears

    n = len(listOfCountries) #number of countries
    countryToIndex = {}
    count = 0
    for c in listOfCountries:
        sum[c] = [0 for i in range(n)]
        counter[c]= [0 for i in range(n)]
        countryToIndex[c] = count # {'Slovenia': 0, 'Macedonia': 1, ...}, list of indexes of each country
        count += 1

    f.close()

    #print(countryToIndex)

    f = open(file_name, "rt", encoding="utf8")
    for l in csv.reader(f):
        if l[2]=="From country":
            continue
        year = int(l[0])
        fromCountry=l[2]
        toCountry=l[3]
        points=int(l[4])
        if l[2]=="F.Y.R. Macedonia":
            fromCountry="North Macedonia"
        if l[3]=="F.Y.R. Macedonia":
            toCountry="North Macedonia"

        sum[fromCountry][countryToIndex[toCountry]] += points
        counter[fromCountry][countryToIndex[toCountry]]+=1

    for country in listOfCountries:
        for i in range(n):
            if counter[country][i] != 0:
                sum[country][i] /= counter[country][i]
            else:
                sum[country][i] = None

    f.close()

    """
    #Printing used for figuring out the most and least voted countries in each cluster
    sumcopy = copy.deepcopy(sum)
    
    for x in sumcopy:
        pairs = []
        for i in countryToIndex:
            if sumcopy[x][countryToIndex[i]] == None:
                pairs.append([-1, i])
            else:
                pairs.append([sumcopy[x][countryToIndex[i]], i])
        pairs.sort()

        print(f"Country {x} is voting for")
        for p in pairs:
            print(p)
        print("\n")
    """

    return sum

class HierarchicalClustering:
    def __init__(self, data):
        self.data = data
        self.clusters = [[name] for name in self.data.keys()] # [['Slovenia'], ['France'], ...]
        
        #self.data is a dictionary where the keys are names of countries.
        # This means that function read_data should return that dictionary.
        # data = {'Slovenia': [a1,a2,..,an],
        #          'France': [b1, b2, ..., bn],...}
        # For e.g. a1 represents how many points Slovenia gave to the country on index 0

        

        self.distances={} #Additional attribute - gives us the distances between countries separately, it is a dictionary of lists
        for c in self.data.keys():
            self.distances[c] = {}
            for c2 in self.data.keys():
                self.distances[c][c2]=self.row_distance(c,c2)

        self.history=[]
        #The elements in self.history are like: [['Slovenia'], ['France'], 20] <- meaning that in this step we put together 
        #Slovenia and France in the same cluster whith the distance 20
        #with this we keep up with how the clusters are being formed in each iteration and what is the distance between them
        

    def row_distance(self, r1, r2):
        #im using Euclidean distance
        distance=0.0
        counter=0
        for i in range(len(self.data[r1])):
            if self.data[r1][i]!=None and self.data[r2][i]!=None:
                distance+=pow(self.data[r1][i]-self.data[r2][i],2)
                counter+=1
        #print(len(self.data[r2]))
        return math.sqrt(distance*len(self.data[r1])/counter)

    def numberOfCountries(self,c): #gives us a list of all countries that appear in a cluster
        if len(c)==1: #For e.g c = ["Albert"]
            return c
        return self.numberOfCountries(c[0])+self.numberOfCountries(c[1])


    def cluster_distance(self, c1, c2):
        #im using average linkage
        clusterDistance=0.0
        n1=self.numberOfCountries(c1)
        n2=self.numberOfCountries(c2)
        for x in n1:
            for y in n2:
                clusterDistance+=self.distances[x][y]

        return clusterDistance/(len(n1)*len(n2))

    def closest_clusters(self):
        result=[]
        minDistance=sys.float_info.max
        for x in self.clusters:
            for y in self.clusters:
                if x!=y:
                    d=self.cluster_distance(x,y)
                    if minDistance>d:
                        minDistance=d
                        c1=x 
                        c2=y
        result=[c1,c2,d] #return [first_cluster, second_cluster, distance] - the clusters that we are merging together, d is the distance between them
        return result

    def run(self):
        # we iterate until we dont have only one cluster: len(self.clusters)==1
        rez=[] #here we save the result from the closest clusters
        while len(self.clusters)>1:
            rez=self.closest_clusters()
            cluster=[rez[0], rez[1]]
            self.clusters.append(cluster)
            self.clusters.remove(rez[0])
            self.clusters.remove(rez[1])
            self.history.append(self.clusters)


    def recursivePlot(self,c):
        if len(c)==1:
            #print(c)
            return ["---- "+c[0]]
        left=self.recursivePlot(c[0])
        right=self.recursivePlot(c[1])

        result = []
        for s in left: #go through all the strings in the left "son"
            result.append("    "+s)
        result.append("----|") #the middle
        for s in right: #go through all the strings in the right "son"
            result.append("    "+s)
        return result


    def plot_tree(self):
        for i in range(1):
            printing=self.recursivePlot(self.clusters[i])
            # printing = ['a', 'b', 'c']
            # after join: 'a\nb\nc'
            #print(f"cluster {i+1}:\n")
            print("\n".join(printing))


if __name__ == "__main__":
    DATA_FILE = "eurovision-finals-1975-2019.csv"
    hc = HierarchicalClustering(read_file(DATA_FILE))
    hc.run()
    hc.plot_tree()