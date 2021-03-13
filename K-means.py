from os import listdir
from os.path import join
from unidecode import unidecode #unidecode()
from matplotlib import pyplot as plt
import numpy as np
import math
import sys
import random

def terke(text, n): #returns a dictionary of words of length 3 and their relativni frekvenci 
    """
    Vrne slovar s preštetimi terkami dolžine n.
    'abcd efg' -> 1. 'abc', 'bcd', 'efg' 
                  2. 'abc', 'bcd', 'cd ', 'd e', ' ef', 'efg'
                  #There are two opstions on how to read 3-terke, and im using the 1st one.
    """
    ter={}
    text=unidecode(text.lower())
    for x in range(len(text)-n+1):
        subtext=text[x:x+n] #get a substring of n letters from text
        if not subtext.isalpha(): #if the substring includes a space or any other special characters, skip it
            continue
        if subtext in ter:
            ter[subtext]+=1
        else:
            ter[subtext]=1
    return ter

def read_clustering_data(n_terke):
    # Prosim, ne spreminjajte te funkcije. Vso potrebno obdelavo naredite
    # v funkciji terke.
    lds = {}
    for fn in listdir("clustering"):
        if fn.lower().endswith(".txt"):
            with open(join("clustering", fn), encoding="utf8") as f:
                text = f.read()
                nter = terke(text, n=n_terke)
                lds[fn] = nter
    #lds is a dictionary of dictionaries: {"slovenian.txt": {"abc":3,"efg":4...}, "macedonian.txt":{"abc":5,"efg":6...},...}
    return lds


def read_prediction_data(n_terke):
    # Prosim, ne spreminjajte te funkcije. Vso potrebno obdelavo naredite
    # v funkciji terke.
    lds = {}
    for fn in listdir("prediction"):
        if fn.lower().endswith(".txt"):
            with open(join("prediction", fn), encoding="utf8") as f:
                text = f.read()
                nter = terke(text, n=n_terke)
                lds[fn] = nter
    return lds


def cosine_dist(d1, d2):
    """
    Vrne kosinusno razdaljo med slovarjema terk d1 in d2.
    """
    suma=0
    for x in d1:
        if x in d2:
            suma+=(d1[x]*d2[x])
    sqrt1=0
    sqrt2=0
    for i in d1:
        sqrt1+=math.pow(d1[i],2)
    for i in d2:
        sqrt2+=math.pow(d2[i],2)
    return 1-suma/(math.sqrt(sqrt1)*math.sqrt(sqrt2))


def new_medoid(distances, cluster): #finds the most centered point in a cluster
    # ['slovene', 'macedonian', ...]
    minaveragesum=sys.float_info.max
    medoid=cluster[0]
    for x in cluster:
        if calculate_average(distances, cluster, x)<minaveragesum:
            minaveragesum=calculate_average(distances, cluster, x)
            medoid=x
    return medoid


def calculate_average(distances, cluster, example): #calculates how centered a point/example is
    averagesum=0.0
    for x in distances[example]: #sum the distances between the example and all other points in that cluster
        if x in cluster:
            averagesum+=distances[example][x]
    return averagesum/len(cluster) #to get the average sum we divide with the number of distances that we summed up together


def closest_medoid(distances, medoids, example): #it returns the medoid to which the example is closest to
    mindist=sys.float_info.max
    closestmedoid=""
    for x in medoids:
        if distances[example][x]<mindist:
            mindist=distances[example][x]
            closestmedoid=x
    return closestmedoid

def create_list(clusters): #returns list of lists a.k.a. all clusters together in a list
    listofclusters=[]
    for c in clusters:
        listofclusters.append(clusters[c])
    return listofclusters

def k_medoids(data, medoids):
    # [['X', 'Y'], ['Z']]
    """
    Za podane podatke (slovar slovarjev terk) in medoide vrne končne skupine
    kot seznam seznamov nizov (ključev v slovarju data).
    """
    clusters={} #a dictionary of 5 lists, each list represents one cluster of langugaes, the keys are the medoids
    distances={} #a dictionary of dictionaries that saves the distances between a language and all other languages
    for x in data.keys():
        distances[x]={}
        for y in data.keys():
            if x == y: distances[x][y]=0.0
            else: distances[x][y]=cosine_dist(data[x],data[y])

    for m in medoids:
        clusters[m]=[]

    for x in data.keys():
        m=closest_medoid(distances,medoids,x)
        clusters[m].append(x)
    
    change=False #keeps account if the elements in the clusters change
    while not change: #repeat the iterations until the clusters don't change
        change=True
        #first recalculate the new medoids
        keys = list(clusters.keys()).copy()
        for m in keys:
            newmedoid=new_medoid(distances, clusters[m])
            if m!=newmedoid:
                clusters[newmedoid]=clusters[m]
                del clusters[m]

        #go through all clusters and update them a.k.a. reassign the elements to the cluster to which medoid they are the closest to
        for c in clusters.keys():
            for e in clusters[c]:
                newmedoid=closest_medoid(distances,clusters.keys(),e)
                if newmedoid!=c:
                    clusters[c].remove(e)
                    clusters[newmedoid].append(e)
                    change=False
    
    return create_list(clusters)

def closest_cluster(distances, clusters, element, medoid): #it returns the medoid of the cluster that the element is the closest to
    #medoid is a parameter represinting the medoid of the cluster of which "element" is part of
    minaveragedistace=sys.float_info.max
    m=medoid #m is the medoid of the cluster to which the element is the closest to (m!=medoid)
    for c in clusters:
        if medoid in c: continue
        if minaveragedistace>calculate_average(distances, c, element):
            m=c
            minaveragedistace=calculate_average(distances, c, element)

    return m

def silhouette(data, clusters):
    """
    Za podane podatke (slovar slovarjev terk) in skupine (seznam seznamov nizov:
    ključev v slovarju data) vrne silhueto.
    """
    distances={} #a dictionary of dictionaries that saves the distances between a language and all other languages
    for x in data.keys():
        distances[x]={}
        for y in data.keys():
            if x == y: distances[x][y]=0.0
            else: distances[x][y]=cosine_dist(data[x],data[y])
    s=0.0
    for c in clusters:
        for e in c:
            m=closest_cluster(distances,clusters,e,c[0])
            a=calculate_average(distances,c,e)
            b=calculate_average(distances,m,e)
            s+=((b-a)/max(a,b))

    return s/len(data)

def predict(data, text, n_terke):
    #returns {'slovene': 0.8, 'macedonian': 0.15, ...}
    #data={'slovene': {'abc': 5, 'cde': 1, ..}
    #       'macedonian':...}
    terka = terke(text, n_terke)
    distances={}
    totalsum=0.0
    for l in data:
        distances[l] = 1 / cosine_dist(terka, data[l])
        totalsum+=distances[l]

    for l in distances:
        distances[l] /= totalsum

    return distances



def del2():
    data = read_clustering_data(3)  # dolžino terk prilagodite
    hc = HierarchicalClustering(data)
    hc.run()
    hc.plot_tree()


def del4():
    data = read_clustering_data(3)  # dolžino terk prilagodite
    maxs=sys.float_info.min
    mins=sys.float_info.max
    bestclustering=[]
    worstclustering=[]
    arrayofs = []
    for i in range(50):
        clusters=k_medoids(data,random.sample(data.keys(), 5))
        s=silhouette(data, clusters)
        arrayofs.append(s)
        if s>maxs:
            maxs=s
            bestclustering=clusters.copy()
        if s<mins:
            mins=s
            worstclustering=clusters.copy()
    print("Clustering with the best silhouette\n")
    for a in bestclustering:
        print(a)
    print("Clustering with the wors silhouette\n")
    for a in worstclustering:
        print(a)

    p = plt.hist(arrayofs, bins = np.arange(0, 1.05, 0.05))
    plt.show()


def del5():
    data = read_prediction_data(3)  # dolžino terk prilagodite
    # ... nadaljujte
    # primer klica predict: print(predict(data, "Danes je lep dan", 3))
    print(predict(data,"Visokošolski strokovni študijski program prve stopnje računalništvo in informatika traja 3 leta (6 semestrov) in obsega skupaj 180 kreditnih točk. Študijski program se izvaja v slovenskem jeziku. Pridobljeni strokovni naslov je: • diplomirani inženir računalništva in informatike (VS), • diplomirana inženirka računalništva in informatike (VS) oziroma z okrajšavo dipl. inž. rač. in inf. (VS). Študijsko področje, v katerega se program uvršča (po klasifikaciji ISCED) Osnovno področje je »računalništvo (48)«, program pa delno posega tudi na »tehniške vede (52)« ter vsebuje tudi izobraževalne vsebine s področja informatike, ki pa ni posebej opredeljeno v ISCED klasifikaciji. Znanstvene discipline, na katerih temelji program (po klasifikaciji Frascati) Program sodi na področje »tehniške in naravoslovno-matematične vede«, s svojim znatnim delom pa posega na področje informatike. Razvrstitev v nacionalno ogrodje kvalifikacij, evropsko ogrodje visokošolskih klasifikacij ter evropsko ogrodje kvalifikacij Ravni kvalifikacij: slovensko ogrodje kvalifikacij (SOK) 7; evropsko ogrodje kvalifikacij (EOK) 6; evropsko ogrodje visokošolskih kvalifikacij (EOVK) prva stopnja.",3))
    print(predict(data,"Ed ora finalmente tutto sta per finire fra di noi, il mondo è largo, e dei Leonardi e delle Erneste ce ne possono vivere molte paia senza che siano obbligati a guardarsi nel bianco dell'occhio a tavola, ad andare a braccetto per le vie. Sarò finalmente libera, mi tornerà il respiro.che orrore i diritti ed i doveri dei coniugi per due che non si vogliono bene! E che odioso e fatuo libro il codice colla sua aria di volere, con quattro ciancie numerate, regolare in eterno un affetto che alle volte dura… Quanto ha durato il nostro?Apparentemente tre mesi, in realtà meno di tre quarti d'ora, perchè non ci è mai stato affetto vero tra Leonardo e me; non l'amo e non mi ama, oggi come ieri e come tre mesi or sono.Apparentemente tre mesi, in realtà meno di tre quarti d'ora, perchè non ci è mai stato affetto vero tra Leonardo e me; non l'amo e non mi ama, oggi come ieri e come tre mesi or sono.Tu sai come è andata la cosa; morì la mamma, rimasi sola nel mondo; lo zio Rinucci, la zia Rinucci e mia cugina Rinucci mi aprirono le braccia a modo loro, vale a dire mi accolsero in casa nei primi giorni che succedettero alla sciagura.",3))
    print(predict(data,"Die Erziehung dieses Sohnes, welcher Alaeddin hieß, war sehr vernachlässigt worden, so daß er allerhand lasterhafte Neigungen angenommen hatte. Er war boshaft, halsstarrig und ungehorsam gegen Vater und Mutter. Kaum war er ein wenig herangewachsen, so konnten ihn seine Eltern nicht mehr im Hause zurückhalten. Er ging schon am frühen Morgen aus und tat den ganzen Tag nichts, als auf den Straßen und öffentlichen Plätzen mit kleinen Tagdieben spielen.Als er ein Handwerk erlernen sollte, nahm ihn sein Vater in seine Bude und fing an, ihn in der Handhabung der Nadel zu unterrichten. Allein weder gute Worte noch Drohungen vermochten den flatterhaften Sinn des Sohnes zu fesseln. Kaum hatte Mustafa ihm den Rücken gekehrt, so entwischte Alaeddin und ließ sich den ganzen Tag nicht wieder sehen. Der Vater züchtigte ihn, aber Alaeddin war unverbesserlich, und Mustafa mußte ihn mit großem Bedauern zuletzt seinem liederlichen Leben überlassen. Dies verursachte ihm großes Herzeleid, und der Kummer zog ihm eine hartnäckige Krankheit zu, an der er nach einigen Monaten starb.", 3))
    print(predict(data,"Maar meer nog dan hun hoofd zal hun hart door de lezing winnen. Voor de vorming daarvan vooral verdient dit zeldzaam voortreffelijke boek algemeen gelezen te worden. Maar tot mijn achtste jaar geloofde ik, evenals alle andere kinderen, ook eene moeder te hebben, want als ik weende, was er eene vrouw die mij in hare armen nam en mij tegen haar boezem drukte totdat mijne tranen ophielden te vloeien. Nooit werd ik in mijn bedje gelegd of eene vrouw gaf mij een kus, en als de Decemberwind de sneeuwvlokken tegen de bevroren ruiten joeg, nam zij mijne voetjes in hare beide handen om ze te verwarmen en zij zong dan een liedje, waarvan de wijs en ook eenige woorden nog niet uit mijn geheugen zijn gewischt.Als ik onze koe hoedde op het gras langs de wegen of onder de boomen en door een stortregen overvallen werd, dan kwam ze mij tegemoet en dwong me een schuilplaats op in haar wollen rok, dien zij optilde om er mijn hoofd en schouders mede te bedekken.Als ik twist had met een van mijn makkers, liet ze mij mijn hart lucht geven en altijd wist ze mij te troosten en met een enkel woord mij gelijk te geven.", 3))
    print(predict(data,"Господинот и госпоѓата Дарсли од бројот четирина Шимшировата улица со гордост истакнуваа дека тие се сосема нормални, благодарам на прашањето. Од нив вие најмалку би очекувале да бидат вмешани во нешто чудно или таинствено, затоа што тие воопшто не трпеа такви глупости. Господинот Дарсли беше директор на фирмата „Гранингз“, која правеше дупчалки. Тој беше крупен набиен човек, речиси без врат, но затоа пак имаше големи мустаќи. Госпоѓата Дарсли беше слаба и руса и имаше двапати подолг врат од вообичаеното, што ѝ беше од голема корист бидејќи многу време поминуваше извивајќи ја главата преку оградите за да ги шпионира соседите. Дарслиеви имаа синче по име Дадли и, според нивното мислење, на светот немаше подобро момче од него. Дарслиеви имаа сѐ што им требаше, но имаа и една тајна, и повеќе од сѐ се плашеа дека кога-тогаш некој ќе ја открие. Тие не би можеле да поднесат некој да дознае за Потерови. Госпоѓата Потер ѝ беше сестра на госпоѓата Дарсли, но се немаа видено со години; всушност, госпоѓата Дарсли се преправаше дека воопшто нема сестра, затоа што сестра ѝ и нејзиниот неспособен маж беа чиста спротивност на Дарслиеви.", 3))


class HierarchicalClustering:
    def __init__(self, data):
        self.data = data
        self.clusters = [[name] for name in self.data.keys()]        

        self.distances={} #Additional attribute - gives us the distances between countries separately, it is a dictionary of lists
        for c in self.data.keys():
            self.distances[c] = {}
            for c2 in self.data.keys():
                self.distances[c][c2]=self.row_distance(c,c2)

        self.history=[]
        #with this we keep up with how the clusters are being formed in each iteration and what is the distance between them
        

    def row_distance(self, r1, r2):
        #r1 = 'finnish.txt'
        #d1 = {'abc': 3, ...}
        return cosine_dist(self.data[r1], self.data[r2])

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
            #print("cluster ", i+1, "\n")
            printing=self.recursivePlot(self.clusters[i])
            # printing = ['a', 'b', 'c']
            # after join: 'a\nb\nc'
            #print(f"cluster {i+1}:\n")
            print("\n".join(printing))


if __name__ == "__main__":
    #print(terke("abcd efg, hgujkl!fad 12bhwj abcabč",3))
    #odkomenirajte del naloge, ki ga želite pognati
    del2()
    del4()
    del5()
