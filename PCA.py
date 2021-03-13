import numpy as np
from sklearn.manifold import MDS
from os import listdir
from os.path import join
from unidecode import unidecode #unidecode()
from matplotlib import pyplot as plt
import math


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

def idf(lds, t): #calculates the idf for a tuple t
    counter=1
    for x in lds:
        if t in lds[x]:
            counter+=1
    return math.log((len(lds.keys())/counter),2)

def listOfTuples(lds): #returns a list of pairs of the best 100 tuples based on their idfs, out of all languages
    terke=[]
    result=[]
    for x in lds:
        for y in lds[x]:
            if y not in terke:
                terke.append(y) #remembers which tuples were already added
                result.append([idf(lds,y),y])

    #result.sort(key=lambda x:x[0], reverse=True) #result is a list of the 100 best tuples and their idf's out of all languages, a list of pairs of a float and a string 
    result.sort()#reverse=True)
    l=[]
    for i in range(len(result)):
        l.append(result[i][1])

    return l #l is a list of the top 100 tuples

def prepare_data_matrix():
    """
    Return data in a matrix (2D numpy array), where each row contains triplets
    for a language. Columns should be the 100 most common triplets
    according to the idf measure.
    """
    # create matrix X and list of languages

    lds = {}
    for fn in listdir("clustering"):
        if fn.lower().endswith(".txt"):
            with open(join("clustering", fn), encoding="utf8") as f:
                text = f.read()
                nter = terke(text, n=3)
                lds[fn] = nter
    #print(lds.keys())
    
    #lds is a dictionary of dictionaries: {"slovenian.txt": {"abc":3,"efg":4...}, "macedonian.txt":{"efg":6...},...}
    l=listOfTuples(lds) #list of strings
    #print(l[:100])
    languages = list(lds.keys()) # ['Slo', 'Mac', ]
    # which language represents row number i: languages[i]
    # which row does language s represent: languagues.index(s)
    X=np.zeros([len(languages),100])
    for i in range(len(languages)):
        #print(languages[i])
        count = 0
        for j in range(100):
            if l[j] in lds[languages[i]]:
                X[i,j]=lds[languages[i]][l[j]]
                count += 1
    #    print(count)

    #print([sum(x) for x in X])
   
    return X, languages
    # X, languages = prepare_data_matrix()


def power_iteration(X):
    """
    Compute the eigenvector with the greatest eigenvalue
    of the covariance matrix of X (a numpy array).

    Return two values:
    - the eigenvector (1D numpy array) and
    - the corresponding eigenvalue (a float)
    """
    #X, languages=prepare_data_matrix()
    M=X
    M=M-np.mean(M, axis=0)
    M=np.cov(M, rowvar=False) #the covariance matrix, size 100x100
    x=np.ones(len(M)) #a random starting vector composed of 100 ones, it only cant be of all zeros
    difference=np.ones(len(x))

    #print(np.linalg.norm(difference))
    while np.linalg.norm(difference) >= 10**-5: #we iterate until the difference between the previous and the new x is really small, lets say 10^-5
        #print(x.T.shape)
        oldx=x
        z=M.dot((x.T))
        x=z.T
        x=x/np.linalg.norm(x)
        difference=np.linalg.norm(oldx-x)
    #the x that we get at the end of this loop is our eigenvector

    #print(x.dot(M).shape)
    #print(x.shape)
    y=(x.dot(M)).dot(x.T) #y is the corresponding eigenvalue to the eigenvector x
    
    return x, y

def power_iteration_two_components(X):
    """
    Compute first two eigenvectors and eigenvalues with the power iteration method.
    This function should use the power_iteration function internally.

    We calculate e.vectors and e.values of X by calling the previous function, and after the first iteration
    we first neeed to subtract PCA1 from X, and then find the second component PCA2

    Return two values:
    - the two eigenvectors (2D numpy array, each eigenvector in a row) and
    - the corresponding eigenvalues (a 1D numpy array)
    """
    evector1, evalue1=power_iteration(X)
    X=X-np.outer(evector1.dot(X.T), evector1) #project each point(aka each row of X) on the first euginevector and then substract that from each row
    evector2, evalue2=power_iteration(X)

    eigenvectors=np.vstack([evector1,evector2])
    eigenvalues=np.array([evalue1,evalue2])

    return eigenvectors, eigenvalues

def project_to_eigenvectors(X, vecs):
    """
    Project matrix X onto the space defined by eigenvectors.
    The output array should have as many rows as X and as many columns as there
    are vectors.
    """

    return (X-np.mean(X, axis=0)).dot(np.transpose(vecs)) #PCA assumes that the data is centered, so we need to do that before doing the calculations

def total_variance(X):
    """
    Total variance of the data matrix X. You will need to use for
    to compute the explained variance ratio.
    """
    return np.var(X, axis=0, ddof=1).sum()


def explained_variance_ratio(X, eigenvectors, eigenvalues):
    """
    Compute explained variance ratio.
    """
    return np.sum(eigenvalues)/total_variance(X)


def plot_PCA():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of PCA on languages data.
    """
    X, languages = prepare_data_matrix()
    #print(X)
    eigenvectors, eigenvalues=power_iteration_two_components(X)
    explain = explained_variance_ratio(X, eigenvectors, eigenvalues)
    X=project_to_eigenvectors(X,eigenvectors)

    #print(X)
    plt.title('Explained variance: %.3f' % explain)
    plt.scatter(X[:,0], X[:,1])
    for i in range(len(X)):
        plt.text(X[i,0], X[i,1], languages[i][:3])
    plt.show()

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

def plot_MDS():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of MDS on languages data.

    Use sklearn.manifold.MDS and explicitly run it with a distance
    matrix obtained with cosine distance on full triplets, like
    in the previous homework.
    """
    lds = {}  #lds is a dictionary of dictionaries: {"slovenian.txt": {"abc":3,"efg":4...}, "macedonian.txt":{"abc":5,"efg":6...},...}
    for fn in listdir("clustering"):
        if fn.lower().endswith(".txt"):
            with open(join("clustering", fn), encoding="utf8") as f:
                text = f.read()
                nter = terke(text, n=3)
                lds[fn] = nter
                
    distances={} #a dictionary of dictionaries that saves the distances between a language and all other languages
    
    for x in lds.keys():
        distances[x]={}
        for y in lds.keys():
            if x == y: distances[x][y]=0.0
            else: distances[x][y]=cosine_dist(lds[x],lds[y])

    dst=np.zeros([len(lds.keys()), len(lds.keys())])
    i=0
    j=0
    for x in lds.keys():
        j=0
        for y in lds.keys():
            dst[i,j]=distances[x][y]
            j+=1
        i+=1

    X, languages = prepare_data_matrix()

    transformer = MDS(n_components=2, dissimilarity='precomputed')
    transformed = transformer.fit_transform(dst)

    plt.scatter(transformed [:,0], transformed [:,1])
    for i in range(len(transformed)):
        plt.text(transformed[i,0], transformed[i,1], languages[i][:3])
    plt.show()

if __name__ == "__main__":
   plot_MDS()
   plot_PCA()