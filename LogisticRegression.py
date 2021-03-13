import numpy as np
from matplotlib import pyplot
from scipy.optimize import fmin_l_bfgs_b
import math


def draw_decision(X, y, classifier, at1, at2, grid=50):
    points = np.take(X, [at1, at2], axis=1)
    maxx, maxy = np.max(points, axis=0)
    minx, miny = np.min(points, axis=0)
    difx = maxx - minx
    dify = maxy - miny
    maxx += 0.02 * difx
    minx -= 0.02 * difx
    maxy += 0.02 * dify
    miny -= 0.02 * dify

    pyplot.figure(figsize=(8, 8))

    for c,(x,y) in zip(y,points):
        pyplot.text(x, y, str(c), ha="center", va="center")
        pyplot.scatter([x], [y], c=["b", "r"][int(c) != 0], s=200)

    num = grid
    prob = np.zeros([num, num])
    for xi, x in enumerate(np.linspace(minx, maxx, num=num)):
        for yi, y in enumerate(np.linspace(miny, maxy, num=num)):
            # probability of the closest example
            diff = points - np.array([x, y])
            dists = (diff[:, 0]**2 + diff[:, 1]**2)**0.5  # euclidean
            ind = np.argsort(dists)
            prob[yi,xi] = classifier(X[ind[0]])[1]

    pyplot.imshow(prob, extent=(minx, maxx, maxy, miny), cmap="seismic")

    pyplot.xlim(minx, maxx)
    pyplot.ylim(miny, maxy)
    pyplot.xlabel(at1)
    pyplot.ylabel(at2)

    pyplot.show()

def transformProb(results, threshold=0.5):
    r=[]
    for i in range(len(results)):
        if results[i][1]>=threshold:
            r.append(1)
        else:
            r.append(0)
    return r

def load(name):
    """
    Odpri datoteko. Vrni matriko primerov (stolpci so znacilke)
    in vektor razredov.
    """
    data = np.loadtxt(name)
    X, y = data[:, :-1], data[:, -1].astype(np.int)
    return X, y


def h(x, theta):
    """
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """
    # ... dopolnite (naloga 1)

    return 1/(1+np.exp(-np.dot(theta, x)))
    #return math.exp(np.dot(theta, x))/(1+math.exp(-np.dot(theta, x)))

def cost(theta, X, y, lambda_):
    """
    Vrednost cenilne funkcije.
    """
    # ... dopolnite (naloga 1, naloga 2)

    sum=0.0
    for i in range(len(X)):
        sum=sum+(y[i]*np.log(h(X[i],theta))+(1-y[i])*np.log(1-h(X[i],theta)))
    sumT=0.0
    for j in range(len(X[0])):
        sumT=sumT+theta[j]**2
    sumT = 2 * lambda_ * sumT / len(X[0])
    return (-sum + sumT)/ len(X)

def grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije. Vrne 1D numpy array v velikosti vektorja theta.
    """
    # ... dopolnite (naloga 1, naloga 2)

    grad=np.zeros(len(X[0]))
    for i in range(len(X[0])):
        sum=0.0
        for j in range(len(X)):
            sum=sum+(y[j]-h(X[j],theta))*X[j][i]
        grad[i]=(-sum + 2*lambda_*(theta[i]*2)/len(X[0]))/len(X)

    return grad


def num_grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije izracunan numericno.
    Vrne numpyev vektor v velikosti vektorja theta.
    Za racunanje gradienta numericno uporabite funkcijo cost.
    """
    # ... dopolnite (naloga 1, naloga 2)
    # uses the formula (f(x+h)-f(x-h))/2*h where f is cost function, x is thetai, and h is something small
    num_grad = np.zeros(len(X[0]))
    h=0.00001
    for i in range(len(X[0])):
        t1=np.copy(theta)
        t2=np.copy(theta)
        t1[i]=theta[i]-h
        t2[i]=theta[i]+h
        num_grad[i]=(cost(t2,X,y,lambda_)-cost(t1,X,y,lambda_))/(2*h)

    return num_grad

class LogRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x = np.hstack(([1.], x))
        p1 = h(x, self.th)  # verjetno razreda 1
        return [1-p1, p1]


class LogRegLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X = np.hstack((np.ones((len(X),1)), X))

        # optimizacija
        theta = fmin_l_bfgs_b(
            cost,
            x0=np.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        return LogRegClassifier(theta)


def test_learning(learner, X, y):
    """ vrne napovedi za iste primere, kot so bili uporabljeni pri učenju.
    To je napačen način ocenjevanja uspešnosti!

    Primer klica:
        res = test_learning(LogRegLearner(lambda_=0.0), X, y)
    """
    c = learner(X,y)
    results = [c(x) for x in X]

    return results

def test_cv(learner, X, y, k=5):
    # print('X', X)
    # print('y', y)
    """"
    Primer klica:
        res = test_cv(LogRegLearner(lambda_=0.0), X, y)
    ... dopolnite (naloga 3)
    """
    rows, columns = X.shape
    shuffled = np.random.permutation(rows) #we take a random permutation with as many elements as there are examples in X
    sets = np.array_split(shuffled,k) #we split that array of indexes in k-many sets
    predictions = np.zeros((rows,2))
    iterations = 0
    for i in range(k):
        indexes = [j for j in shuffled if not j in sets[i]] #we take all indexes from shuffled except those that are in the set 'sets[i]'
        Xtrain = X[indexes,:]
        ytrain = y[indexes]
        Xtest = X[sets[i],:]
        classifier = learner(Xtrain,ytrain)
        for j in range(len(sets[i])):
            predictions[shuffled[iterations]] = classifier(Xtest[j])
            iterations = iterations + 1
    return predictions

def CA(real, predictions):
    p=transformProb(predictions)
    counter=0 #counts how many of the predictions are equal to the real values
    for i in range(len(predictions)):
        if real[i]==p[i]:
            counter=counter+1

    return counter/len(predictions)

def TPR(real, predictions, threshold):
    p=transformProb(predictions, threshold)
    counterP=0 #how many positivies there are in the real values
    counterTP=0 #how many positivies were correctly predicted
    for i in range(len(predictions)):
        if real[i]==1:
            counterP=counterP+1
            if p[i]==1:
                counterTP=counterTP+1

    if counterP == 0:
        return 0

    return counterTP / counterP

def FPR(real, predictions, threshold):
    p = transformProb(predictions, threshold)
    counterN = 0  #how many negatives there are in the real values
    counterFP = 0  #how many positivies were wrongly predicted
    for i in range(len(predictions)):
        if real[i] == 0:
            counterN = counterN + 1
            if p[i] == 1:
                counterFP = counterFP + 1

    if counterN == 0:
        return 0

    return counterFP / counterN

def lengthOfSide(x1, y1, x2, y2): #calculates the distance between 2 points
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def area(x1,y1,x2,y2): #calculates a part of the total area under the ROC curve, the part is in a
                        #form of a triangle and a rectangle together, (x1,y1) is always a point before (x2, y2)
    # the triangle has the points (x1,y1), (x2,y1) and (x2,y2)
    # area of a triangle: a*h/2
    # the rectangle has the points (x1, 0), (x2,0), (x1,y1) and (x2, y1)
    # area of a rectangle a*b
    a = lengthOfSide(x1, y1, x2, y1)
    b = lengthOfSide(x1, 0, x1, y1)
    h = lengthOfSide(x2, y1, x2, y2)
    return a*h/2 + a*b

def AUC(real, predictions):
    #predictions=np.concatenate([predictions,real], axis=1)
    predictions = np.column_stack((predictions,real))
    predictions = predictions[predictions[:, 0].argsort()]
    #print(predictions[:,2])
    # we put real and predictions into one matrix, and then we sort the whole matrix in ascending order by the first column
    x=[0]
    y=[0]
    # x and y are the coordinates (TPR, FPR) for each threshold and on the beginning we have (0,0) for threshold 1, and on the end (1,1) for threshold 0
    for i in range(len(predictions)):
        x.append(FPR(predictions[:,2],predictions,predictions[i][1]))
        y.append(TPR(predictions[:,2],predictions,predictions[i][1]))
    #print("X: ")
    #print(x)
    #print("Y: ")
    #print(y)
    pyplot.plot(x, y)
    pyplot.xlabel('FPR')
    pyplot.ylabel('TPR')
    pyplot.title('ROC')
    pyplot.show()

    sum=0.0
    for i in range(1, len(x)):
        sum=sum+area(x[i-1], y[i-1], x[i], y[i])

    return sum

def del2():
    X, y = load('reg.data')

    learner1 = LogRegLearner(lambda_=0.00001)
    classifier1 = learner1(X, y)
    draw_decision(X, y, classifier1, 0, 1)

    learner2 = LogRegLearner(lambda_=0.1)
    classifier2 = learner2(X, y)
    draw_decision(X, y, classifier2, 0, 1)

    learner3 = LogRegLearner(lambda_=0.6)
    classifier3 = learner3(X, y)
    draw_decision(X, y, classifier3, 0, 1)


def del3():
    X, y = load('reg.data')

    lambdas=[0.00001, 0.0001, 0.001, 0.01, 0.1]
    for l in lambdas:
        learner=LogRegLearner(lambda_=l)
        predictionsTL=test_learning(learner, X, y) #list of predictions gotten with test_learning
        predictionsTCV=test_cv(learner, X, y, 5) #list of predictions gotten with test_cv
        print("lambda =", l)
        print("Tocnost z test_learning:", CA(y,predictionsTL))
        print("Tocnost z test_cv:", CA(y, predictionsTCV), flush=True)

def del4():
    #odpiranje GDS350
    X, y = load('GDS360.data')
    #print(X)
    #print(y)
    #print(X.shape, y.shape)
    #print(X, y)
    learner=LogRegLearner(lambda_=0.0)
    predictions=test_cv(learner, X, y, 5)
    #print(predictions)
    auc=AUC(y, predictions)
    print("The area under the curve is:", auc)


def del5():
    # ... dopolnite
    pass


if __name__ == "__main__":
    X, y = load('reg.data')
    """
    # Primer uporabe, ki bo delal, ko implementirate cost in grad

    learner = LogRegLearner(lambda_=0.0)
    classifier = learner(X, y) # dobimo model

    napoved = classifier(X[0])  # napoved za prvi primer
    print(napoved)

    # izris odlocitev
    draw_decision(X, y, classifier, 0, 1)
    """

    # testing of del1
    # Če gradnjo modela logistične regresije brez regularizacije (lambda=0) poženete na celotnih podatkih reg.data, vam
    # mora zgrajen model vse primere uvrstiti prav tako, kot je zapisano v reg.data.
    results = test_learning(LogRegLearner(lambda_=0.0), X, y)
    print(np.array_equal(transformProb(results), y))
    #print(transformProb(results))
    #print(y)

    del2()
    del3()
    del4()

