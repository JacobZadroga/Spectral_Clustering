import math
import matplotlib.pyplot as plt
from numpy import linalg as linalg
import numpy as np

class datapt:
    def __init__(this, num, age, rating):
        this.number = num
        this.age = age
        this.rating = rating

    def dist(this, other):
        return math.sqrt(((this.age - other.age)**2) + (((this.rating*5) - (other.rating*5))**2))

    def __str__(this):
    	return "Num: " + str(this.number) + " | age: " + str(this.age) + " | rating: " + str(this.rating)

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
 
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
 
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
 
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    return b_0, b_1

def eigenvalues(nparray, datapts):
	A = np.array(nparray)

	D = np.diag(A.sum(axis=1))

# graph laplacian
	L = D-A
	#print(L)
	#eig_values, eig_vectors = linalg.eig(L)
	#fiedler_pos = np.where(eig_values.real == np.sort(eig_values.real)[1])[0][0]
	#fiedler_vector = np.transpose(eig_vectors)[fiedler_pos]
	#print(np.where(eig_values.real == np.sort(eig_values.real)[1]))
	v, vecs = np.linalg.eig(L)
	#print(vecs)
	vecs = vecs[:,np.argsort(v)]
	v = v[np.argsort(v)]
	plt.figure(2)
	vecs = np.transpose(vecs)
	x=vecs[1]
	mean = np.mean(x)
	for i in range(len(v)):
		#print(vecs[i].real)
		print(x[i])
		#print(y)
		if x[i] <= mean:
			#print("--")
			plt.scatter(x=datapts[i].age,y=datapts[i].rating, c='red')
		else:
			plt.scatter(x=datapts[i].age,y=datapts[i].rating, c='blue')
			#print("++")
	#plt.show()



def main():
    ages = [16,17,19,20,21,22,23,25,27,30,36,37,38,38,39,44,46,47,50,52]
    ratings = [5,5,4,4,4,3,3,3,3,2,1,2,2,3,3,3,4,4,5,5]
    datapts = []
    print(len(ages))
    print(len(ratings))

    for i in range(len(ages)):
    	datapts.append(datapt(i, ages[i], ratings[i]))

    plt.figure(1)
    plt.scatter(x=ages, y=ratings,c='black')
    #print("\n-----------------------------------\n")
    adjacencymatrix = [[0 for y in range(len(ages))] for x in range(len(ages))]
    for i in range(len(ages)):
    	one = datapt(1000,1000,1000)
    	two = datapt(1001,1000,1000)
    	three = datapt(1002,1000,1000)
    	for k in range(len(ages)):
    		if i == k:
    			continue
    		dist = datapts[i].dist(datapts[k])
    		if dist < datapts[i].dist(one):
    			temp=two
    			two=one
    			three=temp
    			one=datapts[k]
    		elif dist < datapts[i].dist(two):
    			three = two
    			two=datapts[k]
    		elif dist < datapts[i].dist(three):
    			three=datapts[k]
    	adjacencymatrix[i][one.number] = one.dist(datapts[i])
    	adjacencymatrix[i][two.number] = two.dist(datapts[i])
    	adjacencymatrix[i][three.number] = three.dist(datapts[i])
    	x = [datapts[i].age, one.age]
    	y = [datapts[i].rating, one.rating]
    	plt.plot(x,y,c="red")
    	x = [datapts[i].age, two.age]
    	y = [datapts[i].rating, two.rating]
    	plt.plot(x,y,c="red")
    	x = [datapts[i].age, three.age]
    	y = [datapts[i].rating, three.rating]
    	plt.plot(x,y,c="red")
    
    #print(adjacencymatrix)
    eigenvalues(adjacencymatrix,datapts)
    print(ratings[10:])
    b, m = estimate_coef(np.array(ages[:10]), np.array(ratings[:10]))
    print(b)
    print(m) 
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = b + m * x_vals
    plt.plot(x_vals, y_vals, '--')

    b, m = estimate_coef(np.array(ages[10:]), np.array(ratings[10:]))
    print(b)
    print(m) 
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = b + m * x_vals
    plt.plot(x_vals, y_vals, '--')
    #plt.figure(1)
    plt.show()

if __name__ == "__main__":
    main()