import cv2
import random as rand
import math
import numpy as np
import os
import arrow
from Main import keywords, width, height #imports the public variables to the class

#Loops through all the testing data set and calls predictObject() method
def makePredictions(Theta1, Theta2): #starts in the main directory
    os.chdir("testing_dataset")
    imageNames = os.popen("ls").read().strip().split("\n") #retrieves testing images from folder
    correctPredictions = 0
    
    for imageName in imageNames:
        #predictObject() - from here, all the neural network feedforwarding will occur
        prediction = predictObject(os.getcwd(), imageName, Theta1, Theta2)
        actual = getActualCategory(imageName)
        # print("Actual: " + actual + " ; Prediction: " + prediction, end=" ")
        if prediction == actual:
            # print("✓")
            correctPredictions += 1
        # else:
        #     # print("✖")
            
    accuracy = (correctPredictions * 100) / len(imageNames)
    print("Accuracy: " + str(round(accuracy, 3)) + "%")
        
#returns the actual category of the image
def getActualCategory(imageName):
    categoryName = ""
    for ch in imageName:
        if ch.isdigit():
            break
        categoryName += ch
    return categoryName


#returns the category as a string
def predictObject(fileDir, imageName, Theta1, Theta2):
    origImgFilePath = fileDir + "/" + imageName
    # print(origImgFilePath)
    img = cv2.imread(origImgFilePath, cv2.IMREAD_GRAYSCALE)
    dim = (width, height)
    resizedImg = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    pixelArr = getPixelArr(resizedImg)
    a3 = getFeedForwardEndResult(Theta1, Theta2, [pixelArr])
    maxValue = max(a3)
    indOfMax = np.where(a3 == maxValue)[0][0]
    return keywords[indOfMax]

#This X is different than the others (only 1 row and it's the testing data)
def getFeedForwardEndResult(Theta1, Theta2, X):
    X = np.array(X)
    onesCol4X = np.ones([1, 1], dtype = float)
    X2 = np.insert(X, 0, onesCol4X, axis=1)
    
    a1 = np.array(X2) #one row of X2 (one training example at a time) (1x901)
    z2 = a1.dot(Theta1.transpose()) #1x40
    a2 = sigmoid(z2)
    onesColForA2 = np.ones([1, 1], dtype = float)
    a2 = np.insert(a2, 0, onesColForA2, axis=1) #adds the bias unit 1 to the front of a2
    z3 = a2.dot(Theta2.transpose()) #1x4
    a3 = sigmoid(z3) #1x4
    # print(a3[0])
    return a3[0]
        

def getPixelArr(resizedImg):
    arr = [] #stores grayscale values in a 1D array
    for i in range(height):
        for j in range(width):
            grayInt = resizedImg[i, j]
            arr.append(grayInt)
    return arr

#----Machine Learning Section below----#

#L_in -> incoming connections; L_out -> outgoing connections
def randInitializeWeight(L_in, L_out):
    epsilon = 0.09
    outputMatrix = (np.random.rand(L_out, L_in + 1) * 2 * epsilon) - epsilon
    return outputMatrix

#Trains the model and will return the trained Theta1 and Theta2
def trainModel(datasetX, datasetY):
    print("Training Model Started ...")
    start_time = arrow.utcnow()
    input_layer_size = len(datasetX[0])
    hidden_layer_size = 80
    output_layer_size = len(keywords)
    
    Theta1_init = randInitializeWeight(input_layer_size, hidden_layer_size)
    Theta2_init = randInitializeWeight(hidden_layer_size, output_layer_size)
    
    Theta1, Theta2 = minJWithGradDescent(Theta1_init, Theta2_init, datasetX, datasetY)
    print("Training Model Completed!")
    end_time = arrow.utcnow()
    print("Time to Train (minutes):", str((end_time-start_time).total_seconds()/60))
    return Theta1, Theta2

def sigmoid(z):
    z[z < -700] = -700 #After this becomes positive, it will be infinity which is out of bounds
    return 1 / (1 + np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z) * (1-sigmoid(z))

#Cost function J
#X is the datasetX
#y is the datasetY (nx1 dimensional matrix of strings) -> will be converted to Y (an nxlen(keywords) dimensional matrix of 0s and 1s)
#Theta1: 40x901, Theta2: 4x41
def J(Theta1, Theta2, X, y):
    X = np.array(X)
    J = 0
    m = len(X)
    lambdaVal = 3
    
    onesCol4X = np.ones([1, len(X)], dtype = float)
    X2 = np.insert(X, 0, onesCol4X, axis=1)
    Y = convYToNewFormat(y) # n x len(keywords) matrix
    # print("X Dim: ", str(X.shape))
    # print("X2 Dim: ", str(X2.shape))
    # print("Y Dim: ", str(Y.shape))
    
    for i in range(m):
        a1 = np.array([X2[i]]) #one row of X2 (one training example at a time) (1x901)
        z2 = a1.dot(Theta1.transpose()) #1x40
        a2 = sigmoid(z2)
        onesColForA2 = np.ones([1, 1], dtype = float)
        a2 = np.insert(a2, 0, onesColForA2, axis=1) #adds the bias unit 1 to the front of a2
        z3 = a2.dot(Theta2.transpose()) #1x4
        a3 = sigmoid(z3) #1x4
        
        innerAns = -Y[i] * np.log(a3) - ((1-Y[i]) * np.log(1-a3))
        J += sum(sum(innerAns))
        
    J /= m
    
    #Regularization Part
    A = np.array(Theta1[:][1:])
    B = np.array(Theta2[:][1:])
    reg = sum(sum(A*A)) + sum(sum(B*B))
    reg = reg * lambdaVal / (2 * m) 
    
    return J        
        

#gets the derivative of all thetas, will return Theta1_grad and Theta2_grad
def getGrad(Theta1, Theta2, X, y):
    X = np.array(X)
    m = len(X)
    lambdaVal = 3
    onesCol4X = np.ones([1, len(X)], dtype = float)
    X2 = np.insert(X, 0, onesCol4X, axis=1)
    
    Y = convYToNewFormat(y) # n x len(keywords) matrix
    delta1 = np.zeros([len(Theta1), len(Theta1[0])], dtype = float)
    delta2 = np.zeros([len(Theta2), len(Theta2[0])], dtype = float)
    
    for i in range(m):
        a1 = np.array([X2[i]])
        z2 = a1.dot(Theta1.transpose()) #1x40
        a2 = sigmoid(z2)
        onesColForA2 = np.ones([1, 1], dtype = float)
        a2 = np.insert(a2, 0, onesColForA2, axis=1) #adds the bias unit 1 to the front of a2
        z3 = a2.dot(Theta2.transpose()) #1x4
        a3 = sigmoid(z3) #1x4
        
        currY = np.array([Y[i]]) #1x4
        errorVector3 = a3 - currY #1x4
        errorVector2 = errorVector3.dot(Theta2) #1x41
        errorVector2 = np.array([errorVector2[0][1:]]) * sigmoidGradient(z2)
        delta1 += errorVector2.transpose().dot(a1)
        delta2 += errorVector3.transpose().dot(a2)
        
    Theta1_grad = delta1 / m
    Theta2_grad = delta2 / m
    Theta1_grad[:][1:] += (lambdaVal/m) * Theta1[:][1:]
    Theta2_grad[:][1:] += (lambdaVal/m) * Theta2[:][1:]
    return Theta1_grad, Theta2_grad


#Uses gradient descent to reduce the J value and then returns the Theta1 and Theta2
def minJWithGradDescent(Theta1, Theta2, X, y):
    alpha = 0.004 #learning rate
    Theta1_grad, Theta2_grad = getGrad(Theta1, Theta2, X, y)
    
    for i in range(6501): #deriv tries to get as small as possible
        if i % 50 == 0:
            jVal = J(Theta1, Theta2, X, y)
            print(i, jVal)
            if jVal < 1.34:
                alpha = 0.0015
        if i < 400 and alpha != 0.0015:
            alpha = 0.004
        elif i < 3850 and alpha != 0.0015:
            alpha = 0.003
        elif alpha != 0.0015:
            alpha = 0.002
        Theta1_grad, Theta2_grad = getGrad(Theta1, Theta2, X, y)
        Theta1 -= (alpha * Theta1_grad)
        Theta2 -= (alpha * Theta2_grad)
        
    return Theta1, Theta2

#convert y to Y (an nxlen(keywords) dimensional matrix of 0s and 1s)
def convYToNewFormat(y):
    Y = np.zeros([len(y), len(keywords)], dtype = float)
    
    for i in range(len(y)):
        Y[i][keywords.index(y[i][0])] = 1
    return Y


#Only for initial testing to make sure back-propagation works
def getAvgNumerGrad(Theta1, Theta2, X, y):
    numgrad1 = np.zeros([len(Theta1), len(Theta1[0])], dtype = float)
    numgrad2 = np.zeros([len(Theta2), len(Theta2[0])], dtype = float)
    perturb1 = np.zeros([len(Theta1), len(Theta1[0])], dtype = float)
    perturb2 = np.zeros([len(Theta2), len(Theta2[0])], dtype = float)
    e = 1E-4
    for i in range(len(Theta1)):
        for j in range(len(Theta1[0])):
            print(i, j, end='\r')
            perturb1[i][j] = e
            loss1 = J(Theta1 - perturb1, Theta2, X, y)
            loss2 = J(Theta1 + perturb1, Theta2, X, y)
            numgrad1[i][j] = (loss2 - loss1) / (2*e)
            perturb1[i][j] = 0
    print("", end='\r')
    for i in range(len(Theta2)):
        for j in range(len(Theta2[0])):
            print(i, j, end='\r')
            perturb2[i][j] = e
            loss1 = J(Theta1, Theta2 - perturb2, X, y)
            loss2 = J(Theta1, Theta2 + perturb2, X, y)
            numgrad2[i][j] = (loss2 - loss1) / (2*e)
            perturb2[i][j] = 0
    
    avg1 = sum(sum(numgrad1)) / (40*901)
    avg2 = sum(sum(numgrad2)) / (4*41)
    return avg1, avg2 