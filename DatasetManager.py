from bing_image_downloader import downloader
import ImageConverter as ic
import os
import Util as util
import pickle
import cv2
from Main import keywords, width, height #imports the public variables to the class

def downloadImages():
    for keyword in keywords:
        downloader.download(keyword, limit=200, output_dir='training_dataset', adult_filter_off=True, force_replace=True, timeout=120)

def resizeAllImages():
    for keyword in keywords:
        os.chdir(keyword)
        print(keyword[0].upper() + keyword[1:])
        images = os.popen("ls").read().strip().split("\n")
        
        #just to show the progress bar
        countImages = 0
        for image in images:
            if (".j" in image.lower() or ".p" in image.lower()) and "resized" not in image:
                countImages += 1
        util.printProgressBar(0, countImages, prefix="Resizing")
        #------#
        
        i = 0
        for image in images:
            if (".j" in image.lower() or ".p" in image.lower()) and "resized" not in image:
                ic.resizeImg(os.getcwd(), image, width, height)
                #just to show progress bar
                i+=1
                util.printProgressBar(i, countImages, prefix="Resizing")
        print()
        
        os.chdir("..")
        
def getDatasets():
    
    datasetX = []
    datasetY = [] #needs to be a nx1 dimensional matrix (column vector), not an array (because otherwise it can't use matrix ops)
    for keyword in keywords:
        os.chdir(keyword)
        
        images = os.popen("ls").read().strip().split("\n")
        for image in images:
            if (".j" in image.lower() or ".p" in image.lower()) and "resized" in image:
                datasetX.append(ic.getPixelArr(os.getcwd(), image, width, height))
                datasetY.append([keyword])
        
        os.chdir("..")
        
    os.system('find . -name "*resized*" -delete') #deletes the resized images
    addTrainingToTestingSet()
    os.chdir("..") #move out of training_dataset directory
    
    print("Dimensions of X: ", len(datasetX), len(datasetX[0]))
    print("Length of Y: ", len(datasetY), len(datasetY[0]))
    return datasetX, datasetY

#Automatically adds the training set data to the testing set
def addTrainingToTestingSet(): #starts in the training_dataset directory
    
    os.chdir("..")
    os.chdir("testing_dataset")
    testingDataFilePath = os.getcwd()
    #moves back to training_dataset
    os.chdir("..")
    os.chdir("training_dataset") 
    
    for keyword in keywords:
        os.chdir(keyword)
        i = 1
        images = os.popen("ls").read().strip().split("\n")
        for imageName in images:
            if ".j" in imageName.lower() or ".p" in imageName.lower():
                img = cv2.imread(os.getcwd() + "/" + imageName, cv2.IMREAD_UNCHANGED)
                indOfDot = imageName.find('.')
                imgExtension = imageName[indOfDot:]
                
                finalFilePath = testingDataFilePath + "/" + keyword + str(i) + imgExtension
                #writes the data to the testing dataset
                cv2.imwrite(finalFilePath, img)
                i+=1
        os.chdir("..")


#Storage and Retrieval

def storeTrainingDataInFiles(datasetX, datasetY):
    with open('training_dataset.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(datasetX, filehandle)
        pickle.dump(datasetY, filehandle)
        
def retrieveDataFromFiles():
    with open('training_dataset.data', 'rb') as filehandle:
        # read the data as binary data stream
        datasetX = pickle.load(filehandle)
        datasetY = pickle.load(filehandle)
    print("Dimensions of X: ", len(datasetX), len(datasetX[0]))
    print("Length of Y: ", len(datasetY), len(datasetY[0]))
    return datasetX, datasetY

def storeWeightsDataInFiles(Theta1, Theta2):
    with open('weights.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(Theta1, filehandle)
        pickle.dump(Theta2, filehandle)
        
def retrieveWeightsFromFiles():
    with open('weights.data', 'rb') as filehandle:
        # read the data as binary data stream
        Theta1 = pickle.load(filehandle)
        Theta2 = pickle.load(filehandle)
    print("Dimensions of Theta1: ", len(Theta1), len(Theta1[0]))
    print("Dimensions of Theta2: ", len(Theta2), len(Theta2[0]))
    return Theta1, Theta2