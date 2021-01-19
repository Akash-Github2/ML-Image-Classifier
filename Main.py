import os

#global variables
keywords = ["car", "motorcycle", "boat", "plane"]
width = 32
height = 32
shouldTrainModel = True

def main():
    #imports are here to avoid circular import loop error
    import DatasetManager as dm
    import ComputerVision as compVis
    
    # functionTester() #to test specific functions in the machine learning stuff
    
    if shouldTrainModel: #Training the model
        # dm.downloadImages()
        os.chdir("training_dataset")
        input("Press Enter to continue once you finish manual removals...") #waits until manual removals are completed
        dm.resizeAllImages()

        datasetX, datasetY = dm.getDatasets()
        dm.storeTrainingDataInFiles(datasetX, datasetY)
        Theta1, Theta2 = compVis.trainModel(datasetX, datasetY)
        dm.storeWeightsDataInFiles(Theta1, Theta2)
    else: #Testing model and making predictions
        Theta1, Theta2 = dm.retrieveWeightsFromFiles()
        compVis.makePredictions(Theta1, Theta2)
        
def functionTester():
    import DatasetManager as dm
    import ComputerVision as compVis
    
    datasetX, datasetY = dm.retrieveDataFromFiles()
    Theta1, Theta2 = compVis.trainModel(datasetX, datasetY)
    dm.storeWeightsDataInFiles(Theta1, Theta2)

if __name__ == "__main__":
    main()