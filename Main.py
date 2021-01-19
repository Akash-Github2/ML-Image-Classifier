import os

#global variables
keywords = ["car", "motorcycle", "boat", "plane"]
width = 32
height = 32
shouldDownloadImages = False
shouldResizeImages = False
shouldTrainModel = False
shouldMakePredictions = True

def main():
    #imports are here to avoid circular import loop error
    import DatasetManager as dm
    import ComputerVision as compVis
    
    if shouldDownloadImages:
        dm.downloadImages()
        
    if shouldResizeImages:
        input("Press Enter to continue once you finish manual removals...") #waits until manual removals are completed
        os.chdir("training_dataset")
        dm.resizeAllImages()

        datasetX, datasetY = dm.getDatasets()
        dm.storeTrainingDataInFiles(datasetX, datasetY)
        
    if shouldTrainModel:
        if not(shouldResizeImages):
            datasetX, datasetY = dm.retrieveDataFromFiles()
        Theta1, Theta2 = compVis.trainModel(datasetX, datasetY)
        dm.storeWeightsDataInFiles(Theta1, Theta2)
    
    if shouldMakePredictions:
        if not(shouldTrainModel):
            Theta1, Theta2 = dm.retrieveWeightsFromFiles()
        compVis.makePredictions(Theta1, Theta2)
        

if __name__ == "__main__":
    main()