import cv2
 
def resizeImg(fileDir, imageName, newWidth, newHeight):
    origImgFilePath = fileDir + "/" + imageName
    indOfDot = imageName.find('.')
    newImgFilePath = fileDir + "/" + imageName[:indOfDot] + "_resized" + imageName[indOfDot:]
    
    img = cv2.imread(origImgFilePath, cv2.IMREAD_GRAYSCALE)
    # print('Original Dimensions : ',img.shape)
    dim = (newWidth, newHeight)
    # resize image
    resizedImg = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(newImgFilePath, resizedImg)
    
def getPixelArr(fileDir, imageName, width, height):
    filePath = fileDir + "/" + imageName
    img = cv2.imread(filePath)
    arr = [] #stores grayscale values in a 1D array
    for i in range(height):
        for j in range(width):
            grayInt = img[i, j][0]
            arr.append(grayInt)
    return arr
    