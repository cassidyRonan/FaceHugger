import cv2 as cv;

faceList = ['crowd.jpg','generated.jpg','ovalFace.jpg','Profile_950.jpg']

faceCascade = cv.CascadeClassifier('haarcascade_frontalface.xml');

for f in faceList:
    #Read input image in
    image = cv.imread('FrontalFaces/' + f);

    #Convert to grayscale to reduce variables
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    #Detection
    faces = faceCascade.detectMultiScale(grayImage,1.2,5)

    #Draw rectangle around face
    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    #Display the output
    cv.imshow('Face Finder', image)
    cv.waitKey()