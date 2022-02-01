import cv2 as cv

# getting all 3 cascade.xml files from 3 different accuracies 
cascade1 = cv.CascadeClassifier('data/train1/cascade.xml')
cascade2 = cv.CascadeClassifier('data/train2/cascade.xml')

capture = cv.VideoCapture(0)

if not capture.isOpened():
    raise IOError("Webcam is not available")

while 1:
    ret, img = capture.read()

    # converting to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # detecting object and creating cascade variable
    object1 = cascade1.detectMultiScale(gray, 1.3, 5)
    object2 = cascade2.detectMultiScale(gray, 1.3, 5)

    # creating current cascade variable for easy change of training set 
    current_cascade = object2    
    
    # main loop 
    for (x,y,w,h) in current_cascade:
        
        # drawing a green rectangle on the face
        cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    cv.imshow('img',img)
    
    # waiting for the ESC key to be pressed for exit
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

capture.release()
cv.destroyAllWindows()