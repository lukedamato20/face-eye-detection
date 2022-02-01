import cv2 as cv

# HAAR cascades for frontal face detection
faceHaarCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
# HAAR cascades for eye detection
eyeHaarCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

# using opencv for webcam capture
capture = cv.VideoCapture(0)

# checking if webcam is available
if not capture.isOpened():
    raise IOError("Webcam is not available")

while 1:
    ret, img = capture.read()
    
    # converting to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # using the haar cascade for face detection
    face = faceHaarCascade.detectMultiScale(gray, 1.3, 5)

    # main loop 
    for (x,y,w,h) in face:
        
        # drawing a green rectangle on the face
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        
        # variables for eye detection (gray and colour)
        roiGray = gray[y:y+h, x:x+w]
        roiColor = img[y:y+h, x:x+w]

        # using the haar cascade for eye detection
        eyes = eyeHaarCascade.detectMultiScale(roiGray)
    
        # loop for eye checking
        for (ex,ey,ew,eh) in eyes:
            # drawing a red rectangle on the eyes
            cv.rectangle(roiColor,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

    cv.imshow('img',img)
    
    # waiting for the ESC key to be pressed for exit
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

capture.release()
cv.destroyAllWindows()