import cv2 as cv

# getting all images
# img1 = cv.imread('test_images/glasses_dark_noreflection.jpg')
img2 = cv.imread('test_images/glasses_dark_reflection.jpg')
img3 = cv.imread('test_images/glasses_light_noreflection.jpg')
# img4 = cv.imread('test_images/glasses_light_reflection.jpg')
img5 = cv.imread('test_images/noglasses_dark.jpg')
img6 = cv.imread('test_images/noglasses_light.jpg')

# variale for the current image for easy swapping of image during testing
current_image = img6

# converting current image to grayscale
gray_img = cv.cvtColor(current_image, cv.COLOR_BGR2GRAY)

# getting the 2 cascade.xml files from 2 different accuracies 
cascade1 = cv.CascadeClassifier('data/train1/cascade.xml')
object1 = cascade1.detectMultiScale(gray_img)
cascade2 = cv.CascadeClassifier('data/train2/cascade.xml')
object2 = cascade2.detectMultiScale(gray_img)

# variale for the current cascade classifier for easy swapping of image during testing
current_cascade = object1

for (x, y, w, h) in current_cascade:
    # draw on image
    cv.rectangle(current_image, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # saving the new image
    cv.imwrite('saved_images/test.jpg', current_image)