import cv2
cam = cv2.VideoCapture(0)
while(True):
    ret,frame = cam.read()
    if ret ==True:
        cv2.imshow('frame',frame)
        cv2.imwrite('photo.png',frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
cam.release()
cv2.destroyAllWindows