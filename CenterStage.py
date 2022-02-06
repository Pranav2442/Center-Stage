import cv2 as cv
import numpy as np





zooml = 0
def func(v):
        global zooml
        zooml = v


def Crop(img, start, end):
    startx, starty = start
    endx, endy = end
    o_w, o_h = img.shape[:2][::-1]
    w, h = (endx - startx, endy - starty)
    
    if (startx < 0):
        startx = 0; endx = w

    if (starty < 0):
        starty = 0; endy = h

    if (endx > o_w):
        endx = o_w; startx = o_w - w

    if (endy > o_h):
        endy = o_h; starty = o_h - h

    return img[starty:endy, startx:endx]

def ImgRatio(size, ratio):
    w, h = size
    ratio_w, ratio_h = ratio

    if (h / ratio_h * ratio_w > 0):
        # change width
        return (int(h / ratio_h * ratio_w), h)
    else:
        # change height
        return (w, int(w / ratio_w * ratio_h))

def changeRatio(img, ratio):
    w, h = img.shape[:2][::-1]
    outputW, outputH = ImgRatio((w, h), ratio)

    startX = (outputW - w) // 2
    startY = (outputH - h) // 2

    output = np.full((outputH, outputW, 3), 0, np.uint8)
    output[startY:startY + h, startX:startX + w] = img

    return output

CasScale = 1.3
InterAmount = 0.10
Thres = 0.1
EyeOffset = np.array((0, 10))

Crop_Ratio = np.array((4, 3))
Crop_Aspect = np.array((16, 9))

Csize = Crop_Ratio * 135
Osize = ImgRatio(Csize, Crop_Aspect)


ZOOM_GLO = True




    
cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")

SCREEN_SIZE = np.array(cap.read()[1].shape[:2][::-1], int)

pos = (SCREEN_SIZE - Csize) // 2

    
        
if (ZOOM_GLO):
        cv.imshow("Window", np.zeros((Osize[1], Osize[0], 3)))
        cv.createTrackbar("Zoom ", "Window", 0, 40, func)

        while(True):
            ret, img = cap.read()
            
            face = face_cascade.detectMultiScale(img, CasScale, 5)
            eyes = eye_cascade.detectMultiScale(img, CasScale, 5)
            

            
            if ((len(face) + len(eyes))> 0):
                mid = np.zeros((2), int)
                
                for _ in eyes:
                    x,y,w,h=_
                    mid+=np.array((x + w // 2, y + h // 2)) + EyeOffset
                
                for j in face:
                    x,y,w,h=j
                    mid += np.array((x + w // 2, y + h // 2))
                
                mid=mid//(len(face)+len(eyes))
                TopLeft=(int(mid[0]-Csize[0]//2),int(mid[1]-Csize[1]//2))
                threshold=Thres * (1 * zooml / 100)
                
                M= np.linalg.norm(np.array(pos + InterAmount * (TopLeft - pos), int) - pos)
                if(M>threshold):
                    pos = np.array(pos + InterAmount * (TopLeft - pos), int)
                else:
                    pos

            
            

            
            zooms = np.array(pos + Csize * zooml / 100, int)
            zoome = np.array(pos + Csize - Csize * zooml / 100, int)
            img = Crop(img, zooms, zoome)

            
            img = cv.resize(img, Csize)
            img = changeRatio(img, Osize)

            
            
            img=cv.flip(img,1)
            cv.imshow("Window", img)

            
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            
            
            if (cv.waitKey(1)==27):
                break

        cap.release()

        cv.destroyAllWindows()

