import cv2
import numpy as np
import time
from PIL import Image
import modpredict as predict


def nothing(x):    
    print("nothing")

img = cv2.imread("imgs/xori.png")
gray = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )

cv2.namedWindow("thresh")
cv2.createTrackbar("low","thresh",0,255, nothing )
cv2.createTrackbar("heigh","thresh",0,255, nothing )

#cv2.imshow("image", img)    

while 1:
    wlow  = cv2.getTrackbarPos("low", "thresh")
    wheigh = cv2.getTrackbarPos("heigh", "thresh")
        
    #blur = cv2.medianBlur( gray ,3)
    blur= cv2.GaussianBlur( gray, (3,3), 0)
    
    ret,thresh = cv2.threshold(blur, 9, 255, cv2.THRESH_BINARY)
    
    #ret,thresh = cv2.threshold(blur, wlow, wheigh, cv2.THRESH_BINARY_INV)
    #kernel = np.ones((3,3),np.uint8) 
    
    #opening = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel,iterations=2)
    
    cv2.imshow("thresh" ,thresh)
    image , contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    xywh=[]
    for idx,cnt in enumerate(contours):
        approx=cv2.approxPolyDP( cnt,30,True)    
        area = cv2.contourArea(cnt)

        if  area > 4000 and area < 6000:              
            #print( contours[idx] )
            #print( approx )

            x,y,w,h = cv2.boundingRect( cnt )   
            xywh.append( (x,y,w,h) )

            #img = cv2.rectangle( img , (x,y) , (x+w,y+h),(0,255,0,2),2)
            #img = cv2.circle( img, (x,y), 2, (0,255,0) , 2)
            
    
            #cv2.circle( img, (contours[idx][0],contours[idx][1]),2, (0,255,0),2)            
            #cv2.drawContours( img, contours, idx, (0,255,0),2)            
        # if  area > 30000 and area < 40000 :                    
        #     if len(approx) in [4] : 
        #         print( idx)      0
        #         print( area)   
        #         index.append( idx)
        #         cv2.drawContours( img, contours, idx, (0,255,0),2)

    #排序y,由小至大
    yL = sorted( xywh , key=lambda x:x[1])
    #print( yL )
    for x,y,w,h in yL:
        img = cv2.rectangle( img , (x,y) , (x+w,y+h),(0,255,0,2),2)
        #img = cv2.circle( img, (x,y),2,(0,0,255),2)

    #show split 
    top1=[]
    top2=[]
    top3=[]
    top4=[]
    
    for i,(x,y,w,h) in enumerate(yL):        
        #影像加強
        my=thresh[ y:y+h,x:x+w]        
        #kernel = np.ones((3,3),np.uint8) 
        #my = cv2.morphologyEx(my, cv2.MORPH_ERODE, kernel)

        #inverse
        my = 255-my
        my = predict.scaleImg(my)
        trmy = Image.fromarray( my )
        #trmy = Image.fromarray(cv2.cvtColor(my,cv2.COLOR_GRAY2RGB))
        
        if i < 5:
            #cv2.imshow("my%s"%i, my)                        
            top1.append(  predict.predict_char( trmy ) )

        if i >=5 and i < 10 :                   
            top2.append(  predict.predict_char( trmy ) )
        
        if i >=10 and i < 15 :                      
            top3.append(  predict.predict_char( trmy ) )

        if i>=15:
            top4.append(  predict.predict_char( trmy ) )

    print("top1",top1[::-1] )
    print("top2",top2[::-1] )
    print("top3",top3[::-1] )
    print("top4",top4[::-1] )

    cv2.imshow("img", img)
    
    k = cv2.waitKey(10) & 0xFF     
    if k == 27:
        break


cv2.destroyAllWindows()
