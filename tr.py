import cv2
import numpy as np

img1=np.zeros((128,128,3),dtype=np.uint8)
img2=cv2.imread("3.png")

rows,cols,channel = img2.shape 

#roi=img1[0:rows, 0: cols]
xx=64-int(rows/2)
yy=64-int(cols/2)
roi = img1[  xx:rows+xx, yy:cols+yy ]

img2gray = cv2.cvtColor( img2, cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(img2gray , 175, 255, cv2.THRESH_BINARY)
mask_inv= cv2.bitwise_not(mask)

# cv2.imshow("mask",mask)
# cv2.imshow("mask_inv",mask_inv)

img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

dst = cv2.add(img1_bg,img2_fg)
img1[ xx:rows+xx, yy:cols+yy ] = dst

cv2.imshow("img1_bg",img1_bg)
cv2.imshow("img2_fg",img2_fg)
cv2.imshow("img1",img1)
cv2.imwrite("33.png",img1)


# my = cv2.bitwise_or( roi , roi, mask=mask )
# my2 = cv2.bitwise_or( img2, img2, mask=mask_inv )
# dst= cv2.add(my,my2)

# img[0:rows, 0: cols] = dst

# cv2.imshow("my",my)
# cv2.imshow("my2",my2)
# cv2.imshow("img",img)


cv2.waitKey()
cv2.destroyAllWindows()

