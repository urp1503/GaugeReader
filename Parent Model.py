import matplotlib.pylab as plt
from cv2 import cv2
import numpy as np
import os
import pandas as pd
os.chdir(r'C:\GEOLOGIX\Analogue-Gauge-Reader-master\Analogue-Gauge-Reader-master')
retval = os.getcwd()
print ("Current working directory %s" % retval)

def avg_circles(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    for i in range(b):
        #optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r

def dist_2_pts(x1, y1, x2, y2):
    #print np.sqrt((x2-x1)^2+(y2-y1)^2)
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

img= cv2.imread(r"C:\Users\utkar\OneDrive\Desktop\py\bestfitgauge.png")
output= img.copy()
output1= img.copy()
output2= img.copy()
output3= img.copy()
output4= img.copy()
output5= img.copy()
output6= img.copy()


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
height, width= img.shape[:2]
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convert to gray

#canny= cv2.Canny(gray, 100, 200)

#cv2.imshow('output', gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

min_value = 0  #usually zero
max_value = 300
#input('Max value: ') #maximum reading of the gauge


#Using Hough cirlces to find the cirlces in the image
#detect circles
#restricting the search from 35-48% of the possible radii gives fairly good results across different samples.  Remember that
#these are pixel values which correspond to the possible radii search range.

circle_img= cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1, 20, np.array([]), 100, 50, int(height*0.35), int(height*0.50))
a, b, c = circle_img.shape
circle_img

a,b,c = circle_img.shape
a,b,c

for (x,y,r) in circle_img[0,:]:
    cv2.circle(output2, (x,y), r, (0,255,0), 3)
    cv2.circle(output2, (x,y), 2, (0,255,0), 3)
    print(x,y,r)


cv2.imshow('output0', output2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Averaging out nearby circles incase
x,y,r = avg_circles(circle_img, b)
cv2.circle(output3, (x,y), r, (0,255,0), 3)
cv2.circle(output3, (x,y), 2, (0,255,0), 3)

cv2.imshow('output', output3)
cv2.waitKey(0)
cv2.destroyAllWindows()

x,y,r

separation = 10  # in degrees
interval = int(360 / separation)
p1 = np.zeros((interval, 2))  # set empty arrays
p2 = np.zeros((interval, 2))
p_text = np.zeros((interval, 2))

for i in range(0, interval):
    for j in range(0, 2):
        if (j % 2 == 0):
            p1[i][j] = x + 0.9 * r * np.cos(separation * i * np.pi / 180)  # point for lines
        else:
            p1[i][j] = y + 0.9 * r * np.sin(separation * i * np.pi / 180)

text_offset_x = 10
text_offset_y = 5

for i in range(0, interval):
    for j in range(0, 2):
        if (j % 2 == 0):
            p2[i][j] = x + r * np.cos(separation * i * np.pi / 180)
            p_text[i][j] = x - text_offset_x + 1.2 * r * np.cos(
                (separation) * (i + 9) * np.pi / 180)  # point for text labels, i+9 rotates the labels by 90 degrees
        else:
            p2[i][j] = y + r * np.sin(separation * i * np.pi / 180)
            p_text[i][j] = y + text_offset_y + 1.2 * r * np.sin(
                (separation) * (i + 9) * np.pi / 180)  # point for text labels, i+9 rotates the labels by 90 degrees

# add the lines and labels to the image
for i in range(0, interval):
    cv2.line(output3, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])), (0, 255, 0), 2)
    cv2.putText(output3, '%s' % (int(i * separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX,
                0.3, (0, 0, 0), 1, cv2.LINE_AA)

cv2.imshow('output', output3)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('gauge-%s-calibration.%s' % (gauge_number, file_type), img)

separation= 10 #in degrees
interval = int(360/separation)
p3 = np.zeros((interval,2))  #set empty arrays
p4 = np.zeros((interval,2))

for i in range(0,interval):
    for j in range(0,2):
        if (j%2==0):
            p3[i][j] = x + 0.8 * r * np.cos(separation * i * np.pi / 180) #point for lines
        else:
            p3[i][j] = y + 0.8 * r * np.sin(separation * i * np.pi / 180)


region_of_interest_vertices= p3

def region_of_interest(img, vertices):
    mask= np.zeros_like(img)
    match_mask_color= 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image= cv2.bitwise_and(img, mask)
    return masked_image

canny= cv2.Canny(gray, 200, 20)
region_of_interest_vertices= p3
cropped_image= region_of_interest(canny, np.array([region_of_interest_vertices], np.int32))
cv2.imshow('cropped', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, _= cv2.findContours(cropped_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

int_cnt= []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area<15:
        cv2.drawContours(output3, cnt, -1, (255,0,0), 3)
        int_cnt.append(cnt)
cv2.imshow('output', output3)
cv2.waitKey(0)
cv2.destroyAllWindows()

frth_quad_index=[]
thrd_quad_index=[]
reference_zero_angle= 35
reference_end_angle= 330
min_angle=90
max_angle=270

for i in range(len(int_cnt)):
    a = int_cnt[i]
    a = a.reshape(len(a), 2)
    a = pd.DataFrame(a)
    x1 = a.iloc[:, 0].mean()
    y1 = a.iloc[:, 1].mean()

    xlen = x1 - x
    ylen = y - y1

    # Taking arc-tan of ylen/xlen to find the angle
    # res= np.arctan(np.divide(float(ylen), float(xlen)))
    # res= np.rad2deg(res)

    if xlen < 0 and ylen < 0:
        res = np.arctan(np.divide(float(abs(ylen)), float(abs(xlen))))
        res = np.rad2deg(res)
        final_start_angle = 90 - res
        # print(i , final_angle)
        frth_quad_index.append(i)
        if final_start_angle > reference_zero_angle:
            if final_start_angle < min_angle:
                min_angle = final_start_angle

    elif xlen > 0 and ylen < 0:
        res = np.arctan(np.divide(float(abs(ylen)), float(abs(xlen))))
        res = np.rad2deg(res)
        final_end_angle = 270 + res
        thrd_quad_index.append(i)
        # print(i , res)
        if final_end_angle < reference_end_angle:
            if final_end_angle > max_angle:
                max_angle = final_end_angle

print(f'Zero reading corresponds to {min_angle}')
print(f'End reading corresponds to {max_angle}')
