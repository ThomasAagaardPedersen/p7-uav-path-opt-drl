import cv2
import numpy as np
from matplotlib import pyplot as plt

def reward_center(center, width, limit):
        if center >= 0 and center < (width/2-limit):
            reward = ((1/(width/2-limit)) * center) - 1
        elif center >= (width/2-limit) and center <= (width/2+limit):
            reward = -(1/limit)*abs(center-(width/2)) + 1 
        elif center > (width/2+limit) and center <= width:
            reward = -(1/(width/2-limit))*(center-(width/2+limit)) 
        else:
            reward = -1
        return reward

width = 768
height = 432
limit = 160
output = np.zeros((height,width,3), dtype=np.uint8)

print("Array Dimension = ",len(output.shape))

for x in range(width):
    for y in range(height):
        #reward = int((reward_center(x, width, limit)+2)*255/4)
        reward = int((reward_center(x, width, limit) + reward_center(y, height, limit) + 2)*255/4)
        if reward >= 128:
            output[y,x] =  [255, 0, -2*reward+255*2] #R,G,B
        else:
            output[y,x] =  [reward*2, 0, 255] #R,G,B
        #print("Current value at X: ", x, " and Y: ", y, " is ", output[x,y])
        print("Progress: ", round(x/width*100, 2))

print(output)
output = output.reshape(height,width,3)
#cv2.rectangle(image, (int(obj.xmin), int(obj.ymin)), (int(obj.xmax), int(obj.ymax)), (0,255,0),2)
cv2.rectangle(output,(int(width/2+limit), int(height/2+limit)),(int(width/2-limit),int(height/2-limit)),(0,255,0),2)
cv2.imshow("Gradient", output)

plt.imshow(output, interpolation='nearest')
plt.show()



