# DRL Oprimized Drone Path
# Created by: AL, TAP, SFMS // APEL1-1
# P7 Project
# Aalborg Univeristy Esbjerg AAU / Fall_2022

# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/main/docs/image_apis.md#computer-vision-mode

from re import I
import setup_path
import airsim

# requires Python 3.5.3 :: Anaconda 4.4.0
# pip install opencv-python
import cv2
import os
import time
import sys
import numpy as np
import torch
import threading
import pandas as pd

from PIL import Image


#model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best.pt', source = 'local', force_reload=True)  #  local model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='F:/Unreal Projects/P7/Script/path/to/best_WTB.pt', force_reload = True)  #  local model
print('Model has been downloaded and created')

def detectAndMark(image):
    result = model(image)
    result.print()
    objs = result.pandas().xyxy[0]
    objs_name = objs.loc[objs['name'] == "WTB"]
    height = image.shape[0]
    width = image.shape[1]
    x_middle = 0
    y_middle = 0
    x_min = None
    y_min = None
    x_max = None
    y_max = None
    confidence = 0
    try:
        obj = objs_name.iloc[0]
        x_min = obj.xmin
        y_min = obj.ymin
        x_max = obj.xmax
        y_max = obj.ymax
        confidence = obj.confidence
        x_middle = x_min + (x_max-x_min)/2
        y_middle = y_min + (y_max-y_min)/2
        
        print(objs)
                    
        x_middle = round(x_middle, 0)
        y_middle = round(y_middle, 0)
        # Calculate the distance from the middle of the camera frame view, to the middle of the object
        x_distance = x_middle-width/2
        y_distance = y_middle-height/2

        cv2.rectangle(image, (int(obj.xmin), int(obj.ymin)), (int(obj.xmax), int(obj.ymax)), (0,255,0),2)
        cv2.circle(image, (int(x_middle), int(y_middle)), 5, (0, 255, 0), 2)
        cv2.circle(image, (int(width/2), int(height/2)), 5, (0, 0, 255), 2)
        cv2.line(image, (int(x_middle), int(y_middle)), (int(width/2), int(height/2)), (0,0,255), 2)
        cv2.line(image, (int((width/2)-200), int(0)), (int((width/2)-200), int(height)), (255,0,0), 2)
        cv2.line(image, (int((width/2)+200), int(0)), (int((width/2)+200), int(height)), (255,0,0), 2)
    except:
        print("Error")
        print(objs)
        data = pd.DataFrame({tuple(confidenceData)})
        data.to_excel('confidenceData.xlsx', sheet_name='sheet1', index=False)
    return image , x_min, y_min, x_max, y_max, confidence


cameraTypeMap = {
 "depth": airsim.ImageType.DepthVis,
 "segmentation": airsim.ImageType.Segmentation,
 "seg": airsim.ImageType.Segmentation,
 "scene": airsim.ImageType.Scene,
 "disparity": airsim.ImageType.DisparityNormalized,
 "normals": airsim.ImageType.SurfaceNormals
}


client = airsim.MultirotorClient()

print("Connected: now while this script is running, you can open another")
print("console and run a script that flies the drone and this script will")
print("show the depth view while the drone is flying.")

help = False

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
thickness = 2
textSize, baseline = cv2.getTextSize("FPS", fontFace, fontScale, thickness)
print(textSize)
textOrg = (10, 10 + textSize[1])
frameCount = 0
startTime = time.time()
fps = 0

prev_x_min = 0
prev_y_min = 0
prev_x_max = 0
prev_y_max = 0
confidenceData = []

while True:
    # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
    #rawImage = client.simGetImage("0", cameraTypeMap[cameraType])
    rawImages = client.simGetImages([airsim.ImageRequest("high_res", airsim.ImageType.Scene, False, False)])
    #rawImageDepth = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False)])
    
    #rawImageDepth = client.simGetImage("high_res", airsim.ImageType.DepthVis)
    #rawImageDepth = client.simGetImage("high_res", airsim.ImageType.DepthPerspective, True)
    #responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)])
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)])

    if (rawImages == None or responses == None):
        print("Camera is not returning image, please check airsim for error messages")
        sys.exit(0)
    else:
        # High resolution, color image
        response = rawImages[0]
        rawImage = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        rawImage = rawImage.reshape(response.height, response.width, 3)
        rawImage, x_min, y_min, x_max, y_max, confidence = detectAndMark(rawImage)

        confidenceData.append(confidence)
        data = pd.DataFrame({tuple(confidenceData)})
        data.to_excel('confidenceData.xlsx', sheet_name='sheet1', index=False)
        
        if x_min == None or y_min == None or x_max == None or y_max == None:
            x_min = prev_x_min
            y_min = prev_y_min
            x_max = prev_x_max
            y_max = prev_y_max
        else:
            prev_x_min = x_min
            prev_y_min = y_min
            prev_x_max = x_max
            prev_y_max = y_max
        
            
        # Depth camera

        img_depth = np.asarray(responses[0].image_data_float)
        img_depth = img_depth.reshape(responses[0].height, responses[0].width)
        #print("Depth max:", np.nanmax(img_depth))
        img_depth[img_depth > 16000] = np.nan

        #print("test shape original: ", img_depth.shape)
        img_depth = cv2.resize(img_depth, (1920,1080), interpolation = cv2.INTER_AREA)
        #print("test shape interpolated: ", img_depth.shape)

        img_depth = img_depth[int(y_min):int(y_max), int(x_min):int(x_max)]
        #print("test shape cut: ", img_depth.shape)


        x_small_val=16000
        y_small_val=16000
        x_small = 0
        y_small = 0
        #print("Dimensions: ", img_depth.shape[0], " ", img_depth.shape[1])
        '''
        for x in range (0,int(img_depth.shape[1])):
            for y in range (0,int(img_depth.shape[0])):
                if img_depth[y,x]<x_small_val:
                    x_small_val=img_depth[y,x]
                    x_small=x
                if img_depth[y,x]<y_small_val:
                    y_small_val=img_depth[y,x]
                    y_small=y
        '''
        
        #print("X small = ", x_small)
        #print("Y small = ", y_small)
        depthDistance = int(np.nanmin(img_depth))
        print("Distance = ", depthDistance)
        #dist_100m = 321.5*100/70
        #img_depth[img_depth > dist_100m] = dist_100m

        depth_range = np.array([np.nanmin(img_depth), np.nanmax(img_depth)])
        #print("Depth max:", depth_range[1], " Depth min:", depth_range[0])
        #depth_map = np.around((img_depth - 0)*(255-0)/( dist_100m - 0))

        depth_map = np.around((img_depth - depth_range[0])*(255-0)/( depth_range[1] - depth_range[0]))
        #depth_map = np.around(255*(img_depth - depth_range[0])/(depth_range[1] - depth_range[0])) # Scaling depth map from [0,255]
        #print(depth_map)
        #print("Middle point: ", img_depth[int(img_depth.shape[0]/2),int(img_depth.shape[1]/2)])


        '''
        img1d = np.array(rawImageDepth[0].image_data_float, dtype=np.float64)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        print("Brightest pixel for test: ", img1d)
        if((rawImageDepth[0].width!=0 or rawImageDepth[0].height!=0)):
            img2d = np.reshape(img1d, (rawImageDepth[0].height, rawImageDepth[0].width))
        else:
            print("Error - depth not recorded")

        image = np.invert(np.array(Image.fromarray(img2d.astype(np.uint8), mode='L')))
        # Decrease intensity such that dark pixels become much darker, bright pixels become slightly dark
        factor = 10
        maxIntensity = 255.0  # depends on dtype of image data
        #maxIntensity = 65535.0
        newImage1 = (maxIntensity) * (image / maxIntensity) ** factor
        newImage1 = np.array(newImage1, dtype=np.uint8)
        rawImageDepth = newImage1
        '''
        
        '''
        # Decrease intensity such that dark pixels become much darker, bright pixels become slightly dark
        factor = 10
        #maxIntensity = 255.0  # depends on dtype of image data
        maxIntensity = 65535.0
        newImage1 = (maxIntensity) * (image / maxIntensity) ** factor
        newImage1 = np.array(newImage1, dtype=np.uint16)

        imageDepth = newImage1
        small = cv2.resize(newImage1, (0, 0), fx=1.0, fy=1.0)

        cut = small[20:40, :]
        # print(cut.shape)
        
        info_section = np.zeros((10, small.shape[1]), dtype=np.uint16) + 255
        info_section[9, :] = 0

        #imageDepth = np.concatenate((info_section, small), axis=0)
        '''
        '''
        print("test shape1: ", rawImageDepth.shape)
        rawImageDepth = cv2.resize(rawImageDepth, (1920,1080), interpolation = cv2.INTER_AREA)
        print("test shape2: ", rawImageDepth.shape)
        testvar = rawImageDepth[500:500]
        print("test value: ", rawImageDepth[500:500])
        rawImageDepth = rawImageDepth[int(y_min):int(y_max), int(x_min):int(x_max)]
        x_small_val=255
        y_small_val=255
        x_small = 0
        y_small = 0
        print("Dimensions: ", rawImageDepth.shape[0], " ", rawImageDepth.shape[1])
        for x in range (0,int(rawImageDepth.shape[1])):
            for y in range (0,int(rawImageDepth.shape[0])):
                if rawImageDepth[y,x]<x_small_val:
                    x_small_val=rawImageDepth[y,x,0]
                    x_small=x
                if rawImageDepth[y,x]<y_small_val:
                    y_small_val=rawImageDepth[y,x,0]
                    y_small=y

        print("X small = ", x_small)
        print("Y small = ", y_small)

        distance = (rawImageDepth[y_small, x_small]/255)*100
        print("Darkest pixel value: ", rawImageDepth[y_small, x_small])
        print("Distance to wind turbine: ", distance)
        '''
        
        '''
        rawImageDepth = cv2.imdecode(airsim.string_to_uint8_array(rawImageDepth), cv2.IMREAD_UNCHANGED)
        
        rawImageDepth = cv2.resize(rawImageDepth, (1920,1080), interpolation = cv2.INTER_AREA)
        
        rawImageDepth = rawImageDepth[int(y_min):int(y_max), int(x_min):int(x_max)]
        x_small_val=255
        y_small_val=255
        x_small = 0
        y_small = 0
        print("Dimensions: ", rawImageDepth.shape[0], " ", rawImageDepth.shape[1])
        for x in range (0,int(rawImageDepth.shape[1])):
            for y in range (0,int(rawImageDepth.shape[0])):
                if rawImageDepth[y,x,0]<x_small_val:
                    x_small_val=rawImageDepth[y,x,0]
                    x_small=x
                if rawImageDepth[y,x,0]<y_small_val:
                    y_small_val=rawImageDepth[y,x,0]
                    y_small=y

        print("X small = ", x_small)
        print("Y small = ", y_small)

        distance = (rawImageDepth[y_small, x_small,0]/255)*100
        print("Darkest pixel value: ", rawImageDepth[y_small, x_small])
        print("Distance to wind turbine: ", distance)
        cv2.circle(rawImageDepth, (int(x_small), int(y_small)), 2, (0, 255, 0), 2)
        '''

        #GOOD STUFF
        #cv2.imshow("FPV", rawImage)
        #cv2.imshow("depth", depth_map)
        
        
        #cv2.imshow("depth", newImage1)

        
        #png = cv2.imdecode(airsim.string_to_uint8_array(rawImageDepth), cv2.IMREAD_UNCHANGED)
        #cv2.putText(png,'FPS ' + str(fps),textOrg, fontFace, fontScale,(255,0,255),thickness)
        #cv2.imshow("Depth", png)
        

    frameCount = frameCount  + 1
    endTime = time.time()
    diff = endTime - startTime
    if (diff > 1):
        fps = frameCount
        frameCount = 0
        startTime = endTime

    key = cv2.waitKey(1) & 0xFF
    if (key == 27 or key == ord('q') or key == ord('x')):
        data = pd.DataFrame({tuple(confidenceData)})
        data.to_excel('confidenceData.xlsx', sheet_name='sheet1', index=False)
        break