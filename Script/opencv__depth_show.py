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
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='F:/Unreal Projects/P7/Script/path/to/best_WTB.pt', force_reload = True)  #  local model
#print('Model has been downloaded and created')
'''
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
'''
client = airsim.MultirotorClient()

print("Connected: now while this script is running, you can open another")
print("console and run a script that flies the drone and this script will")
print("show the depth view while the drone is flying.")

help = False

prev_x_min = 0
prev_y_min = 0
prev_x_max = 0
prev_y_max = 0
confidenceData = []

while True:
    #Parse the FPV view and operate on it to get the bounding box + camera view parameters
    responses = client.simGetImages([
        airsim.ImageRequest("high_res", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("high_res", airsim.ImageType.DepthPlanar, True)
        ])
    if (responses == None):
        print("Camera is not returning image, please check airsim for error messages")
        sys.exit(0)
    else:
        # Color Camera
        response = responses[0]
        img_color = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_color = img_color.reshape(response.height, response.width, 3)
        #rawImage, xmin, ymin, xmax, ymax, width, height, detected = self.detectAndMark(rawImage)
        '''    
        self.cam_coords["xmin"] = xmin
        self.cam_coords["ymin"] = ymin
        self.cam_coords["xmax"] = xmax
        self.cam_coords["ymax"] = ymax
        self.cam_coords["height"] = height
        self.cam_coords["width"] = width
        '''
        # Depth Camera
        img_depth = np.asarray(responses[1].image_data_float)
        img_depth = img_depth.reshape(responses[1].height, responses[1].width)
        img_depth[img_depth > 16000] = np.nan #16000 np.nan
        img_depth = cv2.resize(img_depth, (1280,720), interpolation = cv2.INTER_AREA)
        #img_depth = img_depth[int(ymin):int(ymax), int(xmin):int(xmax)]
        print("Min-Max: ", np.min(img_depth), " ", np.max(img_depth))
        depth_range = np.array([np.nanmin(img_depth), np.nanmax(img_depth)])
        depth_map = np.around((img_depth - depth_range[0])*(255-0)/( depth_range[1] - depth_range[0]))


        #GOOD STUFF
        #cv2.imshow("img_color", img_color)
        cv2.imshow("img_depth", depth_map)
        
        key = cv2.waitKey(1) & 0xFF
        if (key == 27 or key == ord('q') or key == ord('x')):
            data = pd.DataFrame({tuple(confidenceData)})
            data.to_excel('confidenceData.xlsx', sheet_name='sheet1', index=False)
            break