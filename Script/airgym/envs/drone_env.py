from xml.etree.ElementTree import tostring
import setup_path
import airsim
import numpy as np
import math
from time import perf_counter
from argparse import Action, ArgumentParser
import torch
import cv2
import os
import sys
import gym
import csv
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv

model = torch.hub.load('ultralytics/yolov5', 'custom', path='F:/Unreal Projects/P7/Script/path/to/best_WTB.pt', force_reload = True)  #  local model
print('Model has been downloaded and created')

global curr_time, prev_time, detected, episode_length
curr_time = 0
prev_time = perf_counter()
detected = True
episode_length = 0

class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
            "orientation": np.zeros(3)
            #"cam_coords": np.zeros(4),
        }

        self.cam_coords = {
            "xmin" : 0,
            "ymin" : 0,
            "xmax" : 0,
            "ymax" : 0,
            "height" : 0,
            "width" : 0,
        }
        self.edge_coords = {
            "edge_x1" : 0,
            "edge_y1" : 0,
            "edge_x2" : 0,
            "edge_y2" : 0,
        }

        self.depthDistance = 30.0
        self.prev_depthDistance = 30.0

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7) # Number of possible actions/movements in the action_space

        self._setup_flight()
        ''' DEBUGGING
        init_camera_info = self.drone.simGetCameraInfo("high_res")
        print(type(init_camera_info))

        self.drone_state = self.drone.getMultirotorState()
        self.state["orientation"] = self.drone_state.kinematics_estimated.orientation
        print(self.state["orientation"])
        quaterion = self.state["orientation"]
        z = quaterion.z_val
        print(z)
        #camera_pose = airsim.Pose(airsim.Vector3r(self.state(position)), airsim.to_quaternion(0, 0, )
        '''
        self.prev_x_size = 0
        self.prev_y_size = 0
        self.x_size = 0
        self.y_size = 0

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        #self.drone.moveToPositionAsync(-0.55265, -31.9786, -19.0225, 10).join()
        #self.drone.moveToPositionAsync(256, -4, -60, 10).join()
        #self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()
        self.drone.moveByVelocityAsync(0, 0, -0.8, 5).join() # move 4 meters up

    def detectAndMark(self, image):
        result = model(image)
        is_detected = True
        objs = result.pandas().xyxy[0]
        objs_name = objs.loc[objs['name'] == "WTB"]
        height = image.shape[0]
        width = image.shape[1]
        x_middle = 0
        y_middle = 0
        x_min = 0
        y_min = 0
        x_max = 0
        y_max = 0
        try:
            obj = objs_name.iloc[0]
            
            x_min = obj.xmin
            y_min = obj.ymin
            x_max = obj.xmax
            y_max = obj.ymax
            x_middle = x_min + (x_max-x_min)/2
            y_middle = y_min + (y_max-y_min)/2
                    
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
            is_detected = False
        return image, x_min, y_min, x_max, y_max, width, height, is_detected

    def edge_detection(self,depth_image):
        depth_image = cv2.convertScaleAbs(depth_image, alpha=255.0/depth_image.max())
        # Convert to grayscale
        #gray = cv2.cvtColor(depth_image,cv2.COLOR_BGR2GRAY)
        # Apply Gausian blur
        blur = cv2.GaussianBlur(depth_image, (5,5),0)
        # Use canny edge detection
        edges = cv2.Canny(blur,50,150)
        # Apply HoughLinesP method to to directly obtain line end points
        lines_list =[]
        lines = cv2.HoughLinesP(
			        edges, # Input edge image
			        1, # Distance resolution in pixels
			        np.pi/180, # Angle resolution in radians
			        threshold=70, # Min number of votes for valid line
			        minLineLength=30, # Min allowed length of line
			        maxLineGap=40 # Max allowed gap between line for joining them
			        )
        # Eliminate non-vertical lines
        non_ver_lines =[]
        for points in lines:
            x1,y1,x2,y2=points[0]
            if abs(x1-x2)>10:
                non_ver_lines.append([x1,y1,x2,y2])
        # Get the longest line
        longest_lines = []

        for points in non_ver_lines:
            x1,y1,x2,y2=points
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            # if the length of line is greater than the length of current longest lines, add it to the longest lines list
            if len(longest_lines) < 2:
                longest_lines.append((points,length))
            else:
                for i, l in enumerate(longest_lines):
                    if length > l[1]:
                        longest_lines[i] = (points,length)
                        break
        # Iterate over points
        for points in longest_lines:
            (x1,y1,x2,y2), _=points
            cv2.line(depth_image,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.circle(depth_image,(x1,y1),7,(255,0,0),2)
            cv2.circle(depth_image,(x2,y2),7,(255,0,0),2)
            cv2.circle(depth_image,(round((x1+x2)/2),round((y1+y2)/2)),7,(255,0,0),2)

        (x1,y1,x2,y2), _ = longest_lines[0]

        return depth_image, x1, y1, x2, y2
    
    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _get_obs(self): # Observation space
        global detected

        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        
        #Locking the camera orientation to pose of the drone and the orientation to (0,0,drone_yaw)
        self.state["orientation"] = self.drone_state.kinematics_estimated.orientation
        drone_orientation = self.state["orientation"]
        yaw_drone_frame = drone_orientation.z_val
        x_drone_pos = self.state["position"].x_val
        y_drone_pos = self.state["position"].y_val
        z_drone_pos = self.state["position"].z_val
        camera_pose = airsim.Pose(airsim.Vector3r(x_drone_pos, y_drone_pos, z_drone_pos), airsim.to_quaternion(0, 0, yaw_drone_frame))
        self.drone.simSetCameraPose("high_res", camera_pose)
        #self.drone.simSetCameraPose("0", camera_pose)
        
        self._log_position_state(x_drone_pos, y_drone_pos, z_drone_pos)

        #Parse the FPV view and operate on it to get the bounding box + camera view parameters
        responses = self.drone.simGetImages([
            airsim.ImageRequest("high_res", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("high_res", airsim.ImageType.DepthPlanar, True)
            ])
        response = responses[0]
        rawImage = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        rawImage = rawImage.reshape(response.height, response.width, 3)
        rawImage, xmin, ymin, xmax, ymax, width, height, detected = self.detectAndMark(rawImage)
        
        self.cam_coords["xmin"] = xmin
        self.cam_coords["ymin"] = ymin
        self.cam_coords["xmax"] = xmax
        self.cam_coords["ymax"] = ymax
        self.cam_coords["height"] = height
        self.cam_coords["width"] = width

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        #Depth Camera
        img_depth = np.asarray(responses[1].image_data_float)
        img_depth = img_depth.reshape(responses[1].height, responses[1].width)
        img_depth[img_depth > 16000] = np.nan
        img_depth = cv2.resize(img_depth, (1920,1080), interpolation = cv2.INTER_AREA)
        img_depth_crop = img_depth[int(ymin):int(ymax), int(xmin):int(xmax)]
        
        try:
            self.depthDistance = (70/328)*int(np.nanmin(img_depth_crop))
        except:
            self.depthDistance = self.prev_depthDistance


        depth_range = np.array([np.nanmin(img_depth), np.nanmax(img_depth)])
        depth_map = np.around((img_depth - depth_range[0])*(255-0)/(depth_range[1] - depth_range[0]))
        
        try:
            _ , edge_x1, edge_y1, edge_x2, edge_y2 = self.edge_detection(depth_map)
        except:
            print("No lines detected")
            edge_x1 = 0
            edge_y1 = 0
            edge_x2 = 0
            edge_y2 = 0

         
        self.cam_coords["edge_x1"] = edge_x1
        self.cam_coords["edge_y1"] = edge_y1
        self.cam_coords["edge_x2"] = edge_x2
        self.cam_coords["edge_y2"] = edge_y2
                    
        fake_return = np.zeros((84, 84, 1))
        
        return fake_return

    def _do_action(self, action):
        quad_offset, rotate = self.interpret_action(action)
        #quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityBodyFrameAsync(
            quad_offset[0],
            quad_offset[1],
            quad_offset[2],
            5,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(True, rotate)
        ).join()    

    #Gradient (linear) reward for the bounding box location
    def reward_center(self, center, width, limit):
        if center >= 0 and center < (width/2-limit):
            reward = ((1/(width/2-limit)) * center) - 1
        elif center >= (width/2-limit) and center <= (width/2+limit):
            reward = -(1/limit)*abs(center-(width/2)) + 1 
        elif center > (width/2+limit) and center <= width:
            reward = -(1/(width/2-limit))*(center-(width/2+limit)) 
        else:
            reward = -1
        return reward

    def line_maximization(self, x1, y1, x2, y2, frame_width, frame_height):
        line_length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5 # Calculate the length of the line
        if x1 == x2:
            max_length = frame_height # if vertical, max length is height
        elif y1 == y2:
            max_length = frame_width # if horizontal, max length is width
        else:
            slope = (y2 - y1) / (x2 - x1) 
            angle = math.atan(slope)
            max_length = (frame_width * math.cos(angle)) + (frame_height * math.sin(angle)) # this, if at angle
        
        maximization_value = line_length / max_length

        return maximization_value
        

    def _compute_reward(self):
        global curr_time, prev_time, detected, episode_length

        curr_time = perf_counter()
        reward = 0
        reward1 = 0
        reward2 = 0
        done = 0
        
        self.x_obj_middle = self.cam_coords["xmin"] + (self.cam_coords["xmax"]-self.cam_coords["xmin"])/2
        self.y_obj_middle = self.cam_coords["ymin"] + (self.cam_coords["ymax"]-self.cam_coords["ymin"])/2
        self.x_cam_middle = self.cam_coords["width"] / 2
        self.y_cam_middle = self.cam_coords["height"] / 2

        self.x_size = self.cam_coords["xmax"] - self.cam_coords["xmin"]
        self.y_size = self.cam_coords["ymax"] - self.cam_coords["ymin"]

        self.x_edge_middle = self.edge_coords["edge_x1"] + (self.edge_coords["edge_x2"]-self.edge_coords["edge_x1"])/2
        self.y_edge_middle = self.edge_coords["edge_y1"] + (self.edge_coords["edge_y2"]-self.edge_coords["edge_y1"])/2

        if self.state["collision"]:
            reward = -100
            done = 1
            episode_length = 0
        else:
            if not detected:
                done = 1
                episode_length = 0
                print("Agent update - detection lost, exiting")
            
            print("Distance = ", self.depthDistance)

            # REWARD 1
            if self.depthDistance < self.prev_depthDistance:
                print("Agent update - getting closer")
                reward1 += 1
                self.prev_depthDistance = self.depthDistance
            else:
                print("Agent update - getting further")
                reward1 -= 1
                self.prev_depthDistance = self.depthDistance

            reward_obj_center = self.reward_center(self.x_obj_middle, self.cam_coords["width"], 400) + self.reward_center(self.y_obj_middle, self.cam_coords["height"], 400)
            reward1 += reward_obj_center
            
            # REWARD 2
            reward_edge_center = self.reward_center(self.x_edge_middle, self.cam_coords["width"], 400) + self.reward_center(self.y_edge_middle, self.cam_coords["height"], 400)
            reward2 += reward_edge_center
            reward2 += self.line_maximization(self.edge_coords["edge_x1"], # how big is the line wrt to the camera view
                                              self.edge_coords["edge_y1"], # outputs 0-1 range
                                              self.edge_coords["edge_x2"],
                                              self.edge_coords["edge_y2"],
                                              self.cam_coords["width"],
                                              self.cam_coords["height"])

            # Weighted sum of the two rewards (dynamic reward function)
            W1 = 0.5 * (1 + np.tanh(2 * (self.depthDistance - 30))) # Smooth transition around the 30m mark
            W2 = 1 - W1
            reward = reward + W1*reward1 + W2*reward2

            if episode_length >= 50 or self.depthDistance < 10.0:
                print("Agent stopped - max time_step in episode exceeded or distance < 30m")
                done = 1
                episode_length = 0

        return reward, done

    def step(self, action):
        global episode_length
        self._do_action(action)
        obs = self._get_obs()
        episode_length += 1
        print("Episode - timestep: " , episode_length)
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        rotate = 0
        quad_offset = (0, 0, 0)

        if action == 0:     # FRONT
            quad_offset = (self.step_length, 0, 0)
            rotate = 0
        elif action == 1:   # BACK
            quad_offset = (-self.step_length, 0, 0)
            rotate = 0
        elif action == 2:   # RIGHT
            quad_offset = (0, 0, 0)
            rotate = 2
        elif action == 3:   # LEFT
            quad_offset = (0, 0, 0)
            rotate = -2
        elif action == 4:   # UP
            quad_offset = (0, 0, self.step_length)
            rotate = 0
        elif action == 5:   # DOWN
            quad_offset = (0, 0, -self.step_length)
            rotate = 0
        else:               # STOP
            quad_offset = (0, 0, 0)
            rotate = 0        

        return quad_offset, rotate


    def _log_position_state(self, position_x: int, position_y: int, position_z: int):
        """Save position of the drone into a CSV file

        Args:
            position_x (int): Position in X in world coordinates
            position_y (int): Position in Y in world coordinates
            position_z (int): Position in Z in world coordinates
        """
        with open("drone_position.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([position_x, position_y, position_z])