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

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(4)
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

        

        #self.image_request = airsim.ImageRequest(
        #    3, airsim.ImageType.DepthPerspective, True, False
        #)

        


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
        self.drone.moveByVelocityAsync(0, 0, -0.8, 5).join()

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
    
    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
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

        #Parse the FPV view and operate on it to get the bounding box + camera view parameters
        rawImage = self.drone.simGetImages([airsim.ImageRequest("high_res", airsim.ImageType.Scene, False, False)])
        response = rawImage[0]
        rawImage = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        rawImage = rawImage.reshape(response.height, response.width, 3)
        image, xmin, ymin, xmax, ymax, width, height, detected = self.detectAndMark(rawImage)
        #cv2.imshow("FPV", rawImage)
        #self.state["cam_coords"] = np.array ([xmin,ymin,xmax,ymax])



        self.cam_coords["xmin"] = xmin
        self.cam_coords["ymin"] = ymin
        self.cam_coords["xmax"] = xmax
        self.cam_coords["ymax"] = ymax
        self.cam_coords["height"] = height
        self.cam_coords["width"] = width

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        fake_return = np.zeros((84, 84, 1))

        #return image
        #return image
        return fake_return

    def _do_action(self, action):
        quad_offset, rotate = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityBodyFrameAsync(
            quad_offset[0],
            quad_offset[1],
            quad_offset[2],
            5,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(True, rotate)
        ).join()
        '''
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            5,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(True, rotate)
        ).join()
        
        self.drone.moveByVelocityBodyFrameAsync(
            0,
            0,
            rotate*self.step_length,
            5,
        ).join()
        '''

    def _compute_reward(self):
        global curr_time, prev_time, detected, episode_length

        curr_time = perf_counter()
        reward = 0
        done = 0
        
        self.x_obj_middle = self.cam_coords["xmin"] + (self.cam_coords["xmax"]-self.cam_coords["xmin"])/2
        self.y_obj_middle = self.cam_coords["ymin"] + (self.cam_coords["ymax"]-self.cam_coords["ymin"])/2
        self.x_cam_middle = self.cam_coords["width"] / 2
        self.y_cam_middle = self.cam_coords["height"] / 2
        
        self.x_size = self.cam_coords["xmax"] - self.cam_coords["xmin"]
        self.y_size = self.cam_coords["ymax"] - self.cam_coords["ymin"]

        if self.state["collision"]:
            reward = -100
            done = 1
            episode_length = 0
        else:
            if not detected:
                done = 1
                episode_length = 0
                print("Agent update - detection lost, exiting")
            
            if self.x_size - self.prev_x_size > 0 or self.y_size - self.prev_y_size > 0:
                reward += 1
                print("Agent update - getting closer")
                self.prev_x_size = self.x_size
                self.prev_y_size = self.y_size
            else:
                reward -= 1
                print("Agent update - getting further")
                self.prev_x_size = self.x_size
                self.prev_y_size = self.y_size
            
            if self.x_obj_middle > (self.x_cam_middle + 200) or self.x_obj_middle < (self.x_cam_middle - 200):
                reward -= 1
                print("Agent update - not centered")
            else:
                reward += 1
                print("Agent update - centered")

            if episode_length >= 30:
                print("Agent stopped - max time_step in episode exceeded")
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

        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, 0, 0)
            rotate = 2
        elif action == 2:
            quad_offset = (0, 0, 0)
            rotate = -2
        else:
            quad_offset = (0, 0, 0)
            rotate = 0

        return quad_offset, rotate


