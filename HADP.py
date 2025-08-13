#!/usr/bin/env python3

import cv2
import numpy as np
import os
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import LaserScan
from rclpy.node import Node
from nav_msgs.msg import Path
from sensor_msgs.msg import Image  as im
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

import time
import tf_transformations
import pandas as pd

import math
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
import copy

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from typing import Tuple, Sequence, Dict, Union, Optional, Callable

import math
import torch
import torch.nn as nn
import torchvision
from diffusers import DDIMScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
# from torch2trt import TRTModule


seed = 20.0
cmd_vel_topic_name = '/cmd_vel'
test_approach = 'crowd' #just_forward umap crowd ped10

save_dir = f'{test_approach}'
save_log_dir = f'/home/kmg/nav2_swerve_for_data_collecting/src/swerve_drive_semantic_navigation/navigation_log/{save_dir}/'
navigation_start_status = False
robot_current_pose = None
navigation_result=None
start_time = None
cmd_vel_logs = []
cost_map_image = None
map_image = None
origin_x = None
origin_y = None
width = None
height = None
resolution = None


class TestDynamicObsAvoidance(Node):

    def __init__(self):
        super().__init__("test_dynamic_obs_avoidance")
        
        nav2_cb_group = MutuallyExclusiveCallbackGroup()
        dynamic_obs_detect = MutuallyExclusiveCallbackGroup()

        topic_cb_group = MutuallyExclusiveCallbackGroup()

        publish_cb_group = MutuallyExclusiveCallbackGroup()


        # Subscribers
        self.detected_obs_publisher = self.create_subscription(
            MarkerArray,
            '/detected_objects_in_map',
            self.get_dynamic_obs_pose,
            2,
            callback_group = dynamic_obs_detect
        )
        self.goal_subscription = self.create_subscription(
            PoseStamped,
            '/goal',
            self.goal_callback,
            10,
            callback_group = nav2_cb_group
        )
        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10,
            callback_group = publish_cb_group
        )
        self.laser_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10,
            callback_group=publish_cb_group
        )

        # set publishers
        self.initial_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

        self.goal_publisher = self.create_publisher(PoseStamped, '/goal', 10)

        self.path_marker_publisher = self.create_publisher(MarkerArray, '/diffusion_plan', 10)

        self.local_goal_marker_publisher = self.create_publisher(Marker, '/local_goal', 10)

        self.semantic_map_publisher = self.create_publisher(im, '/semantic_map', 10)
        
        self.crop_center_marker_publisher = self.create_publisher(Marker, '/crop_center_marker', 10)
        self.marker_publisher = self.create_publisher(Marker, '/record_boundary_marker', 10)
        
        self.initial_time = time.time()

        self.initial_pose_timer = self.create_timer(3.0, 
                                                    self.publish_initial_pose)
        self.inference_dummy_path_for_first_step_timer = self.create_timer(10.0, 
                                                    self.inference_dummy_path_for_first_step)
        self.semantic_map_timer = self.create_timer(0.1, 
                                                    self.update_semantic_map,
                                                    callback_group = topic_cb_group)

        self.costmap_publish_timer = self.create_timer(9.0, self.publish_global_costmap)
        
        self.local_goal_marker_publish_timer = self.create_timer(0.2, self.local_goal_marking, callback_group= publish_cb_group)


        self.seed = seed
        self.go_to_set_goal_timer = self.create_timer(self.seed, self.set_goal_and_start_navigation, callback_group=nav2_cb_group)

        
        self.laser_scan = None
        self.detected_collision = False
        self.detected_status = False
        self.width = None
        self.height = None
        self.resolution = None
        self.origin_x = None
        self.origin_y = None
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None

        self.area = 4
        self.d_offset = 1.5
        self.crop_size = None

        self.half_crop_size = None
        
        self.episode_data = [] 
        self.linear_x =  None
        self.linear_y =  None
        self.angular_z = None

        self.pred_horizon = None
        self.obs_horizon = None
        self.action_horizon  = None
        self.test_nets = None
        self.action_dim = None
        self.device = None
        self.num_diffusion_iters = None
        self.noise_scheduler = None
        
        self.initial_robot_pose = None   
        self.fixed_cost_map = None
        self.cropped_map_size = None
        self.scale_factor = None
        self.semantic_maps = []
        self.semantic_map_save_count = 0
        self.undetectable_count_total = 0
        self.undetected_dynamic_obs_during_generative_plan = 0
        self.semantic_record_status = False
        self.record_first_state = True
        self.robot_pose = None
        self.goal_pose = None
        self.final_goal_pose = None
        self.cropped_goal_pose = None

        self.original_size = None

        self.initial_path = None
        
        self.init_yaw = None   
        
        self.map_data = None
        
        self.initial_crop_center_x = None
        self.initial_crop_center_y = None

        self.initial_crop_goal = None
        self.transformed_initial_crop_goal = None

        self.detected_status = False
        self.dynamic_in_area_status = False
        self.detected_dynamic_objects = []

        self.custom_plan_in_progress = False
        self.center_x = None
        self.center_y = None
        self.map_count = 0

        self.distance_to_goal = None
        
        self.navigator = BasicNavigator()
        self.is_dynamic_obs_in_area_status = False
        
        self.costmap = None
        self.costmap_np = None
        self.is_driving = False
        self.global_path_while_one_episode = None
        self.original_local_goal = None
        self.accumulate_semantic_map = None
        self.during_create_custom_path = False
        self.first_inference = True
        self.nav2_start_time = None

        self.initialize_diffuison_model()

        self.semantic_map_input_dim = self.action_horizon 
        

    def initialize_diffuison_model(self):
        #revision crowd
        model_save_path = f"./diffuison_model.pth"

        # parameters
        self.num_diffusion_iters = 40  #Default 50

        self.noise_scheduler = DDIMScheduler( #DDIMScheduler , DDPMScheduler
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

        self.pred_horizon = 20 #20
        
        self.obs_horizon = 5 
        self.action_horizon = self.obs_horizon

        vision_encoder = get_resnet('resnet18')

        vision_encoder = replace_bn_with_gn(vision_encoder)

        # ResNet18 has output dim of 512
        vision_feature_dim = 512
        lowdim_obs_dim = 4
        obs_dim = vision_feature_dim + lowdim_obs_dim
        self.action_dim = 2
        noise_test_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=obs_dim*self.obs_horizon
        )

        # the final arch has 2 parts
        self.test_nets_before_tensorrt = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'noise_pred_net': noise_test_net
        })

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        loaded_state_dict = torch.load(model_save_path, map_location="cuda:0")
        self.test_nets_before_tensorrt.load_state_dict(loaded_state_dict)

        self.test_nets = self.test_nets_before_tensorrt
        # Transfer the model to the appropriate device if needed
        self.test_nets.to(self.device)

        # self.get_logger().info(f"Model successfully loaded and ready for use.{self.test_nets}")

    def inference_dummy_path_for_first_step(self):
        self.inference_dummy_path_for_first_step_timer.cancel()

        #dummy pose
        robot_poses = []
        for _ in range(self.semantic_map_input_dim):
            robot_poses.append((0,0))
        
        #dummy goal_pose
        goal_poses = []
        for _ in range(self.semantic_map_input_dim):
            goal_poses.append((0,0))
        #dummpy semantic map
        semantic_maps = []  
        for _ in range(self.semantic_map_input_dim):
            semantic_maps.append(np.zeros((96, 96, 3), dtype=np.uint8)
        )
        _ = self.inference_path(robot_poses, semantic_maps, goal_poses)
        

    def calculate_path_length(self, path):
        total_length = 0.0

        if path is None or len(path.poses) < 2:
            return total_length
        for i in range(1, len(path.poses)):
            prev_pose = path.poses[i-1].pose.position
            curr_pose = path.poses[i].pose.position

            dx = curr_pose.x - prev_pose.x
            dy = curr_pose.y - prev_pose.y
            dz = curr_pose.z - prev_pose.z
            segment_length = math.sqrt(dx * dx + dy * dy + dz * dz)
            total_length += segment_length

        return total_length
    
    def get_path_callback(self):
        if self.goal_pose:
            current_pose = self.robot_pose
            if not isinstance(self.robot_pose, PoseStamped):
                robot_pose_stamped = PoseStamped()
                robot_pose_stamped.header.frame_id = "map"
                robot_pose_stamped.header.stamp = self.get_clock().now().to_msg()
                robot_pose_stamped.pose = self.robot_pose
                current_pose = robot_pose_stamped
                
            self.initial_path = self.navigator.getPath(current_pose, self.goal_pose, use_start=True)
        
        if self.initial_path is not None:
            path_length = self.calculate_path_length(self.initial_path)
            self.get_logger().info(f"Global path generated. Path length: {path_length:.2f} meters")


    def normalize_data(self, data, max_value):
        ndata  = data / max_value
        ndata = ndata * 2 - 1
        return ndata

    def unnormalize_data(self, ndata, max_value):
        ndata = (ndata + 1) / 2
        data = ndata * max_value
        return data
    
    def inference_path(self, nagent_poses, nimages, ngoal):
        
        B = 1
        
        # infer action
        with torch.no_grad():
            # get image features
            try:
                nimages = torch.tensor(nimages, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
                nagent_poses = torch.tensor(nagent_poses, dtype=torch.float32).to(self.device)
                ngoal = torch.tensor(ngoal, dtype=torch.float32).to(self.device)

                nagent_poses = self.normalize_data(nagent_poses, 96)
                ngoal = self.normalize_data(ngoal, 96)
    
                image_features = self.test_nets['vision_encoder'](nimages)
                
                obs_features = torch.cat([image_features, nagent_poses, ngoal], dim=-1)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                noisy_action = torch.randn(
                    (B, self.pred_horizon, self.action_dim), device=self.device)
                naction = noisy_action
                self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

                for k in self.noise_scheduler.timesteps:
                    noise_pred = self.test_nets['noise_pred_net'](
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )
                    naction = self.noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
                    
                naction = naction.detach().to('cpu').numpy()
                naction = naction[0]
                
                return naction[self.semantic_map_input_dim:]
        
            except RuntimeError as e:
                self.get_logger().error(f"error occured")
                return None

    def calc_goal_distance(self,robot_pose, goal_pose):
        distance_to_goal = np.sqrt((robot_pose.position.x - goal_pose.pose.position.x) ** 2 +
                                    (robot_pose.position.y - goal_pose.pose.position.y) ** 2)
        return distance_to_goal
    def calc_local_goal_distance(self, robot_pose, local_goal_pose):
        distance_to_goal = np.sqrt((robot_pose.position.x - local_goal_pose[0]) ** 2 +
                                    (robot_pose.position.y - local_goal_pose[1]) ** 2)
        return distance_to_goal
        
    def publish_global_costmap(self):
        try:
            self.costmap = self.navigator.getGlobalCostmap()
            
            self.width = self.costmap.metadata.size_x
            self.height = self.costmap.metadata.size_y
            self.resolution = self.costmap.metadata.resolution
            self.origin_x = self.costmap.metadata.origin.position.x
            self.origin_y = self.costmap.metadata.origin.position.y

            global width, height, resolution, origin_x, origin_y
    
            width = self.width
            height = self.height
            resolution = self.resolution
            origin_x = self.origin_x
            origin_y = self.origin_y

            self.crop_size = int(self.area / self.resolution)

            self.half_crop_size = self.crop_size // 2
            self.costmap_np =  self.initialize_map_image(self.costmap)
            self.costmap_np = self.reduce_inflation(self.costmap_np)

            cv2.imwrite("./swerve_drive_semantic_navigation/map_data/cost_map.png", self.costmap_np)
            
            global cost_map_image
            cost_map_image = self.costmap_np

            global map_image
            map_image = self.reduce_inflation(self.costmap_np, 40)
            self.costmap_publish_timer.cancel()
            
        except Exception as e:
            self.get_logger().error(f"Global costmap generation be failed: {e}")
            raise  
            
    def reduce_inflation(self, costmap_np, kernel_dim = 20):
        kernel = np.ones((kernel_dim, kernel_dim), np.uint8)

        inverted_map = cv2.bitwise_not(costmap_np)
        eroded_map = cv2.erode(inverted_map, kernel, iterations=1)

        final_map = cv2.bitwise_not(eroded_map)

        return final_map

    def initialize_map_image(self, costmap):
        height = costmap.metadata.size_y
        width = costmap.metadata.size_x

        costmap_data = np.array(costmap.data).reshape(height, width)
        map_image = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                if costmap_data[y, x] == -1:
                    map_image[y, x] = [128, 128, 128]  
                elif costmap_data[y, x] == 0:
                    map_image[y, x] = [255, 255, 255]  
                else:
                    map_image[y, x] = [0, 0, 0]  
        return map_image

    def publish_initial_pose(self):
        initial_pose_msg = PoseWithCovarianceStamped()
        initial_pose_msg.header.stamp = self.get_clock().now().to_msg()
        initial_pose_msg.header.frame_id = 'map'
        # initial_pose_msg.pose.pose.position.x = 2.25
        initial_pose_msg.pose.pose.position.x = 0.0
        initial_pose_msg.pose.pose.position.y = 0.0
        initial_pose_msg.pose.pose.position.z = 0.0
        initial_pose_msg.pose.pose.orientation.x = 0.0
        initial_pose_msg.pose.pose.orientation.y = 0.0

        
        initial_pose_msg.pose.pose.orientation.z = -0.7071  # 90 degrees clockwise rotation
        initial_pose_msg.pose.pose.orientation.w = 0.7071   # 90 degrees clockwise rotation
        initial_pose_msg.pose.covariance = [0.0] * 36           
            

        self.initial_pose_publisher.publish(initial_pose_msg)
        self.initial_pose_timer.cancel()
    
    def pose_callback(self, msg):
        self.robot_pose = msg.pose.pose
        global robot_current_pose
        robot_current_pose = self.robot_pose
    
    def laser_callback(self,msg):
        self.laser_scan = msg
        min_distance = 0.25
        sensor_min_range = min(msg.ranges)
        
        if  sensor_min_range >= 0.00 and  sensor_min_range <= min_distance and self.is_driving:
            self.detected_collision = True
            self.get_logger().info(f"collision occured")
            self.navigator.cancelTask()
        
    def set_goal_and_start_navigation(self):
        self.nav2_start_time = time.time()
        self.go_to_set_goal_timer.cancel()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        
        global navigation_start_status 
        navigation_start_status = True  
        global start_time
        start_time = self.nav2_start_time
        
        if test_approach == "just_forward":
            goal_pose.pose = Pose(
                position = Point(x=-0.17435789108276367, y=10.21552562713623, z=0.0),
                orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            )
        elif test_approach == "umap":
            goal_pose.pose = Pose(
                position = Point(x=15.624313354492188, y=-0.3219001293182373, z=0.0),
                orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            )
        elif test_approach == "crowd":
            goal_pose.pose = Pose(
                position = Point(x=-20.2978439331054, y=9.513629913330078, z=0.0),
                orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            )
        elif test_approach == "ped10":
            goal_pose.pose = Pose(
                position = Point(x=20.63, y=15.08, z=0.0),
                orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            )
        self.goal_callback(goal_pose)
    
    def goal_callback(self, msg):
        self.get_logger().info(f"Received new goal: {msg}")
        # self.goal_pose = msg
        if not isinstance(msg, PoseStamped):
            goal_pose_stamped = PoseStamped()
            goal_pose_stamped.header.frame_id = "map"
            goal_pose_stamped.header.stamp = self.get_clock().now().to_msg()
            goal_pose_stamped.pose = msg.pose
            self.goal_pose = goal_pose_stamped
        else:
            self.goal_pose = msg

        self.get_path_callback()

        self.is_driving = True
        self.get_logger().info(f"is_driving set to True")

        time.sleep(0.10)
        self.navigator.goToPose(self.goal_pose)
        
        result = self.navigate_to_goal_recursive()

        global navigation_start_status
        global navigation_result
        if result:
            navigation_result = "success"
            navigation_start_status = False
        else :
            navigation_result = "fail"
            navigation_start_status = False
         
        return result

    def navigate_to_goal_recursive(self):
        
        self.distance_to_goal = self.calc_goal_distance(self.robot_pose, self.goal_pose)

        if self.distance_to_goal <= 1.30:
            nav2_end_time = time.time()
            self.is_driving = False
            self.navigator.cancelTask()
            return True  

        if self.check_special_condition():
            self.global_path_while_one_episode = self.initial_path
            
            self.trigger_custom_plan()

            if self.detected_collision:
                self.navigator.cancelTask()
                return False  

            self.episode_data = [] 
            self.semantic_maps = []
            self.semantic_map_save_count = 0
            self.first_inference = True

            # self.need2inference_during_drive = False
            if self.calc_goal_distance(self.robot_pose, self.goal_pose) > 0.85:
                
                return self.goal_callback(self.goal_pose)
            else:
                
                self.is_driving = False
                return True  

        time.sleep(0.1)
        return self.navigate_to_goal_recursive() 

    #처음 trigger 조건 
    def check_special_condition(self):
        if self.semantic_map_save_count == self.semantic_map_input_dim:
            call_diffusion_policy_status = True
        else :
            call_diffusion_policy_status = False

        return call_diffusion_policy_status

    def trigger_custom_plan(self):
        self.custom_plan_in_progress = True
        false_inference_status = False
        if self.first_inference:
            self.navigator.cancelTask()
            self.first_inference = False
        
        start_time = time.time()  
        self.during_create_custom_path = True
        custom_path, only_poses, generate_status = self.create_custom_path()
        
        self.during_create_custom_path = False
        end_time = time.time()  
        
        if not self.first_inference:
            self.navigator.cancelTask()
        
        if generate_status:
            self.navigator.followPath(custom_path)
        else :
            false_inference_status = True
        
        closed_status = False
        self.is_dynamic_obs_in_area_status = False
        self.episode_data = [] 
        self.semantic_maps = []
        self.semantic_map_save_count = 0
        self.accumulate_semantic_map = self.fixed_cost_map.copy()
        
        while self.custom_plan_in_progress and len(self.semantic_maps) < self.semantic_map_input_dim:
            if self.dynamic_in_area_status:
                self.is_dynamic_obs_in_area_status = True

            if self.detected_collision:
                closed_status = True
                self.navigator.cancelTask()
                break
            if self.semantic_map_save_count >=0:
                _, semantic_map, robot_position_image = self.make_semantic_map(self.fixed_cost_map, 
                                                                    self.detected_dynamic_objects, 
                                                                    self.initial_path,
                                                                    self.laser_scan)
                self.semantic_maps.append(semantic_map) 

                self.episode_data.append({
                    'x': robot_position_image[0],
                    'y' : robot_position_image[1]
                }) 
            
            self.semantic_map_save_count += 1
            
            distance_to_goal = self.calc_local_goal_distance(self.robot_pose, self.original_local_goal)           
            
            if distance_to_goal <= 1.5:
                self.navigator.cancelTask()
                closed_status = True
                self.get_logger().info(f"is_driving set to False")
                break
            
            time.sleep(0.1)

        if not self.is_dynamic_obs_in_area_status:
            self.undetected_dynamic_obs_during_generative_plan += 1
        else: 
            self.undetected_dynamic_obs_during_generative_plan = 0


        if self.undetected_dynamic_obs_during_generative_plan >=1 :
            
            self.undetected_dynamic_obs_during_generative_plan = 0
            closed_status = True

        if closed_status: 
            self.custom_plan_in_progress = False
        else : 
            self.trigger_custom_plan()

        return 
        
    def check_path_closed_to_goal(self, only_poses):
        closed_goal_status = False
        first_dist_calc = True
        min_dist = 0
        for point in reversed(only_poses):

            distance_to_goal = np.sqrt((point[0] - self.original_local_goal[0]) ** 2 +
                                    (point[1] - self.original_local_goal[1]) ** 2)
            if first_dist_calc:
                min_dist = distance_to_goal

            if distance_to_goal < min_dist:
                min_dist = distance_to_goal
            if distance_to_goal < 1.0:
                closed_goal_status = True
        return closed_goal_status, min_dist

    def create_custom_path(self):
        custom_path = Path()
        custom_path.header.frame_id = 'map'
        
        crop_goal_for_input = []

        for _ in range(self.semantic_map_input_dim):
            crop_goal_for_input.append((self.transformed_initial_crop_goal[0], self.transformed_initial_crop_goal[1]))
        robot_pose_for_input = np.array([[entry['x'], entry['y']] for entry in self.episode_data])

        path = self.inference_path(robot_pose_for_input, self.semantic_maps, crop_goal_for_input)

        if path is None:
            return None, None, False

        
        poses, only_poses = self.get_back_original_path(path)
        
        self.publish_path_markers(poses)
    
        custom_path.poses = poses
        
        return custom_path, only_poses, True

    def get_back_original_path(self,path):
        poses = []
        only_poses = []
        for point in path:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            unnorm_x = (point[0]*48 + 48)
            unnorm_y = (point[1]*48 + 48)

            cv2.circle(self.accumulate_semantic_map, (int(unnorm_x), int(unnorm_y)), 1, (255, 0, 0), -1)
            original_x = unnorm_x / self.scale_factor
            original_y = unnorm_y / self.scale_factor

            rotated_x = self.cropped_map_size - 1 - original_x
            rotated_y = original_y

            retransformed_x = self.cropped_map_size - 1 - rotated_y
            retransformed_y = rotated_x

            map_x = retransformed_x * self.resolution + self.origin_x + self.x_min * self.resolution
            map_y = retransformed_y * self.resolution + self.origin_y + self.y_min * self.resolution

            pose.pose.position.x = map_x
            pose.pose.position.y =map_y
            pose.pose.orientation.w = 1.0
            poses.append(pose)
            only_poses.append([map_x, map_y])

        cv2.imwrite(f"./path_generated_{self.map_count}.png", self.accumulate_semantic_map)
        self.map_count += 1
        self.publish_semantic_map(self.accumulate_semantic_map)
        return poses, only_poses

    def publish_path_markers(self, poses):
        for i, pose in enumerate(poses):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "path_markers"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = pose.pose
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            self.marker_publisher.publish(marker)

    def local_goal_marking(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "local_goal_marker"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(self.original_local_goal[0])
        marker.pose.position.y = float(self.original_local_goal[1])
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        marker.color.a = 1.0
        marker.color.r = 0.4
        marker.color.g = 1.0
        marker.color.b = 1.0

        self.local_goal_marker_publisher.publish(marker)

    def get_dynamic_obs_pose(self, msg):
        self.detected_obs = msg
        dynamic_objects = []
        static_objects = []

        for marker in self.detected_obs.markers:
            position = marker.pose
            if marker.color.r == 1.0 and marker.color.g == 0.0 and marker.color.b == 0.0:
                dynamic_objects.append(position)
            elif marker.color.r == 0.0 and marker.color.g == 1.0 and marker.color.b == 0.0:
                static_objects.append(position)

        if not dynamic_objects and not static_objects:
            self.detected_status = True
            return
        
        if self.is_driving :
            if dynamic_objects: 
                if self.record_first_state:
                    _, _ = self.set_crop_center_position(self.robot_pose)
                    
                self.dynamic_in_area_status, self.detected_dynamic_objects = self.check_dynamic_obs_in_area(self.initial_crop_center_x, self.initial_crop_center_y, dynamic_objects)

            
    def update_semantic_map(self):
        
        if not self.custom_plan_in_progress and self.is_driving:

            if self.dynamic_in_area_status:
                
                self.semantic_map_save_count += 1
                self.undetectable_count_total = 0

                self.semantic_record_status = True

                accumlate_image, semantic_map, robot_position_image = self.make_semantic_map(self.fixed_cost_map, 
                                                                 self.detected_dynamic_objects, 
                                                                 self.initial_path,
                                                                 self.laser_scan)
            
                self.publish_semantic_map(accumlate_image) 
                self.semantic_maps.append(semantic_map) 
                self.episode_data.append({
                    'x': robot_position_image[0],
                    'y' : robot_position_image[1]
                })

            else:
                if self.semantic_record_status:
                    self.semantic_map_save_count += 1
                    self.undetectable_count_total += 1

                    if self.undetectable_count_total == 5:
                    # abort
                        self.reset_semantic_map_seq()
                        return
                    
                    accumlate_image, semantic_map, robot_position_image = self.make_semantic_map(self.fixed_cost_map, 
                                                                 self.detected_dynamic_objects, 
                                                                 self.initial_path,
                                                                 self.laser_scan)
             
                    self.publish_semantic_map(accumlate_image)
                    self.semantic_maps.append(semantic_map) 

                    self.episode_data.append({
                        'x': robot_position_image[0],
                        'y' : robot_position_image[1]
                    })   

    def publish_semantic_map(self,image):
    # 이미지 메시지로 변환 및 발행
        bridge = CvBridge()
        image_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = "map"
        self.semantic_map_publisher.publish(image_msg)  

    def check_dynamic_obs_in_area(self, center_x, center_y, dynamic_objects):
        dynamic_objects_in_area = []
        status = False
        # self.get_logger().info(f"center_x: {center_x}, center_y: {center_y}")

        for obj in dynamic_objects:
            distance = ((center_x - obj.position.x) ** 2 + (center_y - obj.position.y) ** 2) ** 0.5
            if distance < 1.8:
                dynamic_objects_in_area.append(obj)
            
                status = True
        
        return status, dynamic_objects_in_area

    
    def set_crop_center_position(self, current_pose):
        if self.record_first_state:
            initial_robot_x = current_pose.position.x
            initial_robot_y = current_pose.position.y

            initial_orientation_q = [
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w
            ]
            (_, _, init_yaw) = tf_transformations.euler_from_quaternion(initial_orientation_q)

            center_offset = self.d_offset

            if -np.pi / 4 <= init_yaw < np.pi / 4:
                crop_center_x = initial_robot_x + center_offset
                crop_center_y = initial_robot_y
            elif np.pi / 4 <= init_yaw < 3 * np.pi / 4:  
                crop_center_x = initial_robot_x
                crop_center_y = initial_robot_y + center_offset
            elif -3 * np.pi / 4 <= init_yaw < -np.pi / 4:  
                crop_center_x = initial_robot_x
                crop_center_y = initial_robot_y - center_offset
            else:  
                crop_center_x = initial_robot_x - center_offset
                crop_center_y = initial_robot_y

            self.initial_crop_center_x = crop_center_x
            self.initial_crop_center_y = crop_center_y
            self.publish_crop_center_marker()
            self.publish_boundary_marker()

            self.crop_pose = Pose()
            self.crop_pose.position.x = self.initial_crop_center_x
            self.crop_pose.position.y = self.initial_crop_center_y
            self.crop_pose.position.z = 0.0
            self.crop_pose.orientation.w = 1.0

            crop_x_in_img = int((crop_center_x - self.origin_x) / self.resolution)
            crop_y_in_img = int((crop_center_y - self.origin_y) / self.resolution)

            self.x_min = crop_x_in_img - self.half_crop_size
            self.x_max = crop_x_in_img + self.half_crop_size
            self.y_min = crop_y_in_img - self.half_crop_size
            self.y_max = crop_y_in_img + self.half_crop_size

            x_min_map = max(0, self.x_min)
            x_max_map = min(self.width, self.x_max)
            y_min_map = max(0, self.y_min)
            y_max_map = min(self.height, self.y_max)

            map_image = self.costmap_np[y_min_map:y_max_map, x_min_map:x_max_map, :]

            self.fixed_cost_map = map_image
            self.accumulate_semantic_map = map_image.copy()
            self.cropped_map_size = self.fixed_cost_map.shape[0]

            self.publish_crop_semantic_map(self.fixed_cost_map)
            
            cv2.imwrite("./initial_cropped_map.png", self.fixed_cost_map)

        return self.initial_crop_center_x, self.initial_crop_center_y
    
    def publish_crop_semantic_map(self, cropped_map):
        cropped_publish_map = cropped_map.astype(np.uint8)
        bridge = CvBridge()
        image_msg = bridge.cv2_to_imgmsg(cropped_publish_map, encoding="bgr8")
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = "map"
        self.semantic_map_publisher.publish(image_msg)
        
    def publish_crop_center_marker(self):
        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "crop_center"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.initial_crop_center_x
        marker.pose.position.y = self.initial_crop_center_y
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        
        marker.color.g = 0.0
        if self.semantic_record_status:
            marker.color.r = 0.0
            marker.color.b = 1.0
        else:
            marker.color.r = 1.0
            marker.color.b = 0.0

        self.crop_center_marker_publisher.publish(marker)

    def reset_semantic_map_seq(self):
        self.episode_data = [] 
        self.semantic_maps = []
        self.semantic_record_status = False
        self.accumulate_semantic_map = None
        self.undetectable_count_total = 0
        self.semantic_map_save_count = 0
        self.record_first_state = True

    
    def make_semantic_map(self, cropped_map, dynamic_objects, initial_path, laser_scan):
        current_robot_x = int((self.robot_pose.position.x - self.origin_x) / self.resolution)
        current_robot_y = int((self.robot_pose.position.y - self.origin_y) / self.resolution)
    
        current_robot_x_on_image = current_robot_x - self.x_min
        current_robot_y_on_image = current_robot_y - self.y_min
        orientation_q = [self.robot_pose.orientation.x, self.robot_pose.orientation.y, self.robot_pose.orientation.z, self.robot_pose.orientation.w]
        (_, _, yaw) = tf_transformations.euler_from_quaternion(orientation_q)
        
        semantic_cropped_map = cropped_map.copy()

        angle_min = laser_scan.angle_min
        angle_increment = laser_scan.angle_increment
        ranges = laser_scan.ranges
        for i, r in enumerate(ranges):
            if r < laser_scan.range_max:
                angle = angle_min + i * angle_increment + yaw  
                scan_x = int(current_robot_x_on_image + (r * np.cos(angle)) / self.resolution)
                scan_y = int(current_robot_y_on_image + (r * np.sin(angle)) / self.resolution)

                cv2.circle(self.accumulate_semantic_map, (scan_x,scan_y), 1, (0,0,0), -1)
                cv2.circle(semantic_cropped_map, (scan_x,scan_y), 1, (0,0,0), -1)
        
        if 0 <= current_robot_x_on_image < cropped_map.shape[1] and 0 <= current_robot_y_on_image < cropped_map.shape[0]:
            cv2.circle(self.accumulate_semantic_map, (current_robot_x_on_image, current_robot_y_on_image), 1, (0, 0, 255), -1)
            cv2.circle(semantic_cropped_map, (current_robot_x_on_image, current_robot_y_on_image), 1, (0, 0, 255), -1)

        if self.record_first_state:
            self.record_first_state = False
            goal_within_crop = False
            local_goal = None

            if self.goal_pose:
                goal_x = int((self.goal_pose.pose.position.x - self.origin_x) / self.resolution)
                goal_y = int((self.goal_pose.pose.position.y - self.origin_y) / self.resolution)
                goal_x_on_image = goal_x - self.x_min
                goal_y_on_image = goal_y - self.y_min

                if 0 <= goal_x_on_image < cropped_map.shape[1] and 0 <= goal_y_on_image < cropped_map.shape[0]:
                    cv2.circle(self.accumulate_semantic_map, (goal_x_on_image, goal_y_on_image), 1, (0, 255, 0), -1)
                    cv2.circle(semantic_cropped_map, (goal_x_on_image, goal_y_on_image), 1, (0, 255, 0), -1)
                    goal_within_crop = True

                    self.initial_crop_goal = (goal_x_on_image, goal_y_on_image)
                    transnformed_x, transnformed_y = self.transform_coordinates(goal_x_on_image, goal_y_on_image, self.cropped_map_size)
                    self.original_local_goal = (self.goal_pose.pose.position.x, self.goal_pose.pose.position.y)
                    local_goal = (transnformed_x, transnformed_y)

            if not goal_within_crop:
                path_intersects = []
                cropped_path = []
            
                for pose in initial_path.poses:
                    x = int((pose.pose.position.x - self.origin_x) / self.resolution) - self.x_min
                    y = int((pose.pose.position.y - self.origin_y) / self.resolution) - self.y_min
                    cropped_path.append((x, y))

                for i in range(len(cropped_path) - 1):
                    x1, y1 = cropped_path[i]
                    x2, y2 = cropped_path[i + 1]       
                    if (0 <= x1 < cropped_map.shape[1] and 0 <= y1 < cropped_map.shape[0]) and \
                        not (0 <= x2 < cropped_map.shape[1] and 0 <= y2 < cropped_map.shape[0]):
                            intersect = (x1, y1)
                            path_intersects.append(intersect)
                if path_intersects:
                    # self.get_logger().info(f"탐색된 local goal : {path_intersects}")
                    x_intersect, y_intersect = path_intersects[-1]
                    closest_intersect = (x_intersect, y_intersect)
                    self.initial_crop_goal = (x_intersect, y_intersect)
                    
                    self.original_local_goal= (x_intersect * self.resolution + self.origin_x + self.x_min * self.resolution, y_intersect * self.resolution + self.origin_y + self.y_min*self.resolution)
                    
                    transnformed_intersect_x, transnformed_intersect_y = self.transform_coordinates(x_intersect, y_intersect, self.cropped_map_size)
                    local_goal = (transnformed_intersect_x, transnformed_intersect_y)
                    cv2.circle(self.accumulate_semantic_map, closest_intersect, 1, (0, 255, 0), -1)
                    cv2.circle(semantic_cropped_map, closest_intersect, 1, (0, 255, 0), -1)

            if local_goal is None:
                self.get_path_callback()
                return self.make_semantic_map(self.fixed_cost_map, 
                                              self.detected_dynamic_objects,
                                              self.initial_path,
                                              self.laser_scan)
                # resized_transform_goal_x = self.initial_crop_goal[0]
                # resized_transform_goal_y = self.initial_crop_goal[1]
            resized_transform_goal_x, resized_transform_goal_y = self.resize_positions(local_goal[0] , local_goal[1], self.cropped_map_size)
            self.transformed_initial_crop_goal = (resized_transform_goal_x, resized_transform_goal_y)      
        else: 
            cv2.circle(self.accumulate_semantic_map, self.initial_crop_goal, 1, (0, 255, 0), -1)
            cv2.circle(semantic_cropped_map, self.initial_crop_goal, 1, (0, 255, 0), -1)
        if dynamic_objects:
            for obj in dynamic_objects:
                # self.get_logger().info(f'obj {obj}, {type(obj)}')

                obj_x = int((obj.position.x - self.origin_x) / self.resolution - self.x_min)
                obj_y = int((obj.position.y - self.origin_y) / self.resolution - self.y_min)
                cv2.circle(self.accumulate_semantic_map, (obj_x, obj_y), int(0.3 / self.resolution), (255, 255, 0), -1)
                cv2.circle(semantic_cropped_map, (obj_x, obj_y), int(0.3 / self.resolution), (255, 255, 0), -1)
  
        # self.local_goal_marking(self.original_local_goal)
        transform_x , transform_y = self.transform_coordinates(current_robot_x_on_image, current_robot_y_on_image, self.cropped_map_size)
        resized_transform_x, resized_transform_y = self.resize_positions(transform_x, transform_y, self.cropped_map_size)
        robot_position_image = (resized_transform_x, resized_transform_y)

        semantic_cropped_map = self.save_img_correct_side(semantic_cropped_map)
        self.accumulate_semantic_map = semantic_cropped_map.copy()
        cv2.imwrite(f"./semantic_map_total_{self.map_count}.png", self.accumulate_semantic_map)
        cv2.imwrite(f"/./semantic_map_{self.semantic_map_save_count}_ep{self.map_count}.png", semantic_cropped_map)
        
        return self.accumulate_semantic_map, semantic_cropped_map, robot_position_image

    def publish_boundary_marker(self):
        # 녹화시 녹화될 범위를 RVIZ에 표현.
        if self.initial_crop_center_x and self.initial_crop_center_y:
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "boundary"
                marker.id = 0
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD

                # Define the corners of the boundary square
                boundary  = 2.0 
                points = [
                    (self.initial_crop_center_x - boundary, self.initial_crop_center_y - boundary, 0.0),
                    (self.initial_crop_center_x - boundary, self.initial_crop_center_y + boundary, 0.0),
                    (self.initial_crop_center_x + boundary, self.initial_crop_center_y + boundary, 0.0),
                    (self.initial_crop_center_x + boundary, self.initial_crop_center_y - boundary, 0.0),
                    (self.initial_crop_center_x - boundary, self.initial_crop_center_y - boundary, 0.0)  # Close the loop
                ]

                for point in points:
                    p = Point()
                    p.x, p.y, p.z = point
                    marker.points.append(p)

                marker.scale.x = 0.1
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.b = 0.0
                if self.semantic_record_status:
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    
                else :
                    marker.color.r = 1.0
                    marker.color.g = 1.0

                self.marker_publisher.publish(marker)

    def transform_coordinates(self,x, y, image_size):
        rotated_x = y
        rotated_y = image_size - 1 - x

        flipped_x = image_size -1 - rotated_x
        flipped_y = rotated_y

        return flipped_x, flipped_y
    
    def resize_positions(self, x,y,image_size):
        self.scale_factor = 96 / image_size
        transform_resized_x = int(x * self.scale_factor)
        transform_resized_y = int(y * self.scale_factor)
        return transform_resized_x, transform_resized_y
    
    def save_img_correct_side(self, image):
        rotated_map_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        fliped_map_image = cv2.flip(rotated_map_image, 1)
        
        resized_image = cv2.resize(fliped_map_image, (96, 96))
        return resized_image
class Cmd_vel_subscriber(Node):
    def __init__(self):
        super().__init__('cmdvel_subscriber')

        # Subscriber for /cmd_vel topic
        self.subscription = self.create_subscription(
            Twist,
            cmd_vel_topic_name,
            self.cmd_vel_callback,
            10
        )

    def cmd_vel_callback(self, cmd_vel_msg):
        # Extract linear.x and angular.z from the Twist message
        linear_x = cmd_vel_msg.linear.x
        angular_z = cmd_vel_msg.angular.z

        if navigation_start_status:
            global cmd_vel_logs
            cmd_vel_logs.append([linear_x, angular_z])
class RecordRobotPosition(Node):
    def __init__(self):
        super().__init__('record_robot_position')
        self.previous_x = None
        self.previous_y = None
        self.total_distance = 0.0
        self.record_robot_distance_timer = self.create_timer(0.1, self.record_robot_distance)
        
        self.record_position = []
        self.previous_navigation_status = False
        self.root_dir = f'{save_log_dir}{str(int(seed))}/'
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        
    def record_robot_distance(self):
        if navigation_start_status:
            current_x = robot_current_pose.position.x
            current_y = robot_current_pose.position.y
            if self.previous_x is not None and self.previous_y is not None:
                distance_increment = math.sqrt(
                    (current_x - self.previous_x) ** 2 + (current_y - self.previous_y) ** 2 
                )
                self.total_distance += distance_increment
        
            self.previous_x = current_x
            self.previous_y = current_y

            image_x = int((current_x - origin_x) / resolution)
            image_y = int((current_y - origin_y) / resolution)
            
            self.record_position.append([self.previous_x, self.previous_y, image_x, image_y])
        else:
            if self.previous_navigation_status:
                self.save_navigation_logs()
                
        self.previous_navigation_status = navigation_start_status
    
    def record_robot_position_on_map(self):
        
        original_map_in_trajectory = cv2.flip(map_image, 1)  
        cost_map_in_trajectory = cv2.flip(cost_map_image, 1) 

        for position in self.record_position:
            image_x = int((position[0] - origin_x) / resolution)
            image_y = int((position[1] - origin_y) / resolution)
            image_y = height - image_y  

            image_x = max(0, min(image_x, width - 1))
            image_y = max(0, min(image_y, height - 1))
            
            cv2.circle(cost_map_in_trajectory, (image_x, image_y), 2, (0, 0, 255), -1)  
            cv2.circle(original_map_in_trajectory, (image_x, image_y), 2, (0, 0, 255), -1)

        cv2.imwrite(f'{self.root_dir}robot_trajectory_map.png', cost_map_in_trajectory)
        cv2.imwrite(f'{self.root_dir}robot_trajectory_original_map.png', original_map_in_trajectory)

    def save_navigation_logs(self):
        end_time = time.time()
        duration = end_time - start_time

        if duration > 0.0:
            average_speed = self.total_distance / duration
        else:
            average_speed = 0.0

        log_data = {
            'driving_time': [duration],
            'driving_distance': [self.total_distance],
            'avg_speed': [average_speed]
        }       
        log_df = pd.DataFrame(log_data)
        log_df.to_csv(f'{self.root_dir}SEED_{seed}_driving_log.csv', index=False)

        self.record_robot_position_on_map()

        position_df = pd.DataFrame(self.record_position, columns=['x', 'y', 'image_x','image_y'])
        position_df.to_csv(f'{self.root_dir}robot_positions.csv', index=False)
        
        cmd_vel_distribution_df = pd.DataFrame(cmd_vel_logs, columns=['linear_x', 'angular_y'])
        cmd_vel_distribution_df.to_csv(f'{self.root_dir}cmd_vel_distribution.csv', index=False)
        
        global navigation_result
        if navigation_result is None:
            navigation_result = "unknown"
        status_file_path = os.path.join(self.root_dir, f"SEED_{seed}_navigation_result.txt")
        with open(status_file_path, "w") as f:
            f.write(navigation_result)
        
def main():
    rclpy.init()
    test_dynamic_obs_avoidance = TestDynamicObsAvoidance()
    record_robot_position  = RecordRobotPosition()
    record_cmd_vel = Cmd_vel_subscriber()
    executor = MultiThreadedExecutor()
    executor.add_node(test_dynamic_obs_avoidance)
    executor.add_node(record_robot_position)
    executor.add_node(record_cmd_vel)

    try:
        test_dynamic_obs_avoidance.get_logger().info('Beginning client, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        test_dynamic_obs_avoidance.get_logger().info('Keyboard interrupt, shutting down.\n')
    test_dynamic_obs_avoidance.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":

    main()