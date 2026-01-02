"""
Date: 16/12/2025
Desc: FastMCP Server for G1 Retail Demo
Version 2: Uses ARuCo detection for grabbing items.
Author: Reuben Lim
"""
# TODO: Fill up the bottles and container to have some weight
# TODO: Try relative pose control for returning to shelf after grab

import time
import numpy as np
import traceback
import cv2
import threading
import argparse
import rclpy
import logging
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from a2d_sdk.robot import RobotController, RobotDds, Slam
from fastmcp import FastMCP
try:
	from cv_bridge import CvBridge
	_HAS_CV_BRIDGE = True
except Exception:
	CvBridge = None
	_HAS_CV_BRIDGE = False

"""Hard coded params""" 
LEFT_MAX_XYZ = [0.813, 0.894, 1.347]
LEFT_MIN_XYZ = [0.155, -0.1, 0.0]
RIGHT_MAX_XYZ = [0.813, 0.1, 1.347]
RIGHT_MIN_XYZ = [0.155, -0.895, 0.0]
RESET_JOINT_ANGLES = [-1.3105124235153198, 0.7436492443084717, 1.0693359375, -1.4095935821533203, 
					  0.3221793472766876, 1.0558122396469116, -0.011185449548065662, 1.1802656650543213, 
					  -0.6814573407173157, -0.6968657970428467, 1.2513394355773926, -0.5542469024658203, 
					  -1.3956334590911865, -0.007189400028437376]
# Initial shelf joint angles for each arm/level
LEFT_TOP_JOINT_ANGLES = [-2.067196846008301, 0.4773622155189514, 2.0629565715789795, -1.1641069650650024, 
						-0.28460949659347534, 0.010278049856424332, 0.09187424927949905, 1.1798816919326782, 
						-0.18097396194934845, -0.6960979104042053, 1.140130639076233, -0.5653974413871765, 
						-1.428910732269287, -0.011150550097227097]
RIGHT_TOP_JOINT_ANGLES = [-1.2916665077209473, 0.32284244894981384, 1.0134261846542358, -1.606621503829956, 
						  0.5536361336708069, 0.8409329652786255, -0.08965810388326645, 1.9115253686904907, 
						-0.38192814588546753, -1.9297605752944946, 1.0358843803405762, 0.10496174544095993, 
						-0.18922780454158783, 0.02709984965622425]
LEFT_BOTTOM_JOINT_ANGLES = [-0.7769088745117188, 0.6959932446479797, 0.6288281679153442, -1.1337440013885498, 
						0.532451868057251, -0.0016054000006988645, -0.0061947498470544815, 1.1798816919326782, 
						-0.18097396194934845, -0.6960979104042053, 1.140130639076233, -0.5653974413871765, 
						-1.428910732269287, -0.011150550097227097]
RIGHT_BOTTOM_JOINT_ANGLES = [-1.2916665077209473, 0.32284244894981384, 1.0134261846542358, -1.606621503829956, 
							 0.5536361336708069, 0.8409329652786255, -0.08965810388326645, 0.6606919169425964, 
						-0.6746518611907959, -0.4877973198890686, 1.2155845165252686, -0.6142574548721313, 
						-0.03964640200138092, 0.02223130129277706]
BACKOFF_X = 0.5
LARGE_CHIP_ARUCO_ID = [0, 1, 2, 3]
SMALL_CHIP_ARUCO_ID = [4, 5, 6]
SMALL_BOTTLE_ARUCO_ID = [7, 8, 9]
PLUS_100_ARUCO_ID = [10, 11, 12]
# Initial inventory counts
INITIAL_LARGE_CHIP_COUNT = 3
INITIAL_SMALL_CHIP_COUNT = 3
INITIAL_100_PLUS_COUNT = 3
INITIAL_SMALL_BOTTLE_COUNT = 3
# SLAM MAP ID
SLAM_MAP_ID = 7

def poll_state(getter, length, name, timeout=2.0, interval=0.1):
	"""Poll robot state until valid data is received."""
	deadline = time.time() + timeout
	last_vals = None
	while time.time() < deadline:
		vals, _ = getter()
		last_vals = vals
		try:
			actual_len = len(vals)
		except Exception:
			actual_len = None
		all_numeric = (actual_len == length) and all(
			isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit())
			for v in vals
		)
		if all_numeric:
			return list(vals)
		time.sleep(interval)
	raise RuntimeError(f"{name} not ready within {timeout}s, last vals={last_vals!r}")


class G1Controller(Node):
	def __init__(self, aruco_dict_id: int, output_video: str | None = None,
				 center_tolerance: int = 15, control_gain: float = 0.0001,
				 forward_step: float = 0.02, display_window: bool = True):
		super().__init__('g1_controller')
		
		# Thread synchronization locks
		self._subscription_lock = threading.Lock()  # Protects subscription operations
		self._image_lock = threading.Lock()  # Protects image-related shared state
		
		# Inventory
		self._large_chip = INITIAL_LARGE_CHIP_COUNT
		self._small_chip = INITIAL_SMALL_CHIP_COUNT
		self._100_plus = INITIAL_100_PLUS_COUNT
		self._small_bottle = INITIAL_SMALL_BOTTLE_COUNT

		self.output_video = output_video
		self.center_tolerance = center_tolerance
		self.control_gain = control_gain
		self.forward_step = forward_step
		
		# Camera and detection state
		self.bridge = CvBridge() if _HAS_CV_BRIDGE else None
		self.latest_image = None
		self.image_center = None
		self.marker_center = None
		self.marker_id = None
		self.image_frame_index = 0  # Track new images received
		
		# Initialize ArUco detector
		self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
		self.aruco_params = cv2.aruco.DetectorParameters()
		self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
		
		# Video writer
		self.video_writer = None
		self.frame_count = 0

		# Display window flag
		self.display_window = display_window
		
		# Robot controllers
		self.get_logger().info("Initializing robot controllers...")
		self.robot_controller = RobotController()
		self.robot_dds = RobotDds()
		time.sleep(2.0)

		# Init grippers to open position
		self.left_gripper = 0.0 
		self.right_gripper = 0.0 
		self.robot_dds.move_gripper([self.left_gripper, self.right_gripper])

		self.arm_to_move = 'left'  # To set either the "left" or "right" arm for visual servoing
		self.level = 'top'  # Shelf level to search ('top' or 'bottom')
		self.bias_y = 0  # Y-axis bias for centering
		self.placemovement_file = None
		self.item_to_grab = None
		self.item_id_list = [] # List of valid ArUco IDs for the selected item

		# Initial arm positions to start searching at different locations of the shelf
		self.left_top_joint_angles = LEFT_TOP_JOINT_ANGLES
		self.left_bottom_joint_angles = LEFT_BOTTOM_JOINT_ANGLES
		self.right_top_joint_angles = RIGHT_TOP_JOINT_ANGLES
		self.right_bottom_joint_angles = RIGHT_BOTTOM_JOINT_ANGLES
		
		# Initialize SLAM module
		self.slam = Slam()
		self.slam.switch_nav_mode(1)  # 1: Manual, 2: Auto; NOTE: We keep it in manual mode by default because continuous operation under auto mode has a weird error
		self.slam.set_agv_col_ctrl(False)
		self.slam.switch_map(SLAM_MAP_ID)

		# Subscribers to camera
		self.left_color_sub = None
		self.right_color_sub = None
		
		# Subscribe based on initial arm selection
		self._update_camera_subscription()

		# Reset arm positions
		self.reset_arm_position()
		time.sleep(1.0)

		# Return the robot to home position (in this case the shelf facing pose) if it is not already there
		chassis_pos = self.slam.get_chassis_position()
		curr_x, curr_y, curr_angle = chassis_pos[0]['agv_pos_x'], chassis_pos[0]['agv_pos_y'], chassis_pos[0]['agv_angle']
		if curr_x != 0.0 or curr_y != 0.0 or curr_angle != 0.0:
			self.get_logger().info("Returning robot to shelf facing pose...")
			self.slam_move_to_pose(0,0,0)
			time.sleep(1.0)

		# Display window
		if self.display_window:
			cv2.namedWindow('Hand Camera Display', cv2.WINDOW_NORMAL)
		
		self.get_logger().info(f"Visual servo initialized, subscribing to /camera/hand_left_color and /camera/hand_right_color")

	def _on_image(self, msg: Image):
		"""Callback for incoming camera images."""
		try:
			if self.bridge is not None:
				cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
			else:
				# Fallback decoding
				arr = np.frombuffer(msg.data, dtype=np.uint8)
				if msg.encoding in ('rgb8', 'bgr8'):
					arr = arr.reshape((msg.height, msg.width, 3))
					if msg.encoding == 'rgb8':
						cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
					else:
						cv_img = arr
				elif msg.encoding == 'mono8':
					cv_img = arr.reshape((msg.height, msg.width))
					cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
				else:
					arr = arr.reshape((msg.height, msg.width, 3))
					cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
			
			with self._image_lock:
				self.latest_image = cv_img.copy()
				h, w = cv_img.shape[:2]
				self.image_center = (w // 2, h // 2)
				self.image_frame_index += 1
			
			# Detect ArUco markers (outside lock to avoid holding lock during heavy processing)
			corners, ids, rejected = self.aruco_detector.detectMarkers(cv_img)
			# self._logger.info(f'Detected IDs: {ids}')
 
			if ids is not None and len(self.item_id_list) > 0:
				# Get the valid IDs
				valid_indices = [i for i, marker_id in enumerate(ids.flatten()) if marker_id in self.item_id_list]
				min_valid_index = np.argmin(ids[valid_indices]) if valid_indices else None
				
				if min_valid_index is not None:
					# Get the lowest valid index
					min_ID = ids[valid_indices][min_valid_index][0]
					valid_corners = [ corners[i] for i in valid_indices ]
					corner = valid_corners[min_valid_index]
					# self._logger.info(f'Chosen corner: {corner}, ID: {min_ID}')
		
					# Calculate marker center
					center_x = int(corner[0][:, 0].mean())
					center_y = int(corner[0][:, 1].mean())
					marker_center_val = (center_x, center_y)

					# Calculate marker size (width of bounding box)
					min_x = int(corner[0][:, 0].min())
					max_x = int(corner[0][:, 0].max())
					marker_size_val = max_x - min_x
					
					# Update shared state under lock
					with self._image_lock:
						self.marker_id = min_ID
						self.marker_center = marker_center_val
						self.marker_size = marker_size_val
					
					if self.display_window or self.output_video:
						# Draw on local copy (outside lock)
						cv2.aruco.drawDetectedMarkers(cv_img, corners, ids)
						cv2.circle(cv_img, marker_center_val, 8, (0, 0, 255), -1)
						cv2.drawMarker(cv_img, marker_center_val, (0, 255, 0), 
									cv2.MARKER_CROSS, 30, 3)
						
						with self._image_lock:
							image_center = self.image_center
						
						cv2.circle(cv_img, image_center, 8, (255, 0, 0), -1)
						cv2.drawMarker(cv_img, image_center, (255, 0, 0), 
									cv2.MARKER_CROSS, 30, 3)
						
						cv2.line(cv_img, marker_center_val, image_center, (255, 255, 0), 2)
						
						# Calculate and display offset
						offset_x = center_x - image_center[0]
						offset_y = center_y - image_center[1]
						offset_distance = np.sqrt(offset_x**2 + offset_y**2)
						
						# Display info
						cv2.putText(cv_img, f"Marker ID: {min_ID}", (10, 30),
									cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
						cv2.putText(cv_img, f"Offset: ({offset_x}, {offset_y})", (10, 60),
									cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
						cv2.putText(cv_img, f"Distance: {offset_distance:.1f}px", (10, 90),
									cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
						cv2.putText(cv_img, f"Marker Size: {marker_size_val}px", (10, 120),
									cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
						
						# Check if centered
						if offset_distance < self.center_tolerance:
							cv2.putText(cv_img, "CENTERED", (10, 150),
										cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
						else:
							cv2.putText(cv_img, "ADJUSTING", (10, 150),
										cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			else:
				with self._image_lock:
					self.marker_center = None
					self.marker_id = None
					self.marker_size = None
				if self.display_window or self.output_video:
					cv2.putText(cv_img, "NO VALID MARKER DETECTED", (10, 30),
								cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			
			# Show image
			if self.display_window:
				cv2.imshow('Hand Camera Display', cv_img)
				cv2.waitKey(1)
			
			# Write to video if enabled
			if self.output_video:
				if self.video_writer is None:
					h, w = cv_img.shape[:2]
					fourcc = cv2.VideoWriter_fourcc(*'mp4v')
					self.video_writer = cv2.VideoWriter(self.output_video, fourcc, 30.0, (w, h))
					if self.video_writer.isOpened():
						self.get_logger().info(f"Recording to '{self.output_video}'")
				
				if self.video_writer is not None:
					self.video_writer.write(cv_img)
					self.frame_count += 1
					
		except Exception as e:
			tb = traceback.format_exc()
			self.get_logger().error(f'Failed to process image: {e}\n{tb}')

	def _update_camera_subscription(self):
		"""Subscribe/unsubscribe based on active arm. MUST be called with _subscription_lock held."""
		# Destroy existing subscriptions
		if self.left_color_sub is not None:
			self.destroy_subscription(self.left_color_sub)
			self.left_color_sub = None
		if self.right_color_sub is not None:
			self.destroy_subscription(self.right_color_sub)
			self.right_color_sub = None
		
		# Create subscription for active arm only
		if self.arm_to_move == 'left':
			self.left_color_sub = self.create_subscription(
				Image, '/camera/hand_left_color',
				self._on_image, 
				QoSProfile(depth=3, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE, lifespan=Duration(seconds=0.1)) 
			)
			self.get_logger().info("Subscribed to left hand camera")
		else:
			self.right_color_sub = self.create_subscription(
				Image, '/camera/hand_right_color',
				self._on_image, 
				QoSProfile(depth=3, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE, lifespan=Duration(seconds=0.1)) 
			)
			self.get_logger().info("Subscribed to right hand camera")

	def set_arm_to_move(self, arm: str):
		"""Set which arm to use for visual servoing ('left' or 'right'). Thread-safe."""
		if arm not in ('left', 'right'):
			raise ValueError("Arm must be 'left' or 'right'")
		
		# Protect subscription changes with lock
		with self._subscription_lock:
			self.arm_to_move = arm
			self._update_camera_subscription()

	def set_level(self, level: str):
		"""Set which shelf level to search on ('top' or 'bottom'). Thread-safe."""
		if level not in ('top', 'bottom'):
			raise ValueError("Level must be 'top' or 'bottom'")
		
		# Protect subscription changes with lock
		with self._subscription_lock:
			self.level = level

	def set_item_to_grab(self, item: str):
		"""Set which item to grab. Thread-safe."""
		if item not in ('large chip', 'small chip', 'small bottle', '100 plus'):
			raise ValueError("Item must be 'large chip', 'small chip', 'small bottle', or '100 plus'")
		with self._subscription_lock:
			self.item_to_grab = item
			if item == 'large chip':
				self.item_id_list = LARGE_CHIP_ARUCO_ID
			elif item == 'small chip':
				self.item_id_list = SMALL_CHIP_ARUCO_ID
			elif item == 'small bottle':
				self.item_id_list = SMALL_BOTTLE_ARUCO_ID
			elif item == '100 plus':
				self.item_id_list = PLUS_100_ARUCO_ID

	def get_marker_size(self):
		"""Get the current marker size in pixels. Thread-safe."""
		with self._image_lock:
			if self.marker_size is None:
				self.get_logger().warn('No marker detected')
				return None
			return self.marker_size
	
	def wait_for_new_image(self, timeout=2.0):
		"""Wait for a new image to be received. Thread-safe."""
		with self._image_lock:
			current_index = self.image_frame_index
		deadline = time.time() + timeout
		
		# Don't spin from background thread - just poll the image index
		while time.time() < deadline:
			with self._image_lock:
				if self.image_frame_index > current_index:
					return True
			time.sleep(0.05)
		
		self.get_logger().warn(f'No new image received within {timeout}s')
		return False
	
	def backoff_arms(self):
		# If the arms are too far front (in X axis ), back them off first
		left_arm_x = self.robot_controller.get_motion_status()['frames']['arm_left_link7']['xyzrpy'][0]
		right_arm_x = self.robot_controller.get_motion_status()['frames']['arm_right_link7']['xyzrpy'][0]
		if left_arm_x > BACKOFF_X or right_arm_x > BACKOFF_X:
			left_delta_x = left_arm_x - BACKOFF_X
			right_delta_x = right_arm_x - BACKOFF_X
			delta_pose = [ -left_delta_x if left_arm_x > BACKOFF_X else 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
						   -right_delta_x if right_arm_x > BACKOFF_X else 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
			self.get_logger().info("Backing off arms before reset...")
			self.robot_controller.trajectory_tracking_control(
				infer_timestamp=int(time.time() * 1e9),
				robot_states={},
				robot_actions=[
					{
						"left_arm": {"action_data": delta_pose[:6], "control_type": "DELTA_POSE"},
						"right_arm": {"action_data": delta_pose[6:12], "control_type": "DELTA_POSE"},
					}
				],
				robot_link="base_link",
				trajectory_reference_time=1.0,
			)
			time.sleep(1.0)
	
	def reset_arm_position(self):
		"""Reset both arms to the predefined reset joint angles."""
		self.backoff_arms()
		# Move the arm to reset joint angles
		robot_actions = [
			{
				"left_arm": {"action_data": RESET_JOINT_ANGLES[:7], "control_type": "ABS_JOINT"},
				"right_arm": {"action_data": RESET_JOINT_ANGLES[7:14], "control_type": "ABS_JOINT"},
			}
		]
		
		self.get_logger().info("Resetting arms to predefined position...")
		self.robot_controller.trajectory_tracking_control(
			infer_timestamp=int(time.time() * 1e9),
			robot_states={},
			robot_actions=robot_actions,
			robot_link="base_link",
			trajectory_reference_time=2.0,
		)
		self.get_logger().info("Arms reset complete.")
		return True
	
	def slam_move_relative(self, linear_x: float, linear_y: float, angular_z: float):
		"""
		Move the robot relatively using SLAM navigation. 
		This function creates a blocking call for the relative movement API by setting the movement goal and checking for completion.
		linear_x: x-coord movement in mm
		linear_y: y-coord movement in mm
		angular_z: rotation around Z axis in degrees
		"""
		self.get_logger().info(f"SLAM moving relatively: x={linear_x}mm, y={linear_y}mm, z={angular_z}deg")
		self.slam.switch_nav_mode(2) # Only be in auto mode when you need to use SLAM
		self.slam.move_to_relative(linear_x, linear_y, angular_z)
		# First, we wait for the robot to start moving
		while True:
			chassis_pos = self.slam.get_chassis_position()
			self.get_logger().debug(f"Chassis speed: linear={chassis_pos[0]['linear_speed']}, angular={chassis_pos[0]['angular_speed']}")
			if abs(chassis_pos[0]["linear_speed"]) > 0 or abs(chassis_pos[0]["angular_speed"]) > 0:
				break
			time.sleep(0.1)
		self.get_logger().info("Robot started moving...")
		# Then, we monitor until speed is zero for a certain duration
		zero_speed_count = 0
		while zero_speed_count < 5:
			chassis_pos = self.slam.get_chassis_position()
			self.get_logger().debug(f"Chassis speed: linear={chassis_pos[0]['linear_speed']}, angular={chassis_pos[0]['angular_speed']}")
			if abs(chassis_pos[0]["linear_speed"]) > 0 or abs(chassis_pos[0]["angular_speed"]) > 0:
				zero_speed_count = 0
			else:
				zero_speed_count += 1
			time.sleep(0.2)
		self.get_logger().info("Robot stopped moving.")
		self.slam.switch_nav_mode(1) # Switch back to manual mode after movement
		return

	def slam_move_to_pose(self, x: float, y: float, angle: float):
		"""
		Move the robot to a specific pose using SLAM navigation. 
		This function creates a blocking call for the navigation API by setting the movement goal and checking for completion.
		x: x-coord position in mm
		y: y-coord position in mm
		angle: rotation around Z axis in degrees
		"""
		self.get_logger().info(f"SLAM moving to pose: x={x}mm, y={y}mm, angle={angle}deg")
		self.slam.switch_nav_mode(2) # Only be in auto mode when you need to use SLAM
		self.slam.navigate_to_pose(x, y, angle)
		# First, we wait for the robot to start moving
		while True:
			chassis_pos = self.slam.get_chassis_position()
			self.get_logger().debug(f"Chassis speed: linear={chassis_pos[0]['linear_speed']}, angular={chassis_pos[0]['angular_speed']}")
			if abs(chassis_pos[0]["linear_speed"]) > 0 or abs(chassis_pos[0]["angular_speed"]) > 0:
				break
			time.sleep(0.1)
		self.get_logger().info("Robot started moving...")
		# Then, we monitor until speed is zero for a certain duration
		zero_speed_count = 0
		while zero_speed_count < 5:
			chassis_pos = self.slam.get_chassis_position()
			self.get_logger().debug(f"Chassis speed: linear={chassis_pos[0]['linear_speed']}, angular={chassis_pos[0]['angular_speed']}")
			if abs(chassis_pos[0]["linear_speed"]) > 0 or abs(chassis_pos[0]["angular_speed"]) > 0:
				zero_speed_count = 0
			else:
				zero_speed_count += 1
			time.sleep(0.2)
		self.get_logger().info("Robot stopped moving.")
		self.slam.switch_nav_mode(1) # Switch back to manual mode after movement
		return

	def publish_wheel(self, linear_x: float, angular_z: float, count: int = 10, rate: float = 1.0):
		"""Publish a TwistStamped to `/mbc/wheel_command`."""
		# Note: using robot_dds.move_wheel will switch the robot to manual mode
		self.slam.switch_nav_mode(1)  # Switch to manual mode
		try:
			for _ in range(int(count)):
					self.robot_dds.move_wheel(linear=linear_x, angular=angular_z)
					time.sleep(1.0 / float(rate))
			self.slam.switch_nav_mode(2)  # Switch back to auto mode
		except KeyboardInterrupt:
			self.slam.switch_nav_mode(2)  # Switch back to auto mode
			return
		
	def replay_movements(self, movement_file):
		"""Replay movements saved as a list of dicts (same format used in other scripts)."""
		movements_list = np.load(movement_file, allow_pickle=True)
		for dic in movements_list:
			if dic.get("Command") == "Move_arm":
				action = {
					"observation_timestamp": int(time.time() * 1e9),
					"arm_cmd": [dic["Joint Angles"]]
				}
				# execute using local robot_controller
				self.robot_controller.trajectory_tracking_control(
					action["observation_timestamp"],
					{},
					[{
						"left_arm": {"action_data": action["arm_cmd"][0][:7], "control_type": "ABS_JOINT"},
						"right_arm": {"action_data": action["arm_cmd"][0][7:14], "control_type": "ABS_JOINT"}
					}],
					"base_link",
					1.0
				)
			elif dic.get("Command") == "Move_right_gripper":
				self.right_gripper = float(dic["Gripper Angle"])
				self.robot_dds.move_gripper([self.left_gripper, self.right_gripper])
			elif dic.get("Command") == "Move_left_gripper":
				self.left_gripper = float(dic["Gripper Angle"])
				self.robot_dds.move_gripper([self.left_gripper, self.right_gripper])
			time.sleep(1.5)
	
	def move_to_shelf_level(self, level: str, arm: str):
		"""Move the specified arm to the predefined joint angles for the shelf level."""
		if arm == 'left':
			if level == 'top':
				joint_angles = self.left_top_joint_angles
			elif level == 'bottom':
				joint_angles = self.left_bottom_joint_angles
			else:
				raise ValueError("Level must be 'top' or 'bottom'")
		elif arm == 'right':
			if level == 'top':
				joint_angles = self.right_top_joint_angles
			elif level == 'bottom':
				joint_angles = self.right_bottom_joint_angles
			else:
				raise ValueError("Level must be 'top' or 'bottom'")
		else:
			raise ValueError("Arm must be 'left' or 'right'")
		
		# Reset the arm positions first
		reset_status = self.reset_arm_position()
		if not reset_status:
			self.get_logger().error("Error resetting arms before moving to shelf level")
			return
		time.sleep(1.0)

		# Move waist up/down
		if level == 'top':
			self.robot_dds.move_waist([0.0, 7])
		else:  # bottom
			self.robot_dds.move_waist([0.0, 2])

		# Build action
		robot_actions = [
			{
				"left_arm": {"action_data": joint_angles[:7], "control_type": "ABS_JOINT"},
				"right_arm": {"action_data": joint_angles[7:14], "control_type": "ABS_JOINT"},
			}
		]
		
		self.get_logger().info(f"Moving {arm} arm to {level} shelf position...")
		self.robot_controller.trajectory_tracking_control(
			infer_timestamp=int(time.time() * 1e9),
			robot_states={},
			robot_actions=robot_actions,
			robot_link="base_link",
			trajectory_reference_time=2.0,
		)
		self.get_logger().info(f"{arm.capitalize()} arm moved to {level} shelf position.")
	
	def check_move_limits(self, delta_pose, arm: str) -> bool:
		"""
		Check if the next movement will exceed arm reach limits.
		delta_pose: [dx, dy, dz]
		"""
		if arm == 'left':
			max_xyz = LEFT_MAX_XYZ
			min_xyz = LEFT_MIN_XYZ
			arm_link = 'arm_left_link7'
		elif arm == 'right':
			max_xyz = RIGHT_MAX_XYZ
			min_xyz = RIGHT_MIN_XYZ
			arm_link = 'arm_right_link7'
		else:
			raise ValueError("Arm must be 'left' or 'right'")
		
		arm_xyzrpy = self.robot_controller.get_motion_status()['frames'][arm_link]['xyzrpy']
		next_x = arm_xyzrpy[0] + delta_pose[0]
		next_y = arm_xyzrpy[1] + delta_pose[1]
		next_z = arm_xyzrpy[2] + delta_pose[2]
		
		if not (min_xyz[0] <= next_x <= max_xyz[0]):
			self.get_logger().error(f'{arm.capitalize()} arm will exceed reach limit for X')
			return False
		if not (min_xyz[1] <= next_y <= max_xyz[1]):
			self.get_logger().error(f'{arm.capitalize()} arm will exceed reach limit for Y')
			return False
		if not (min_xyz[2] <= next_z <= max_xyz[2]):
			self.get_logger().error(f'{arm.capitalize()} arm will exceed reach limit for Z')
			return False
		
		return True

	def move_forward_until_lost(self, robot_states: dict, arm_states: list):
		"""Move the arm forward until marker is no longer detected."""
		marker_size = self.get_marker_size()
		
		if marker_size is None:
			self.get_logger().error('No marker detected, cannot move forward')
			return False
		
		self.get_logger().debug('Starting forward movement until marker is lost...')
		
		# Move forward in steps until marker is lost
		max_steps = 20
		step_count = 0
		consecutive_lost = 0
		required_lost_frames = 2  # Require 2 consecutive frames without detection to confirm
		
		while step_count < max_steps and rclpy.ok():
			# Wait for new image before checking marker detection
			if not self.wait_for_new_image(timeout=2.0):
				self.get_logger().warn('Timeout waiting for image during forward movement')
				continue
			
			marker_size = self.get_marker_size()
			
			if marker_size is None:
				consecutive_lost += 1
				self.get_logger().debug(f'Marker not detected ({consecutive_lost}/{required_lost_frames})')
				
				# Confirm marker is truly lost
				if consecutive_lost >= required_lost_frames:
					self.get_logger().debug(f'Marker lost after {step_count} steps - target reached!')
					return True
			else:
				# Reset counter if marker is detected again
				consecutive_lost = 0

			# Check if next step will exceed reach limits
			if not self.check_move_limits([self.forward_step, 0.0, 0.0], arm=self.arm_to_move):
				return False
			
			# Create delta pose for forward movement (positive x)
			with self._subscription_lock:
				arm_to_move = self.arm_to_move
			
			if arm_to_move == 'left':
				delta_pose = [
					self.forward_step,  # x - forward
					0.0,                # y
					0.0,                # z
					0.0,                # roll
					0.0,                # pitch
					0.0,                # yaw
					0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # right arm - no movement
				]
			else:  # right arm
				delta_pose = [
					0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # left arm - no movement
					self.forward_step,  # x - forward
					0.0,                # y
					0.0,                # z
					0.0,                # roll
					0.0,                # pitch
					0.0                 # yaw
				]
			
			# Execute movement
			robot_actions = [
				{
					"left_arm": {"action_data": delta_pose[:6], "control_type": "DELTA_POSE"},
					"right_arm": {"action_data": delta_pose[6:12], "control_type": "DELTA_POSE"},
				}
			]
			
			self.robot_controller.trajectory_tracking_control(
				infer_timestamp=int(time.time() * 1e9),
				robot_states=robot_states,
				robot_actions=robot_actions,
				robot_link="base_link",
				trajectory_reference_time=1.0,
			)
			
			if marker_size is not None:
				self.get_logger().debug(f'Step {step_count+1}: Moving forward, marker size: {marker_size}px')
			
			# Update arm states after movement
			arm_states = poll_state(self.robot_dds.arm_joint_states, length=14, name="arm_joint_states")
			robot_states["arm"] = arm_states
			
			step_count += 1
			time.sleep(1)
		
		if step_count >= max_steps:
			self.get_logger().warn(f'Reached max steps ({max_steps}) without losing marker')
			return False
		
		return True

	def compute_control_command(self):
		"""Compute delta pose control command to center the marker. Thread-safe."""
		with self._image_lock:
			if self.marker_center is None or self.image_center is None:
				return None
			
			# Calculate pixel offset
			offset_x = self.marker_center[0] - self.image_center[0]
			offset_y = self.marker_center[1] - self.image_center[1] + self.bias_y
			
			# Check if within tolerance
			offset_distance = np.sqrt(offset_x**2 + offset_y**2)
			if offset_distance < self.center_tolerance:
				return None  # Already centered
			
			# Convert pixel offset to robot delta pose
			delta_y = -offset_x * self.control_gain  # Left/Right
			delta_z = -offset_y * self.control_gain  # Up/Down
			
			# Limit maximum movement per step
			max_delta = 0.02  # 2cm max per step
			delta_y = np.clip(delta_y, -max_delta, max_delta)
			delta_z = np.clip(delta_z, -max_delta, max_delta)
			
			# Get arm to move value while holding lock
			arm_to_move = self.arm_to_move
			
			if arm_to_move == 'left':
				delta_pose = [
					0.0,      # x - no forward/backward movement
					delta_y,  # y - left/right
					delta_z,  # z - up/down
					0.0,      # roll
					0.0,      # pitch
					0.0,      # yaw
					0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # right arm - no movement
				]
			else:  # right arm
				delta_pose = [
					0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # left arm - no movement
					0.0,      # x - no forward/backward movement
					delta_y,  # y - left/right
					delta_z,  # z - up/down
					0.0,      # roll
					0.0,      # pitch
					0.0       # yaw
				]
			
			return delta_pose

	def execute_aruco_search(self):
		"""Main control loop for searching for item with Aruco marker."""
		
		# Poll initial robot states
		self.get_logger().info("Polling robot states...")
		head_states = poll_state(self.robot_dds.head_joint_states, length=2, name="head_joint_states")
		waist_states = poll_state(self.robot_dds.waist_joint_states, length=2, name="waist_joint_states")
		arm_states = poll_state(self.robot_dds.arm_joint_states, length=14, name="arm_joint_states")
		self.get_logger().info("Starting visual servoing control loop...")
		
		try:
			while rclpy.ok():
				# Wait for new image to be received
				if not self.wait_for_new_image(timeout=2.0):
					self.get_logger().warn('Timeout waiting for image, continuing...')
					continue
				
				# Compute control command based on latest image
				delta_pose = self.compute_control_command()
				
				if delta_pose is not None:
					
					# Check if next movement will exceed reach limits
					if self.arm_to_move == 'left':
						check_delta = [0, delta_pose[1], delta_pose[2]]
					else:
						check_delta = [0, delta_pose[7], delta_pose[8]]
					if not self.check_move_limits(check_delta, arm=self.arm_to_move):
						self.get_logger().warn("Arm movement limit exceeded, aborting visual servoing")
						return False

					# Create action dictionary
					action = {
						"observation_timestamp": int(time.time() * 1e9),
						"head_joint_states": head_states,
						"waist_joint_states": waist_states,
						"arm_joint_states": arm_states,
						"arm_cmd": [delta_pose]
					}
					
					# Define robot states and actions
					robot_states = {
						"head": action["head_joint_states"],
						"waist": action["waist_joint_states"],
						"arm": action["arm_joint_states"],
					}
					
					robot_actions = [
						{
							"left_arm": {"action_data": delta_pose[:6], "control_type": "DELTA_POSE"},
							"right_arm": {"action_data": delta_pose[6:12], "control_type": "DELTA_POSE"},
						}
					]
					
					# Execute trajectory control
					self.robot_controller.trajectory_tracking_control(
						infer_timestamp=action["observation_timestamp"],
						robot_states=robot_states,
						robot_actions=robot_actions,
						robot_link="base_link",
						trajectory_reference_time=1.0,
					)
					
					# Capture marker_id for logging with lock
					with self._image_lock:
						marker_id = self.marker_id
					
					if self.arm_to_move == 'left':
						self.get_logger().debug(f"Adjusting position - Marker ID: {marker_id}, "
											f"Delta: ({delta_pose[1]:.4f}, {delta_pose[2]:.4f})")
					else:
						self.get_logger().debug(f"Adjusting position - Marker ID: {marker_id}, "
										  f"Delta: ({delta_pose[7]:.4f}, {delta_pose[8]:.4f})")
					
					# Update arm states after movement
					arm_states = poll_state(self.robot_dds.arm_joint_states, length=14, name="arm_joint_states")

				else:
					# Check if marker is centered (safe read under lock)
					with self._image_lock:
						marker_center = self.marker_center
					
					if marker_center is not None:
						self.get_logger().debug("Marker centered! Saving new position and moving forward...")
						
						if self.level == 'top':
							if self.arm_to_move == 'left':
								self.left_top_joint_angles[0:7] = arm_states[0:7]
							else:
								self.right_top_joint_angles[7:] = arm_states[7:]
						else:
							if self.arm_to_move == 'left':
								self.left_bottom_joint_angles[0:7] = arm_states[0:7]
							else:
								self.right_bottom_joint_angles[7:] = arm_states[7:]

						# Marker is centered, now move forward until marker is lost
						time.sleep(0.5)  # Brief pause
						
						# Update robot states before forward movement
						robot_states = {
							"head": head_states,
							"waist": waist_states,
							"arm": arm_states,
						}
						
						success = self.move_forward_until_lost(robot_states, arm_states)
						
						if success:
							self.get_logger().info("Visual servoing complete: marker centered and moved forward until lost")
							if self.arm_to_move == 'left':
								self.left_gripper = 1.0  # Open gripper
							else:
								self.right_gripper = 1.0  # Open gripper
							self.robot_dds.move_gripper([self.left_gripper, self.right_gripper])
							time.sleep(2.0)
							# Move the robot back a little to avoid collision when resetting arms
							self.publish_wheel(linear_x=-0.1, angular_z=0.0, count=3, rate=5.0)
							# self.slam_move_relative(linear_x=-100.0, linear_y=0.0, angular_z=0.0)
							time.sleep(1.0)
							# Reset arm position after task
							self.reset_arm_position()
						else:
							self.get_logger().warn("Forward movement did not complete successfully")
							return False
						
						# Exit after completing the task
						return True
					else:
						# If no Aruco marker detected, search for marker
						with self._subscription_lock:
							arm_to_move = self.arm_to_move
						
						# Move arm sideways to search for marker (if limits allow)
						if arm_to_move == 'left':
							delta_pose = [0, -0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Move left arm right by 2cm
							check_delta = [0.0, -0.02, 0.0]						
						else:
							delta_pose = [0, 0, 0, 0, 0, 0, 0, 0.02, 0, 0, 0, 0]  # Move right arm left by 2cm
							check_delta = [0.0, 0.02, 0.0]
						if not self.check_move_limits(check_delta, arm=arm_to_move):
							self.get_logger().warn("Arm movement limit exceeded, aborting visual servoing")
							return False
						
						# Create action dictionary
						action = {
							"observation_timestamp": int(time.time() * 1e9),
							"head_joint_states": head_states,
							"waist_joint_states": waist_states,
							"arm_joint_states": arm_states,
							"arm_cmd": [delta_pose]
						}
						
						# Define robot states and actions
						robot_states = {
							"head": action["head_joint_states"],
							"waist": action["waist_joint_states"],
							"arm": action["arm_joint_states"],
						}

						robot_actions = [
							{
								"left_arm": {"action_data": delta_pose[:6], "control_type": "DELTA_POSE"},
								"right_arm": {"action_data": delta_pose[6:12], "control_type": "DELTA_POSE"},
							}
						]
						
						# Execute trajectory control
						self.robot_controller.trajectory_tracking_control(
							infer_timestamp=action["observation_timestamp"],
							robot_states=robot_states,
							robot_actions=robot_actions,
							robot_link="base_link",
							trajectory_reference_time=1.0,
						)
						self.get_logger().debug("No marker detected, moving {}".format("left arm right" if arm_to_move == 'left' else "right arm left"))
						
						# Update arm states after movement
						arm_states = poll_state(self.robot_dds.arm_joint_states, length=14, name="arm_joint_states")
				
				time.sleep(1.5)
				
		except Exception as exc:
			tb = traceback.format_exc()
			self.get_logger().error(f'Exception in visual servoing loop: {exc}\n{tb}')
			return False

	def destroy_node(self):
		"""Cleanup resources."""
		if self.video_writer is not None:
			self.video_writer.release()
			self.get_logger().info(f"Saved video to '{self.output_video}' ({self.frame_count} frames)")
		cv2.destroyAllWindows()
		super().destroy_node()

mcp = FastMCP("Retail Demo MCP Server")

# controller singleton shared by MCP handlers
_controller: G1Controller | None = None
_robot_thread: threading.Thread | None = None   

# Threading handlers ===============================================
def _robot_runner(item: str):
	"""Background job that actually runs the demo sequence."""
	try:
		_controller.reset_arm_position()
		time.sleep(1.0)
		_controller.get_logger().info(f"[ROBOT_RUNNER] Starting with item: {item}")
		
		# Determine which arm and movement file to use
		if item == "large chip":
			_controller.set_arm_to_move('left')
			_controller.set_level('top')
			_controller.set_item_to_grab('large chip')
			_controller.bias_y = -30  # pixels
			_controller.move_to_shelf_level(level=_controller.level, arm=_controller.arm_to_move)
			_controller.placemovement_file = "movement_records/arm_move_place_item_left.npy"
			_controller._large_chip -= 1
		elif item == "small chip":
			_controller.set_arm_to_move('right')
			_controller.set_level('top')
			_controller.set_item_to_grab('small chip')
			_controller.bias_y = -20  # pixels
			_controller.move_to_shelf_level(level=_controller.level, arm=_controller.arm_to_move)
			_controller.placemovement_file = "movement_records/arm_move_place_item_right.npy"
			_controller._small_chip -= 1
		elif item == "100 plus":
			_controller.set_arm_to_move('right')
			_controller.set_level('bottom')
			_controller.set_item_to_grab('100 plus')
			_controller.bias_y = 0  # pixels
			_controller.move_to_shelf_level(level=_controller.level, arm=_controller.arm_to_move)
			_controller.placemovement_file = "movement_records/arm_move_place_item_right.npy"
			_controller._100_plus -= 1
		elif item == "small bottle":
			_controller.set_arm_to_move('left')
			_controller.set_level('bottom')
			_controller.set_item_to_grab('small bottle')
			_controller.bias_y = 0  # pixels
			_controller.move_to_shelf_level(level=_controller.level, arm=_controller.arm_to_move)
			_controller.placemovement_file = "movement_records/arm_move_place_item_left.npy"
			_controller._small_bottle -= 1
		else:
			_controller.get_logger().error(f"[ROBOT_RUNNER] Unknown item: {item}")
			return
		
		_controller.get_logger().info(f"[ROBOT_RUNNER] Arm positioned, waiting before ArUco search...")
		time.sleep(3.0)

		_controller.get_logger().info(f"[ROBOT_RUNNER] Starting ArUco search...")
		found_item = _controller.execute_aruco_search()
		time.sleep(1)
		
		if found_item:
			_controller.get_logger().info("[ROBOT_RUNNER] Item found, navigating to counter...")
			# _controller.publish_wheel(linear_x=0.0, angular_z=-5.0, count=17, rate=5.0)
			# The counter is behind the robot. Let's turn it around first and then move towards it.
			_controller.slam_move_to_pose(x=-300.0, y=0.0, angle=0.0)  # Move forward to counter area
			time.sleep(1)
			_controller.slam_move_relative(linear_x=0.0, linear_y=0.0, angular_z=-90.0) # This is to make sure it turns clockwise
			_controller.slam_move_to_pose(x=-300.0, y=0.0, angle=180.0)  # Move forward to counter area
			_controller.robot_dds.move_waist([0, 6])  # Move waist to middle position
			time.sleep(3)
			
			_controller.get_logger().info("[ROBOT_RUNNER] Replaying place movement...")
			_controller.replay_movements(_controller.placemovement_file)
			time.sleep(3)
			
			_controller.get_logger().info("[ROBOT_RUNNER] Navigating back to shelf...")
			# _controller.publish_wheel(linear_x=0.0, angular_z=5.0, count=19, rate=5.0)
			# time.sleep(3)
			# _controller.publish_wheel(linear_x=0.1, angular_z=0.0, count=5, rate=5.0)
			_controller.slam_move_relative(linear_x=0.0, linear_y=0.0, angular_z=90.0) # This is to make sure it turns anti-clockwise
			_controller.slam_move_relative(linear_x=0.0, linear_y=0.0, angular_z=90.0)
			_controller.slam_move_to_pose(x=0.0, y=0.0, angle=0.0)  # Move forward to shelf

			# TODO: Remove the following, only for testing
			# _controller.robot_dds.move_gripper([0.0, 0.0])  # Close gripper to simulate placing item
			# time.sleep(5)

			_controller.get_logger().info("[ROBOT_RUNNER] Task completed successfully")
		else:
			_controller.get_logger().warn("[ROBOT_RUNNER] Item not found during search")
			
	except Exception as exc:
		tb = traceback.format_exc()
		_controller.get_logger().error(f"[ROBOT_RUNNER] Exception occurred: {exc}\n{tb}")
		import sys
		sys.stderr.write(f"[ROBOT_RUNNER] Exception: {exc}\n{tb}\n")
		sys.stderr.flush()
		# log to MCP (use mcp.log if available) and swallow to avoid crashing the thread
		try:
			mcp.log.error(f"Demo runner exception: {exc}")
		except Exception:
			print("Demo runner exception:", exc)
	finally:
		# optional: keep controller alive (do not shutdown) or shutdown automatically:
		# ctrl.shutdown()
		pass

def _robot_reset():
	"""Reset the robot controller and stop any ongoing demo."""
	try:
		_controller.reset_arm_position()
		_controller.get_logger().info("[ROBOT_RESET] Robot has been reset.")
	except Exception as exc:
		tb = traceback.format_exc()
		_controller.get_logger().error(f"[ROBOT_RESET] Exception occurred: {exc}\n{tb}")
		import sys
		sys.stderr.write(f"[ROBOT_RESET] Exception: {exc}\n{tb}\n")
		sys.stderr.flush()
		# log to MCP (use mcp.log if available) and swallow to avoid crashing the thread
		try:
			mcp.log.error(f"Demo runner exception: {exc}")
		except Exception:
			print("Demo runner exception:", exc)
	finally:
		# optional: keep controller alive (do not shutdown) or shutdown automatically:
		# ctrl.shutdown()
		pass

# MCP tool handler
@mcp.tool
def grab_items(item: str, sessionId: str, action: str, chatInput: str, toolCallId: str) -> str:
	"""
	Commands the robot to grab an item specified by user message.
	The item should be one of: {"large chip", "small chip", "100 plus", "small bottle"}.
	"""
	global _robot_thread, _controller
	# Prevent starting multiple demo threads
	if _robot_thread and _robot_thread.is_alive():
		return "Reply to the user: The robot is already performing a task. Please try again later."

	# Check inventory count for item before grabbing
	if item == "large chip" and _controller._large_chip <= 0:
		return "Reply to the user: Sorry, the large chip item is out of stock."
	elif item == "small chip" and _controller._small_chip <= 0:
		return "Reply to the user: Sorry, the small chip item is out of stock."
	elif item == "100 plus" and _controller._100_plus <= 0:
		return "Reply to the user: Sorry, the 100 plus item is out of stock."
	elif item == "small bottle" and _controller._small_bottle <= 0:
		return "Reply to the user: Sorry, the small bottle item is out of stock."
	
	_robot_thread = threading.Thread(
		target=_robot_runner,
		args=(item,),
		daemon=True
	)
	_robot_thread.start()
	
	return f"Reply to the user: The robot has started the grab sequence for the item: {item}."

@mcp.tool
def reset_robot(sessionId: str, action: str, chatInput: str, toolCallId: str) -> str:
	"""Resets the robot to its initial or home position."""
	global _robot_thread, _controller
	# Prevent starting multiple demo threads
	if _robot_thread and _robot_thread.is_alive():
		return "Reply to the user: The robot is already performing a task. Please try again later."
	
	_robot_thread = threading.Thread(
		target=_robot_reset,
		daemon=True
	)
	_robot_thread.start()
	return "Reply to the user: The robot is being reset to its initial position."

@mcp.tool
def reset_inventory(sessionId: str, action: str, chatInput: str, toolCallId: str) -> str:
	"""Resets the inventory counts for all items. Use this tool when items have been restocked."""
	global _controller
	_controller._large_chip = INITIAL_LARGE_CHIP_COUNT
	_controller._small_chip = INITIAL_SMALL_CHIP_COUNT
	_controller._small_bottle = INITIAL_SMALL_BOTTLE_COUNT
	_controller._100_plus = INITIAL_100_PLUS_COUNT
	_controller.left_top_joint_angles = LEFT_TOP_JOINT_ANGLES
	_controller.left_bottom_joint_angles = LEFT_BOTTOM_JOINT_ANGLES
	_controller.right_top_joint_angles = RIGHT_TOP_JOINT_ANGLES
	_controller.right_bottom_joint_angles = RIGHT_BOTTOM_JOINT_ANGLES
	return "Reply to the user: The inventory counts have been reset."

def grab_items_test(item: str) -> str:
	"""Commands the robot to grab an item."""
	global _robot_thread, _controller
	# Prevent starting multiple demo threads
	if _robot_thread and _robot_thread.is_alive():
		return "Reply to the user: The robot is already performing a task. Please try again later."

	_robot_thread = threading.Thread(
		target=_robot_runner,
		args=(item,),
		daemon=True
	)
	_robot_thread.start()
	return f"Reply to the user: The robot has started the grab sequence for the item: {item}."

def run_ros_executor():
	"""Keep ROS2 executor spinning in a background thread to process callbacks."""
	try:
		rclpy.spin(_controller)
	except Exception as e:
		print(f"[ROS_EXECUTOR] Error in executor: {e}")
		traceback.print_exc()

if __name__ == "__main__":
	import sys
	rclpy.init()
	
	_controller = G1Controller(
		aruco_dict_id=cv2.aruco.DICT_4X4_50,
		output_video=None,
		center_tolerance=15,
		control_gain=0.0001,
		forward_step=0.02,
		display_window=False
	)
	
	# Start ROS2 executor in a background thread to process callbacks
	executor_thread = threading.Thread(target=run_ros_executor, daemon=True)
	executor_thread.start()
	
	# Give executor time to start
	time.sleep(0.5)
	
	""" Manual test of the robot demo without MCP interaction """
	# # Now start the robot demo in another thread
	# grab_items_test("100 plus")
	
	# # Wait for robot thread to complete
	# if _robot_thread:
	# 	_robot_thread.join(timeout=300)  # Wait up to 5 minutes
	
	# # Cleanup
	# _controller.get_logger().info("Shutting down...")
	# _controller.destroy_node()
	# rclpy.shutdown()

	""" MCP server mode """
	try:
		mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
	except KeyboardInterrupt:
		_controller.get_logger().info("KeyboardInterrupt received, shutting down...")
	finally:
		_controller.get_logger().info("Shutting down...")
		_controller.destroy_node()
		rclpy.shutdown()
