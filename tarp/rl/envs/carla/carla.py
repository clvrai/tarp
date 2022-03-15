import os, sys
import random
import gym
import numpy as np
import copy
import json
import cv2
import carla
import math
from gym.utils import seeding

try:
    import queue
except ImportError:
    import Queue as queue

from tarp.utils.general_utils import ParamDict, AttrDict, map_recursive
from tarp.utils.pytorch_utils import ar2ten, ten2ar
from tarp.rl.components.environment import GymEnv
from tarp.rl.envs.carla.weather import Weather
from agents.navigation.agent import Agent, AgentState
from agents.tools.misc import is_within_distance_ahead, compute_magnitude_angle
from tarp.rl.envs.carla.planner import compute_route_waypoints
from tarp.rl.envs.carla.utils import distance_to_line, vector, angle_diff, labels_to_cityscapes_palette, labels_to_array

DAVIS17_VIDEOS = [
    'bear', 'bmx-bumps', 'boat', 'boxing-fisheye', 'breakdance-flare', 'bus',
    'car-turn', 'cat-girl', 'classic-car', 'color-run', 'crossing',
    'dance-jump', 'dancing', 'disc-jockey', 'dog-agility', 'dog-gooses',
    'dogs-scale', 'drift-turn', 'drone', 'elephant', 'flamingo', 'hike',
    'hockey', 'horsejump-low', 'kid-football', 'kite-walk', 'koala',
    'lady-running', 'lindy-hop', 'longboard', 'lucia', 'mallard-fly',
    'mallard-water', 'miami-surf', 'motocross-bumps', 'motorbike', 'night-race',
    'paragliding', 'planes-water', 'rallye', 'rhino', 'rollerblade',
    'schoolgirls', 'scooter-board', 'scooter-gray', 'sheep', 'skate-park',
    'snowboard', 'soccerball', 'stroller', 'stunt', 'surf', 'swing', 'tennis',
    'tractor-sand', 'train', 'tuk-tuk', 'upside-down', 'varanus-cage', 'walking'
]


class CarlaEnv(gym.Env):
    def __init__(self, **kwargs):
        self.actor_list = []

    def _default_hparams(self):
        default_dict = ParamDict({
            'name': 'carla-nav-v0',
            'town': "Town05",
            'task_json': './tarp/data/carla/json_data/straight.json',
            'fps': 20,
            'num_vehicles': 300,
            'num_pedestrians': 200,
            'resolution': 64,
            'port': 2000,
            'tm_port': 8000,
            'distance_threshold_proximity': 7.5,
            'num_cameras': 1,
            'fov': 60,
            'changing_weather_speed': 0.3,
            'traffic_light_threshold': 5.0,
            'frame_skip': 1,
            'max_distance': 3,
            'specified_classes': None,
            'video_dir': os.path.join(os.environ['DATA_DIR'], './dmcontrol/background/DAVIS/JPEGImages/480p/'),
            'episodic_weather': False,
        })
        return default_dict

    def set_config(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self.seed(self._hp.seed)
        self.setup()

    def setup(self):
        self.np_random, seed = seeding.np_random(self._hp.seed)
        self.video_paths = [os.path.join(self._hp.video_dir, subdir) for subdir in DAVIS17_VIDEOS]

        self.client = carla.Client("127.0.0.1", self._hp.port)
        self.client.set_timeout(20.0)

        self.world = self.client.load_world(self._hp.town)
        self.map = self.world.get_map()

        self.world.tick()
        actor_list = self.world.get_actors()
        for vehicle in actor_list.filter("*vehicle*"):
            # if vehicle.id != self.vehicle.id:
            print("Warning: removing old vehicle")
            vehicle.destroy()
        for sensor in actor_list.filter("*sensor*"):
            print("Warning: removing old sensor")
            sensor.destroy()

        self.lights_list = actor_list.filter("*traffic_light*")

        # read task config
        with open(self._hp.task_json, 'r') as f:
            self.task = json.load(f)

        self.vehicle = None
        self.vehicle_start_pose = None
        self.vehicles_list = []
        self.pedestrian_list = []
        self.pedestrian_controller_list = []
        self.vehicles = None
        start_t, goal_t = self.sample_start_goal_pose()
        self.start_transform = start_t
        self.goal_transform = goal_t
        self.reset_vehicle()
        self.actor_list.append(self.vehicle)

        blueprint_library = self.world.get_blueprint_library()
        bp = blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self._hp.resolution))
        bp.set_attribute('image_size_y', str(self._hp.resolution))
        bp.set_attribute('fov', str(self._hp.fov))
        location = carla.Location(x=1.6, z=1.7)
        self.camera_rl = self.world.spawn_actor(bp, carla.Transform(location, carla.Rotation(yaw=0.0)), attach_to=self.vehicle)
        if self._hp.num_cameras > 1:
            self.camera_rl_left = self.world.spawn_actor(bp, carla.Transform(location, carla.Rotation(yaw=-float(self._hp.fov))), attach_to=self.vehicle)
            self.camera_rl_right = self.world.spawn_actor(bp, carla.Transform(location, carla.Rotation(yaw=float(self._hp.fov))), attach_to=self.vehicle)

        bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', str(64))
        bp.set_attribute('image_size_y', str(64))
        bp.set_attribute('fov', str(self._hp.fov))
        location = carla.Location(x=1.6, z=1.7)
        self.segmentation = self.world.spawn_actor(bp, carla.Transform(location, carla.Rotation(yaw=0.0)), attach_to=self.vehicle)

        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        self.actor_list.append(self.collision_sensor)
        self._collision_intensities_during_last_time_step = []

        if self._hp.num_cameras == 1:
            self.sync_mode = CarlaSyncMode(self.world, self.camera_rl, self.segmentation, fps=self._hp.fps)
        else:
            self.sync_mode = CarlaSyncMode(self.world, self.camera_rl, self.camera_rl_left, self.camera_rl_right, self.segmentation, fps=self._hp.fps)

        self.weather = Weather(self.world, self._hp.changing_weather_speed)
        self.world.tick()
        self.reset()

    def reset_vehicle(self):
        if self.vehicle is None:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_blueprint = blueprint_library.find('vehicle.audi.a2')
            self.vehicle = self.world.spawn_actor(vehicle_blueprint, self.start_transform)
        else:
            vehicle_control = carla.VehicleControl(
                throttle=float(0.0),
                steer=float(0.0),
                brake=float(0.0),
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False
            )
            self.vehicle.apply_control(vehicle_control)
            self.vehicle.set_simulate_physics(False) # Reset the car's physics
            self.vehicle.set_transform(self.start_transform)
            self.vehicle.set_simulate_physics(True)

    def reset_other_vehicles(self):
        # clear out old vehicles
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        self.world.tick()
        self.vehicles_list = []

        traffic_manager = self.client.get_trafficmanager(self._hp.tm_port)  # 8000? which port?
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        traffic_manager.set_synchronous_mode(True)
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        init_transforms = self.world.get_map().get_spawn_points()
        init_transforms = self.np_random.choice(init_transforms, self._hp.num_vehicles)

        batch = []
        for transform in init_transforms:
            transform.location.z += 0.1
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, self._hp.tm_port)))

        for response in self.client.apply_batch_sync(batch, False):
            self.vehicles_list.append(response.actor_id)

        for response in self.client.apply_batch_sync(batch):
            if response.error:
                pass
            else:
                self.vehicles_list.append(response.actor_id)

        traffic_manager.global_percentage_speed_difference(30.0)


    def reset_pedestrians(self):
        for controller in self.world.get_actors(self.pedestrian_controller_list):
            controller.stop()

        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.pedestrian_controller_list])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.pedestrian_list])
        self.world.tick()
        self.pedestrian_list = []
        self.pedestrian_controller_list = []

        blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")

        batch = []
        for _ in range(self._hp.num_pedestrians):
            blueprint = random.choice(blueprints)
            blueprint.set_attribute("is_invincible", "true")
            transform = carla.Transform()  # pylint: disable=no-member
            transform.location = self.world.get_random_location_from_navigation()
            if transform.location is None:
                continue
            batch.append(carla.command.SpawnActor(blueprint, transform))

        for response in self.client.apply_batch_sync(batch):
            if response.error:
                pass
            else:
                self.pedestrian_list.append(response.actor_id)

        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for idx in self.pedestrian_list:
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), idx))


        for response in self.client.apply_batch_sync(batch):
            if response.error:
                pass
            else:
                self.pedestrian_controller_list.append(response.actor_id)

        self.world.tick()
        actors = self.world.get_actors(self.pedestrian_controller_list)
        for controller in actors:
            controller.start()

    def step(self, action):
        rewards = []
        for _ in range(self._hp.frame_skip):
            obs, reward, done, info = self._simulator_step(action)
            rewards.append(reward)
            if done:
                break
        return obs/255., np.mean(rewards), np.array(done), info

    def _simulator_step(self, action):
        if action is not None:
            steer = float(action[0])
            throttle_brake = float(action[1])
            if throttle_brake >= 0.0:
                throttle = throttle_brake
                brake = 0.0
            else:
                throttle = 0.0
                brake = -throttle_brake
        else:
            throttle, steer, brake = 0.0, 0.0, 0.0

        vehicle_control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        )

        self.vehicle.apply_control(vehicle_control)

        if self._hp.num_cameras == 1:
            snapshot, image_rl, mask_rl = self.sync_mode.tick(timeout=2.0)
        else:
            snapshot, image_rl, image_rl_left, image_rl_right, mask_rl = self.sync_mode.tick(timeout=2.0)
        done = False

        collision_intensities_during_last_time_step = sum(self._collision_intensities_during_last_time_step)
        self._collision_intensities_during_last_time_step.clear()  # clear it ready for next time step
        assert collision_intensities_during_last_time_step >= 0.

        colliding = float(collision_intensities_during_last_time_step > 0.)
        if colliding:
            self.num_collision += 1

        reward = self._compute_reward()
        reward -= 0.0001 * int(collision_intensities_during_last_time_step)

        if int(collision_intensities_during_last_time_step) > 1e6:
            done = True
            print('Too large collision intensity')

        info = AttrDict()
        info['collision_intensity'] = collision_intensities_during_last_time_step
        info['steer'] = steer
        info['brake'] = brake

        if not self._hp.episodic_weather:
            self.weather.tick()

        mask = labels_to_cityscapes_palette(mask_rl, self._hp.specified_classes)

        rgbs = []
        if self._hp.num_cameras == 1:
            imgs = [image_rl]
        else:
            imgs = [image_rl_left, image_rl, image_rl_right]
        for im in imgs:
            bgra = np.array(im.raw_data).reshape(im.height, im.width, 4)
            bgr = bgra[:, :, :3]
            rgb = np.flip(bgr, axis=2)
            rgbs.append(rgb)
        obs = np.concatenate(rgbs, axis=1)
        self.render_image = obs
        self.mask = mask

        # modify later
        self._distance_to_go = self._compute_distance()
        if self.current_waypoint_index >= len(self.route_waypoints)-1 or self._distance_to_go < self._hp.distance_threshold_proximity:
            done = True
            self._success = True
            reward += 10.

        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        return self.render_image

    def render_mask(self):
        return self.mask.astype(np.uint8)

    def _compute_reward(self):
        waypoint_index = self.current_waypoint_index
        transform = self.vehicle.get_transform()
        for _ in range(len(self.route_waypoints)):
            # Check if we passed the next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                         vector(transform.location - wp.transform.location)[:2])
            if dot > 0.0: # Did we pass the waypoint?
                waypoint_index += 1 # Go to next waypoint
            else:
                break
        self.current_waypoint_index = waypoint_index
        # Check for route completion
        if self.current_waypoint_index < len(self.route_waypoints)-1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[self.current_waypoint_index % len(self.route_waypoints)]
        self.routes_completed = self.num_routes_completed + (self.current_waypoint_index + 1) / len(self.route_waypoints)

        # Calculate deviation from center of the lane
        self.distance_from_center = distance_to_line(vector(self.current_waypoint.transform.location),
                                                     vector(self.next_waypoint.transform.location),
                                                     vector(transform.location))

        min_speed = 15.0 # km/h
        max_speed = 30.0 # km/h
        target_speed = 25.0

        fwd = vector(self.vehicle.get_velocity())
        wp_fwd = vector(self.current_waypoint.transform.rotation.get_forward_vector())
        angle = angle_diff(fwd, wp_fwd)
        vel = self.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
            speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
        elif speed_kmh > target_speed:                # When speed is in [target_speed, inf]
                                                      # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
            speed_reward = 1.0 - (speed_kmh-target_speed) / (max_speed-target_speed)
        else:                                         # Otherwise
            speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]

        # Interpolated from 1 when centered to 0 when 3 m from center
        centering_factor = max(1.0 - self.distance_from_center / self._hp.max_distance, 0.0)

        # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
        angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)

        # Final reward
        reward = speed_reward * centering_factor * angle_factor

        # next_goal_waypoint = self.next_waypoint.next(0.1)
        # if len(next_goal_waypoint) == 0:
        #     vel_s = 0
        # else:
        #     vel = self.vehicle.get_velocity()
        #     vehicle_velocity_xy = np.array([vel.x, vel.y])
        #     location_ahead = next_goal_waypoint[0].transform.location
        #     next_location = self.next_waypoint.transform.location
        #     road_vector = np.array([location_ahead.x, location_ahead.y]) - np.array([next_location.x, next_location.y])
        #     unit_vector = np.array(road_vector) / np.linalg.norm(road_vector)
        #     vel_s = np.dot(vehicle_velocity_xy, unit_vector)
        # reward = vel_s * 0.05
        return reward

    def _compute_distance(self):
        vehicle_location = self.vehicle.get_location()
        target_location = self.goal_transform.location
        dist = np.linalg.norm(np.array([vehicle_location.x, vehicle_location.y]) - np.array([target_location.x, target_location.y]))
        return dist

    def _on_collision(self, event):
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self._collision_intensities_during_last_time_step.append(intensity)

    def sample_start_goal_pose(self):
        idx = self.np_random.randint(len(self.task))
        print(idx)
        sample = self.task[idx]
        start_z = sample['start']['z'] + 0.5 if 'z' in sample['start'].keys() else 0.1
        start = carla.Location(x=sample['start']['x'], y=sample['start']['y'],z=start_z)
        print(start)
        start_rotation = carla.Rotation(yaw=sample['start']['yaw'])
        start_t = carla.Transform(start, start_rotation)
        goal_z = sample['destination']['z'] + 0.1 if 'z' in sample['destination'].keys() else 0.1
        destination = carla.Location(x=sample['destination']['x'], y=sample['destination']['y'], z=goal_z)
        print(destination)
        destination_rotation = carla.Rotation(yaw=sample['destination']['yaw'])
        destination_t = carla.Transform(destination, destination_rotation)
        return start_t, destination_t

    def reset(self):
        self._step = 0
        colliding = True
        falling = True
        if self._hp.episodic_weather:
            self.weather.tick()
        while falling:
            start_t, goal_t = self.sample_start_goal_pose()
            self.start_transform = start_t
            self.goal_transform = goal_t
            self.start_wp = self.map.get_waypoint(start_t.location)
            self.goal_wp = self.map.get_waypoint(goal_t.location)
            self.reset_vehicle()
            self.world.tick()
            self.reset_other_vehicles()
            self.world.tick()
            self.reset_pedestrians()
            self.world.tick()
            self.num_collision = 0
            self.num_invasion = 0
            self._distance_to_go = np.inf
            self._success = False

            self.route_waypoints = compute_route_waypoints(self.map, self.start_wp, self.goal_wp, resolution=2.0)
            self.current_waypoint_index = 0
            self.current_waypoint = self.route_waypoints[0][0]
            self.num_routes_completed = 0

            for _ in range(10):
                obs, _, _, _ = self.step(None)

            vel = self.vehicle.get_velocity()
            angular_vel = self.vehicle.get_angular_velocity()
            if angular_vel.x < 1. and angular_vel.y < 1. and angular_vel.z < 1. and vel.z < 1.:
                falling = False
        return obs

    def get_episode_info(self):
        info = AttrDict()
        info['distance_to_go'] = self._distance_to_go
        info['num_collision'] = self.num_collision
        info['success'] = float(self._success)
        return info

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context
        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)
    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

        self.start()

    def start(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

class CarlaStateEnv(CarlaEnv):
    def _simulator_step(self, action):
        if action is not None:
            steer = float(action[0])
            throttle_brake = float(action[1])
            if throttle_brake >= 0.0:
                throttle = throttle_brake
                brake = 0.0
            else:
                throttle = 0.0
                brake = -throttle_brake

            vehicle_control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False
            )
            self.vehicle.apply_control(vehicle_control)
        else:
            throttle, steer, brake = 0.0, 0.0, 0.0

        # for waypoint, _ in self.route_waypoints:
        #     self.world.debug.draw_point(waypoint.transform.location, color=carla.Color(0, 255, 0), life_time=5.0)

        if self._hp.num_cameras == 1:
            snapshot, image_rl, mask_rl = self.sync_mode.tick(timeout=2.0)
        else:
            snapshot, image_rl, image_rl_left, image_rl_right, mask_rl = self.sync_mode.tick(timeout=2.0)

        collision_intensities_during_last_time_step = sum(self._collision_intensities_during_last_time_step)
        self._collision_intensities_during_last_time_step.clear()  # clear it ready for next time step
        assert collision_intensities_during_last_time_step >= 0.
        done = False

        colliding = float(collision_intensities_during_last_time_step > 0.)
        if colliding:
            self.num_collision += 1

        reward = self._compute_reward()
        reward -= 0.0001 * int(collision_intensities_during_last_time_step)

        info = AttrDict()
        info['collision_intensity'] = collision_intensities_during_last_time_step
        info['steer'] = steer
        info['brake'] = brake

        if not self._hp.episodic_weather:
            self.weather.tick()

        mask = labels_to_cityscapes_palette(mask_rl, self._hp.specified_classes)
        mask_ids = labels_to_array(mask_rl)
        rgbs = []
        if self._hp.num_cameras == 1:
            imgs = [image_rl]
        else:
            imgs = [image_rl_left, image_rl, image_rl_right]
        for im in imgs:
            bgra = np.array(im.raw_data).reshape(im.height, im.width, 4)
            bgr = bgra[:, :, :3]
            rgb = np.flip(bgr, axis=2)
            rgbs.append(rgb)
        obs = np.concatenate(rgbs, axis=1)

        self.render_image = obs
        self.mask = mask
        self.mask_ids = mask_ids
        # modify later
        self._distance_to_go = self._compute_distance()
        if self.current_waypoint_index >= len(self.route_waypoints)-1 or self._distance_to_go < self._hp.distance_threshold_proximity:
            done = True
            self._success =True
            reward += 10.

        vehicle_location = self.vehicle.get_location()
        target_location = self.goal_transform.location
        waypoint_location = self.current_waypoint.transform.location
        if self.current_waypoint_index < len(self.route_waypoints)-1:
            next_waypoint_location = self.route_waypoints[self.current_waypoint_index+1][0].transform.location
        else:
            next_waypoint_location = target_location
        # add velocity
        velocity = self.vehicle.get_velocity()
        acc = self.vehicle.get_acceleration()
        obs = np.array([np.linalg.norm(np.array([vehicle_location.x, vehicle_location.y]) - np.array([target_location.x, target_location.y])), vehicle_location.x, vehicle_location.y,
                        waypoint_location.x, waypoint_location.y, next_waypoint_location.x, next_waypoint_location.y,
                        velocity.x, velocity.y, acc.x, acc.y])

        return obs, reward, done, info


