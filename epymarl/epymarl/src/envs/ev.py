import os
import sys
import sumolib
import traci
import numpy as np
from gymnasium import spaces
import logging

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("'SUMO_HOME' environment variable")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

ALPHA = 0.001
BETA = 0.002
GAMMA = 0.1


class ElectricVehicle:
    # logging.basicConfig(level=logging.DEBUG)#输出DeBUG级别以上的日志


    def __init__(self, env, vehicle_id, net, connection,):
        self.id = vehicle_id
        self.net = net
        self.env = env
        self.status = -1
        self.connection = connection       
        self._is_truncated = False
        self._is_terminated = False
        #记录第一次到充电站,给予奖励。因此给予奖励后 _is_stopped = False
        self._is_stopped = False
        #记录是否在去往一个充电站的路上重新规划了路径
        self._is_rerouted = False
        "_locked_cs 判断是否"
        "locked count 记录是否对动作进行了蒙蔽.因为在执行动作后,先计算奖励,再进行选择可用动作，选完可用动作后,locked_cs 和locked_count都会被清除"
        "locked_count =1 表示还未进行动作蒙蔽"
        self._locked_cs = None
        self._locked_count =0
        self._missed_cs = None
        self._missed_count = 0
        #记录无法到达充电站
        "如果车选择的充电站和车是在同一车道上,且车已经超过了cs的末端,则车无法到达充电站"
        self._failure_set = False
        """
        _route_before,记录上一个选择的充电站id,
        _route_after,记录重新选择的充电站id,比较选择的好坏给予奖励或者惩罚
        """
        self._route_before = None
        self._route_after = None
        
        
        self.battery_ratio = self.get_battery_ratio()
        self.charge_station_IDs = self.env.cs_ids
        self.num_stations = self.env.num_stations
        self.observation_dim = {
            'battery_ratio': 1,  # 电池电量比例
            'charging_station_info': self.num_stations * 2,  # 每个充电站的距离和拥挤程度
        }
        total_obs_dim = sum(self.observation_dim.values())
        self.observation_space = spaces.Box(
            low=np.zeros(total_obs_dim, dtype=np.float32),
            high=np.ones(total_obs_dim, dtype=np.float32)
        )
        self.action_space = spaces.Discrete(len(self.charge_station_IDs) + 1)#n-1个有效动作，第n个无效动作
        
        self.total_waiting_time = 0.0
        self.current_waiting_time = 0.0
        self.total_reward = 0.0
        self.current_reward = 0.0
        self.current_step = 0
        self.total_steps = 0
        self.flag = True
        self.lastlocation = None
        self.current_charging_station = None  

    def get_observation(self, is_reset:bool):
        logger.debug(f"is_reset: {is_reset}")
        current_capacity = float(self.connection.vehicle.getParameter(
            self.id, "device.battery.chargeLevel"))
        max_capacity = float(self.connection.vehicle.getParameter(
            self.id, "device.battery.capacity"))
        battery_ratio = current_capacity / max_capacity
        if self._is_terminated or self._is_truncated:
            return None
        if is_reset:
            self.flag =False
            for departure in self.env.EV.departures:
                if departure[0] == self.id:
                    self.current_edge_id = departure[1]
                    break
        else:
            self.current_edge_id = self.connection.vehicle.getRoadID(self.id)
            if not self.current_edge_id:
                self.current_edge_id = self.lastlocation
                logger.debug(f"Vehicle {self.id} invalid edge ID, using last location: {self.current_edge_id}")
        
        self.lastlocation = self.current_edge_id
        
        # 分别存储距离和拥挤程度
        distances = []
        congestions = []
        if is_reset:
            vehPos = 0
        elif self.connection.vehicle.getRoadID(self.id).startswith(":"):
            #如果在内部路里面
            last_edge = self.connection.vehicle.getRoute(self.id)[self.connection.vehicle.getRouteIndex(self.id)]#上一个真实路径
            vehPos = self.connection.lane.getLength(last_edge + "_0")
            self.current_edge_id = last_edge
        else:
            #正常情况
            vehPos = self.connection.vehicle.getLanePosition(self.id)
        
        for cs_id in self.charge_station_IDs:
            cs_lane_id = self.connection.chargingstation.getLaneID(cs_id)
            cs_edge_id = self.connection.lane.getEdgeID(cs_lane_id)
            cs_end_pos = self.connection.chargingstation.getEndPos(cs_id)
                       
            # 计算到充电站的距离
            try:
                dist = self.connection.simulation.getDistanceRoad(
                    self.current_edge_id, vehPos, cs_edge_id, cs_end_pos, isDriving=self.flag)
            except traci.exceptions.TraCIException as e:
                dist = self.connection.simulation.getDistanceRoad(
                    self.current_edge_id, 0, cs_edge_id, cs_end_pos, isDriving=self.flag)
            distances.append(dist)
            
            incoming_lanes = self.net.getLane(cs_lane_id).getIncoming()
            incoming_lanes_ids = set([lane.getID() for lane in incoming_lanes])
            lanes = incoming_lanes_ids.copy()
            lanes.add(cs_lane_id)
            
            lane_density_sum = 0
            for lane in lanes:
                lane_density = self.get_lane_density(lane)
                lane_density_sum += lane_density
            congestion = lane_density_sum / len(lanes)
            congestions.append(congestion)
        
        # 归一化
        max_dist = max(distances + [1.0])
        max_congestion = max(congestions + [1.0])
        
        # 构建观察值：[battery_ratio, dist1, cong1, dist2, cong2, ...]
        observation = [battery_ratio]  # 首先添加电池电量比例
        for i in range(len(self.charge_station_IDs)):
            observation.extend([
                distances[i] / max_dist,  # 归一化距离
                congestions[i] / max_congestion  # 归一化拥挤程度
            ])
        
        observation = np.array(observation, dtype=np.float32)
        return observation
    
    
    # def get_rewards(self):
    #     """
    #     奖励计算情况:选择充电站、truncated、terminated、after truncated、 after terminated、keep charging
    #     选择充电站:cs_id = status, is_stopped = True给予奖励
    #     truncated:cs_id = status
    #     terminated: cs_id = status-n_cs
    #     after truncated: cs_id = status- 2* n_cs
    #     after terminated: cs_id = status -3* n_cs
    #     keep charging : cs_id = status - n_cs    
    #     is_stopped , is_rerouted(route_before, route_after),is_terminated, is_truncated分别给予奖励或者惩罚,给予完后都设置为False
    #     第一次选择充电站时，根据选择充电站的拥挤程度给予奖励(拥挤程度进行归一化,按名次给奖励)
        
    #     """
    #     "通过status判断是否完成或中止任务,通过_is_terminated和_is_truncated判断是否给予奖励"
    #     reward = 0.0
        
    #     current_capacity = float(self.connection.vehicle.getParameter(
    #         self.id, "device.battery.chargeLevel"))
    #     max_capacity = float(self.connection.vehicle.getParameter(
    #         self.id, "device.battery.capacity"))
    #     battery_ratio = current_capacity / max_capacity
    #     if (self.status >= 2*self.num_stations and self._is_terminated) or (self.status >= 2*self.num_stations and self._is_truncated) :
    #         #说明已经完成任务或者任务中止,且未给予过奖励
    #         if self._is_terminated:
    #             self._is_terminated = False
    #             return 930.0
    #         else:
    #             self._is_truncated = False
    #             return -310.0
    #     if self.status > 2* self.num_stations and not self._is_terminated and not self._is_truncated:
    #         #说明已完成任务，并给予过奖励
    #         return 0.0
    #     if (self.num_stations <= self.status < 2* self.num_stations) and self._is_stopped:
    #         #说明已经到达充电站充电，并未给予过奖励
    #         if battery_ratio < 0.3:
    #             reward += 10.0
    #         else:
    #             reward += 5.0
    #         self._is_stopped = False
    #         return reward
    #     if (self.num_stations <= self.status < 2* self.num_stations) and (not self._is_stopped):
    #         return 2.0
    #     if battery_ratio < 0.2:
    #         reward -= 5* (1 -battery_ratio)
    #     if not self.env.is_reset:
    #         current_waiting_time = self.connection.vehicle.getWaitingTime(self.id)
    #         if current_waiting_time >= 0.6:
    #             reward -= 60.0
    #         elif current_waiting_time >= 0.5:
    #             reward -= 30.0
    #         else:
    #             reward += 5.0
            
        
    #     if self.status < self.num_stations:
    #         "说明在前往充电站的路上,1.改变路径的,根据选择的好坏给予奖励或惩罚2.未改变路径的,根据现状给予奖励或者惩罚,第一次做出选择不会有route_before"
    #         "错过充电站,但又重新选择了该充电站作为目的地"
    #         if self._locked_cs is not None:
    #             reward += 1.0
    #         if self._missed_cs is not None:
    #             reward -= 1.0
    #         if self._failure_set:
    #             "虽然想要及时的将目的地改为当前道路上的充电站,但是因为sumo中无法停止而失败"
    #             reward -= 0.5
    #             self._failure_set = False
    #         if self._is_rerouted and (not self.env.is_reset):
    #             "说明重新规划了路径,计算之前路径的相对拥挤程度和新选择路径的拥挤程度"
    #             before_cs_id = self.env.EV.charging_stations_dict[self._route_before]
    #             cs_lane_before = self.connection.chargingstation.getLaneID(before_cs_id)
    #             now_cs_id = self.env.EV.charging_stations_dict[self._route_after]
    #             cs_lane_now = self.connection.chargingstation.getLaneID(now_cs_id)
                
    #             before_vehicle_number = self.connection.lane.getLastStepVehicleNumber(cs_lane_before)
    #             before_vehicle_density= self.get_lane_density(cs_lane_before)
    #             after_vehicle_number = self.connection.lane.getLastStepVehicleNumber(cs_lane_now)
    #             after_vehicle_density = self.get_lane_density(cs_lane_now)
    #             total_vehicle_number = before_vehicle_number + after_vehicle_number
    #             if total_vehicle_number == 0:
    #                 total_vehicle_number = 1e-5
    #             relative_before_vehnumber = before_vehicle_number / (total_vehicle_number)
    #             relative_after_vehnumber = after_vehicle_number / (total_vehicle_number)
    #             relative_before_score = 0.55 * relative_before_vehnumber + 0.45 * before_vehicle_density 
    #             relative_after_score = 0.55 * relative_after_vehnumber + 0.45 * after_vehicle_density  
    #             if relative_before_score < relative_after_score:
    #                 #选择不正确
    #                 self._is_rerouted = False
    #                 self._route_before = None
    #                 self._route_after = None
    #                 reward -= 1.0
    #                 return reward
    #             else:
    #                 #选择正确
    #                 self._is_rerouted = False
    #                 self._route_before = None
    #                 self._route_after = None
    #                 reward += 1.0
    #                 return reward 
    #         else:
    #             "情况1:第一次选择按照拥挤度给奖励,修改为按照车辆意愿度给奖励应该更好. 情况2: 未改变路径"
    #             congestion = []
    #             for cs_id in self.charge_station_IDs:
    #                 cs_lane_id = self.connection.chargingstation.getLaneID(cs_id)
    #                 congestion.append(self.get_lane_density(cs_lane_id))
    #             max_density = max(congestion)
    #             min_density = min(congestion)               
    #             logger.debug(f"FOR first choice or choice unchanged: status:{self.status}")
    #             now_cs_id = self.env.EV.charging_stations_dict[self.status]
    #             cs_lane_now = self.connection.chargingstation.getLaneID(now_cs_id)
    #             now_density = self.get_lane_density(cs_lane_now)
    #             if max_density == min_density:
    #                 norm_density = 0.0  # 所有站点一样拥堵时，不做惩罚
    #             else:
    #                 norm_density = (now_density - min_density) / (max_density - min_density)
    #             reward += 5.0 * (1 - norm_density)
    #             return reward
    #     return reward            
   
    def get_lane_density(self, lane_id):
        MIN_GAP = 2.5
        lane_length = self.connection.lane.getLength(lane_id)
        # logging.debug(f"detect density of lane id:{lane_id}")
        vehicle_number = self.connection.lane.getLastStepVehicleNumber(lane_id)
        vehicle_length = self.connection.lane.getLastStepLength(lane_id)
        
        lane_density = vehicle_number / (lane_length / (MIN_GAP + vehicle_length))
        density = min(1.0, lane_density)
        return density
      
    def get_cs_density(self,cs_lane_id):
        
        incoming_lanes = self.net.getLane(cs_lane_id).getIncoming()
        incoming_lanes_ids = set([lane.getID() for lane in incoming_lanes])
        lanes = incoming_lanes_ids.copy()
        lanes.add(cs_lane_id)
        
        lane_density_sum = 0
        for lane in lanes:
            lane_density = self.get_lane_density(lane)
            lane_density_sum += lane_density
        congestion = lane_density_sum / len(lanes)
        congestion = min(1.0, congestion)
        return congestion
    
    def get_battery_ratio(self):
        current_capacity = float(self.connection.vehicle.getParameter(
            self.id, "device.battery.chargeLevel"))
        max_capacity = float(self.connection.vehicle.getParameter(
            self.id, "device.battery.capacity"))
        battery_ratio = current_capacity / max_capacity
        return battery_ratio

    
   
       

        
    
    


        
