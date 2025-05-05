import os, sys, sumolib
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import xml.etree.ElementTree as ET
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv, AECEnv
from gymnasium.utils import EzPickle
from pettingzoo.utils import agent_selector
import numpy as np
import traci
import logging
import pandas as pd
from pathlib import Path
from gymnasium.utils import seeding
from typing import Optional, Dict, List, Tuple, Union
from .ev import ElectricVehicle
from pettingzoo.utils import agent_selector
LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


class EVParallelEnv(gym.Env):  #  ParallelEnv, EzPickle
    logging.basicConfig(level=logging.DEBUG)
    metadata = {
        "render_modes": ["human"],
    }
    CONNECTION_LABEL = 0

    def __init__(
        self,
        net_file: str,
        sim_file: str,
        rou_file: str,
        begin_time: int = 0,
        seconds: int = 10000,
        enable_gui: bool = False,
        sumo_seed: Union[str, int] = "random",
        sumo_warnings: bool = True,
        virtual_display: Tuple[int, int] = (3200, 1800),
        max_depart_delay: int = -1,
        waiting_time_memory: int = 1000,
        time_to_teleport: int = -1,
        delta_time: int = 5,
        render_mode: Optional[str] = None,
        additional_sumo_cmd: Optional[str] = None,
        output_file: Optional[str] = None,
        reward_scalarisation = sum,
        common_reward = True,
    ):
        super().__init__()
        self._net = net_file#net.xml
        self._sim = sim_file#sumocfg
        self._rou = rou_file#rou.xml
        self.enable_gui = enable_gui
        self.render_mode = render_mode
        
        # SUMO配置
        if self.enable_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")
        
        
        # 时间相关参数
        self.begin_time = begin_time
        self.sim_max_time = seconds
        self.delta_time = delta_time
        self.max_depart_delay = max_depart_delay
        self.waiting_time_memory = waiting_time_memory
        self.time_to_teleport = time_to_teleport
        self.sumo_seed = sumo_seed
        self.sumo_warnings = sumo_warnings
        self.label = str(EVParallelEnv.CONNECTION_LABEL)
        EVParallelEnv.CONNECTION_LABEL += 1
        self.connection = None
        self.additional_sumo_cmd = additional_sumo_cmd
        self.metrics = []
        self.output_file = output_file
        self.virtual_display = virtual_display
        self.reward_scalarisation = reward_scalarisation
        self.common_reward = common_reward
        #traci connection with sumo
        self.connection = None
        
        if LIBSUMO:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net])
            traci_connection = traci
        else:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net],
                        label="init_connection" + self.label)
            traci_connection = traci.getConnection(
                "init_connection" + self.label)
        #获得充电站编号    cs_ids
        self.cs_ids = list(traci_connection.chargingstation.getIDList())
        logging.info(f"charge station: {self.cs_ids}")
        #获得充电站所在道路 cs_edges
        self.cs_edges = {}
        for cs_id in self.cs_ids:
            lane = traci_connection.chargingstation.getLaneID(cs_id)
            edge = traci_connection.lane.getEdgeID(lane)
            self.cs_edges[cs_id] = edge
        #获得车 出发的edge
        self.depart_edges = self._get_vehicle_depart_edges()
        #获得 edge 包含的 lane
        self.depart_lanes = self._get_depart_lanes()
        for edge, lanes in self.depart_lanes.items():
            logging.info(f"Edge {edge} has lanes: {lanes}")
        #获得最大连接车道数
        self.max_depart_lane_connections = self.get_max_connected_lanes(traci_connection, self.depart_lanes)        
        self.env_exist = False               
        self.net = sumolib.net.readNet(self._net)
        ########################################################################################
        self.action_space = spaces.Discrete(len(self.cs_ids) + 1) 
        logging.info(f"action_space{self.action_space}")
        self.observation_space = spaces.Box(
            low=np.zeros(3 * len(self.cs_ids) + self.max_depart_lane_connections + 2, dtype=np.float32),
            high=np.ones(3 * len(self.cs_ids) + self.max_depart_lane_connections + 2, dtype=np.float32)
        )       
        ########################################################################################
        # 初始化状态字典
        self.actions = {}
        self.rewards = {}
        self.observations = {}
        self.terminations = {}
        self.truncations = {}
        self.states = {}
        # 初始化车辆相关属性
        self.vehicles = {}
        self.vehicles_ID = []      
        
        self.completed_vehicles = set()  # 记录已完成任务的车辆ID
        self.truncated_vehicles = set()  # 记录被中断的车辆ID
        self.vehicle_stats = {
            "completed_vehicles": self.completed_vehicles,
            "truncated_vehicles": self.truncated_vehicles,
            "completion_history": [],  # 记录每个时间步新增的完成数量
            "truncation_history": []   # 记录每个时间步新增的中断数量
        }
        self.episode = 0
        self.curr_sim_step = 0
        self.num_steps = 0
        self.action_spaces = {}
        self.observation_spaces = {}
    '''
    xml 数据处理部分
    '''       
    #vtype : edge   EV 出发道路  
    def _get_vehicle_depart_edges(self):
        # 解析 XML 文件
        tree = ET.parse(self._rou)  
        root = tree.getroot()
        depart_edges = {}
        for flow in root.findall('flow'):
            from_edge = flow.get("from")  
            if from_edge is None:
                continue           
            vehicle_type = flow.get("type") 
            if vehicle_type is None:
                continue
            vtype = next((v for v in root.findall("vType") if v.get("id") == vehicle_type), None)
            if vtype is None:
                continue
            has_battery = next((param for param in vtype.findall("param") if param.get("key") == "has.battery.device"), None)
            if has_battery is None or has_battery.get("value") != "true":
                continue
            depart_edges[vehicle_type] = from_edge
        logging.info(f"Successfully got {len(depart_edges)} kinds of edges")
        for vtype, edge in depart_edges.items():
            logging.debug(f"EV: {vtype} departs from Edge: {edge}")
        return depart_edges 
    def _get_depart_lanes(self):
        tree = ET.parse(self._net)
        root = tree.getroot()        
        depart_lanes = {}
        for edge_id in self.depart_edges.values():
            edge = root.find(f".//edge[@id='{edge_id}']")
            if edge is not None:
                lanes = [lane.get('id') for lane in edge.findall('lane')]
                depart_lanes[edge_id] = sorted(lanes)
                logging.debug(f"Edge {edge_id} has lanes: {lanes}")
            else:
                logging.warning(f"Edge {edge_id} not found in network file")               
        return depart_lanes
    def get_max_connected_lanes(self, traci_connection, depart_lanes):

        max_connections = 0  # 初始化最大值

        for edge_id in depart_lanes.values():  
            for lane_id in edge_id:  
                connected_count = len(traci_connection.lane.getLinks(lane_id))  # 获取当前车道连接的车道数
                max_connections = max(max_connections, connected_count)  # 更新最大值

        return max_connections 

    '''
    sumo 模拟部分
    '''

    def _start_simulation(self):
        sumo_cmd = [self._sumo_binary,"-c",self._sim,"--max-depart-delay",str(self.max_depart_delay),
            "--waiting-time-memory",str(self.waiting_time_memory),"--time-to-teleport",str(self.time_to_teleport),]
        if self.begin_time > 0:
            sumo_cmd.append(f"-b {self.begin_time}")
        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")
        if self.additional_sumo_cmd is not None:
            sumo_cmd.extend(self.additional_sumo_cmd.split())
        if self.enable_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])
        if LIBSUMO:
            traci.start(sumo_cmd)
            self.connection = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.connection = traci.getConnection(self.label)
        if self.enable_gui or self.render_mode is not None:
            self.connection.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
                   

    def reset(self, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed,**kwargs)
        logging.info("Reset")
        if self.episode != 0:
            self.num_steps = self.curr_sim_step
            self.close()
            self.save_csv(self.output_file, self.episode)
        self.episode += 1
        self.metrics = []
        if seed is not None:
            self.sumo_seed = seed
            
        self._start_simulation()
        
            
        logging.info("当前车辆 ID 列表: %s", self.vehicles_ID) 
        
        if not self.vehicles_ID:
            logging.warning("No vehicles generated after waiting")
            # 初始化空的环境状态
            self.vehicles = {}
            self.actions = {}
            self.rewards = {}
            self.observations = {}
            self.terminations = {}
            self.truncations = {}
            self.states = {}
            self.completed_vehicles = set()
            self.truncated_vehicles = set()
            self.vehicle_stats = {
                "completed_vehicles": self.completed_vehicles,
                "truncated_vehicles": self.truncated_vehicles,
                "completion_history": [],
                "truncation_history": [],
            }
            self.action_spaces = {"default": self.action_space}
            self.observation_spaces = {"default": self.observation_space}
            return self._get_obs()   
        ##################################
        
        # 初始化车辆字典
        # self.vehicles = {
        #     ev_id: ElectricVehicle(self, ev_id, self.net, self.connection)
        #     for ev_id in self.vehicles_ID 
        #     if self.connection.vehicle.getParameter(ev_id, "has.battery.device").lower() == "true"
        # }
        self.vehicles = {}
        for ev_id in self.vehicles_ID:
            try:
                if self.connection.vehicle.getParameter(ev_id, "has.battery.device").lower() == "true":
                    self.vehicles[ev_id] = ElectricVehicle(self, ev_id, self.net, self.connection)
                    self.action_spaces[ev_id] = self.action_space
                    self.observation_spaces[ev_id] = self.observation_space
            except Exception as e:
                logging.warning(f"Error creating vehicle {ev_id}: {str(e)}")
                continue
        
        # 更新状态字典
        self.actions = {agent: None for agent in self.vehicles.keys()}
        self.rewards = {agent: 0.0 for agent in self.vehicles.keys()}
        self.observations = {agent: None for agent in self.vehicles.keys()}
        self.terminations = {agent: False for agent in self.vehicles.keys()}
        self.truncations = {agent: False for agent in self.vehicles.keys()}
        self.states = self.observations

        
        self.completed_vehicles = set()
        self.truncated_vehicles = set()
        self.vehicle_stats = {
            "completed_vehicles": self.completed_vehicles,
            "truncated_vehicles": self.truncated_vehicles,
            "completion_history": [],
            "truncation_history": [],
        }       
        
        return self._get_obs()
    
    @property
    def sim_step(self) -> float:
        return self.connection.simulation.getTime()

    def step(self, actions: Dict[str, int]):
        logging.debug("Step")
        logging.debug("Actions: {actions}")
        
        for ev_id, action in actions.items():
            if ev_id in self.vehicles:
                self._apply_action(ev_id, action)
        self.connection.simulationStep()
            
        self._update_vehicles()
        self.update_vehicle_colors()
        vehicle_stats = self._update_vehicle_stats()
        
        observations = self._get_obs()
        rewards = self._get_rewards()
        terminations = self._get_terminations()
        truncations = self._get_truncations()
        infos = self._get_infos()
        
        infos["vehicle_stats"] = vehicle_stats
        
        return observations, rewards, terminations, truncations, infos

    def _apply_action(self, ev_id: str, action: int):
        if action == 0:
            return
        if action > len(self.cs_ids):
            logging.warning(f"Invalid action {action}, ignoring...")
            return

        target_station_id = self.cs_ids[action - 1]
        station_lane_id = self.connection.chargingstation.getLaneID(target_station_id)
        station_edge_id = self.connection.lane.getEdgeID(station_lane_id)
        current_edge_id = self.connection.vehicle.getRoadID(ev_id)
        current_route = self.connection.vehicle.getRoute(ev_id)
        route_to_station = self.connection.simulation.findRoute(
            current_edge_id, 
            station_edge_id
        ).edges

        original_destination = current_route[-1]
        route_from_station = self.connection.simulation.findRoute(
            station_edge_id,
            original_destination
        ).edges
        
        new_route = route_to_station[:-1] + route_from_station
        self.connection.vehicle.setRoute(ev_id, new_route)
        station_end_pos = self.connection.chargingstation.getEndPos(target_station_id)
        self.connection.vehicle.setStop(
            ev_id,
            station_edge_id,
            pos=station_end_pos,
            duration=80  # 充电持续时间
        )

    def update_vehicle_colors(self):
        vehicles = self.connection.vehicle.getIDList()
        evs = [v for v in vehicles if self.connection.vehicle.getParameter(
            v, "has.battery.device") == "true"]

        if not LIBSUMO:
            traci.switch(self.label)
            
        for ev_id in evs:
            # 获取电池信息
            current_battery = float(self.connection.vehicle.getParameter(
                ev_id, "device.battery.actualBatteryCapacity"))
            max_battery = float(self.connection.vehicle.getParameter(
                ev_id, "device.battery.maximumBatteryCapacity"))
            
            # 计算电池百分比
            battery_percentage = (current_battery / max_battery) * 100

            if battery_percentage >= 80:
                color = [0, 255, 0, 255]  # RGBA
            elif battery_percentage >= 60:
                color = [255, 255, 0, 255]
            elif battery_percentage >= 20:
                color = [255, 165, 0, 255]
            else:
                color = [255, 0, 0, 255]
            self.connection.vehicle.setColor(ev_id, color)
            
    def _update_vehicles(self):
        for ev in self.vehicles.values():
            # 更新车辆的基本属性
            ev.lane_id = self.connection.vehicle.getLaneID(ev.id)
            ev.current_lane_length = self.connection.lane.getLength(ev.lane_id)
            ev.next_lane_ids = [link[0] for link in self.connection.lane.getLinks(ev.lane_id)] 
            ev.edge_id = self.connection.lane.getEdgeID(ev.lane_id)
            
            # 更新电池状态
            ev.actualCapacity = float(self.connection.vehicle.getParameter(
                ev.id, "device.battery.actualBatteryCapacity"))
            
            # 更新总里程
            ev.totalmileage = float(self.connection.vehicle.getDistance(ev.id))
            
        
    def _update_vehicle_stats(self):
        """
        更新车辆状态统计信息，记录新完成或中断的车辆
        """
        current_completed = 0
        current_truncated = 0
        
        # 检查所有车辆的状态
        for ev_id, ev in self.vehicles.items():
            # 检查完成状态
            if ev.get_termination() and ev_id not in self.completed_vehicles:
                self.completed_vehicles.add(ev_id)
                current_completed += 1
                logging.debug(f"Vehicle {ev_id} completed task successfully")
                
            # 检查中断状态
            if ev.get_truncation() and ev_id not in self.truncated_vehicles:
                self.truncated_vehicles.add(ev_id)
                current_truncated += 1
                logging.debug(f"Vehicle {ev_id} was truncated")
        
        # 记录当前时间步的新增数量
        self.vehicle_stats["completion_history"].append(current_completed)
        self.vehicle_stats["truncation_history"].append(current_truncated)
        
        return {
            "completed_vehicles": list(self.completed_vehicles),  # 转换为列表以便JSON序列化
            "truncated_vehicles": list(self.truncated_vehicles),
            "current_completed": current_completed,
            "current_truncated": current_truncated,
            "total_completed": len(self.completed_vehicles),
            "total_truncated": len(self.truncated_vehicles),
            "active_vehicles": len(self.vehicles),
            "completion_history": self.vehicle_stats["completion_history"],
            "truncation_history": self.vehicle_stats["truncation_history"]
        }
    def observation_spaces(self, vehid: str):
        return self.vehicles[vehid].observation_space
    
    def action_spaces(self, vehid: str):
        return self.vehicles[vehid].action_space

    def _get_obs(self):
        # self.observations.update(
        #     {ev_id: ev.get_observation() 
        #      for ev_id, ev in self.vehicles.items()}
        # )
        observations = {}
        for ev_id, ev in self.vehicles.items():
            try:
                # 检查车辆是否在模拟中
                if ev_id not in self.connection.vehicle.getIDList():
                    continue
                    
                # 检查车辆是否已经进入网络
                if not self.connection.vehicle.getRoadID(ev_id):
                    continue
                    
                # 获取观察值
                obs = ev.get_observation()
                if obs is not None:
                    observations[ev_id] = obs
            except Exception as e:
                logging.warning(f"Error getting observation for vehicle {ev_id}: {str(e)}")
                continue
                
        self.observations.update(observations)        
        return self.observations
    

    def _get_rewards(self):
        # self.rewards.update(
        #     {ev_id: ev.get_rewards()[ev_id] 
        #      for ev_id, ev in self.vehicles.items()}
        # )
        rewards = {}
        for ev_id, ev in self.vehicles.items():
            try:
                # 检查车辆是否在模拟中
                if ev_id not in self.connection.vehicle.getIDList():
                    continue
                    
                # 检查车辆是否已经进入网络
                if not self.connection.vehicle.getRoadID(ev_id):
                    continue
                    
                # 获取奖励值
                reward_dict = ev.get_rewards()
                if ev_id in reward_dict:
                    rewards[ev_id] = reward_dict[ev_id]
            except Exception as e:
                logging.warning(f"Error getting reward for vehicle {ev_id}: {str(e)}")
                continue
                
        self.rewards.update(rewards)
        return self.rewards
    
    def _get_terminations(self):
        self.terminations.update(
            {ev_id: ev.get_termination() 
             for ev_id, ev in self.vehicles.items()}
        )
        return self.terminations

    def _get_truncations(self):
        self.truncations.update(
            {ev_id: ev.get_truncation() 
             for ev_id, ev in self.vehicles.items()}
        )
        return self.truncations

    def _get_infos(self):
        # self.infos = {}
        # self.infos.update(
        #     {ev_id: ev.get_info() 
        #      for ev_id, ev in self.vehicles.items()}
        # )
        infos = {}
        for ev_id, ev in self.vehicles.items():
            try:
                # 检查车辆是否在模拟中
                if ev_id not in self.connection.vehicle.getIDList():
                    continue
                    
                # 检查车辆是否已经进入网络
                if not self.connection.vehicle.getRoadID(ev_id):
                    continue
                    
                # 获取信息
                info = ev.get_info()
                if info is not None:
                    infos[ev_id] = info
            except Exception as e:
                logging.warning(f"Error getting info for vehicle {ev_id}: {str(e)}")
                continue
                
        self.infos = infos
        return self.infos
    
    def render(self):
        if self.render_mode == "human":
            return

    def close(self):
        if not self.connection:
            return

        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()

        self.connection = None

    def __del__(self):
        self.close()
        
    def save_csv(self, output_file, episode):
        if output_file is not None:
            Path(Path(output_file).parent).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.metrics).to_csv(output_file +
                                              f"_episode{episode}" + ".csv", index=False)
            pd.DataFrame(
                self.total_metrics).to_csv(
                output_file + f"_total_metrics_{self.label}" + ".csv", index=False)

class EVPZ(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human"],
        "name": "ev_parallel_v0",
        "is_parallelizable": True
    }

    def __init__(self, **kwargs):
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs
        self.seed()
        self.env = EVParallelEnv(**self._kwargs)
        self.render_mode = kwargs.get("render_mode", None)
        
        # 初始化智能体列表为空
        self.agents = []
        self.possible_agents = []
        self._agent_selector = None
        self.agent_selection = None
        
        # 初始化动作和观察空间
        self.action_spaces = {}
        self.observation_spaces = {}
        
        # 初始化其他状态
        self.rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}
        self._cumulative_rewards = {}

    def seed(self, seed=None):
        self.randomizer, seed = seeding.np_random(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.env.reset(seed=seed, options=options)###启动sumo模拟
        self.vehicles_ID = self.env.connection.vehicle.getIDList()
        
        self.action_spaces = {}
        self.observation_spaces = {}
        
        # 初始化其他状态
        self.rewards = {}
        self._cumulative_rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}
        
        # 初始化智能体列表为空
        self.agents = []
        self.possible_agents = []
        self._agent_selector = None
        self.agent_selection = None
        
        max_wait_steps = 10000
        wait_step = 0
        while wait_step < max_wait_steps:
            self.env.connection.simulationStep()
            self.vehicles_ID = self.env.connection.vehicle.getIDList()
            if self.vehicles_ID:
                break
            wait_step += 1
            
        logging.info("EVPZ当前车辆 ID 列表: %s", self.vehicles_ID)
        
        if not self.vehicles_ID:
            logging.warning("No vehicles generated after waiting")
            # 初始化一个默认的动作和观察空间
            self.action_spaces = {"default": self.env.action_space}
            self.observation_spaces = {"default": self.env.observation_space}
            self.agents = ["default"]
            self.possible_agents = ["default"]
            self._agent_selector = agent_selector(self.agents)
            self.agent_selection = self._agent_selector.reset()
            return None
        
        # ev_vehicles = {
        #     ev_id: ElectricVehicle(self.env, ev_id, self.env.net, self.env.connection)
        #     for ev_id in self.vehicles_ID 
        #     if self.env.connection.vehicle.getParameter(ev_id, "has.battery.device").lower() == "true"
        # }
        ev_vehicles = {}
        for ev_id in self.vehicles_ID:
            try:
                if self.env.connection.vehicle.getParameter(ev_id, "has.battery.device").lower() == "true":
                    ev_vehicles[ev_id] = ElectricVehicle(self.env, ev_id, self.env.net, self.env.connection)
                    self.action_spaces[ev_id] = self.env.action_space
                    self.observation_spaces[ev_id] = self.env.observation_space
            except Exception as e:
                logging.warning(f"Error creating vehicle {ev_id}: {str(e)}")
                continue
        
        if ev_vehicles:
            self.agents = list(ev_vehicles.keys())
            self.possible_agents = self.agents
            self._agent_selector = agent_selector(self.agents)
            self.agent_selection = self._agent_selector.reset()
            
            # 初始化动作和观察空间
            # self.action_spaces = {
            #     agent: ev_vehicles[agent].action_space for agent in self.agents
            # }
            # self.observation_spaces = {
            #     agent: ev_vehicles[agent].observation_space for agent in self.agents
            # }
            
            # 初始化其他状态
            self.rewards = {agent: 0 for agent in self.agents}
            self._cumulative_rewards = {agent: 0 for agent in self.agents}
            self.terminations = {agent: False for agent in self.agents}
            self.truncations = {agent: False for agent in self.agents}
            self.infos = {agent: {} for agent in self.agents}
            
            self.env.vehicles = ev_vehicles
            self.env.observations = self.env._get_obs()
            self.env.rewards = self.env._get_rewards()
            self.env.terminations = self.env._get_terminations()
            self.env.truncations = self.env._get_truncations()
            self.env.infos = self.env._get_infos()
        else:
            logging.warning("No electric vehicles found in the simulation")
            
            self.action_spaces = {"default": self.env.action_space}
            self.observation_spaces = {"default": self.env.observation_space}
            self.agents = ["default"]
            self.possible_agents = ["default"]
            self._agent_selector = agent_selector(self.agents)
            self.agent_selection = self._agent_selector.reset()
        

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        # obs = self.env.observations[agent].copy()
        # return obs
        try:
            # 检查智能体是否存在
            if agent not in self.env.observations:
                logging.warning(f"Agent {agent} not found in observations")
                return None
                
            # 检查观察值是否为None
            if self.env.observations[agent] is None:
                logging.warning(f"Observation for agent {agent} is None")
                return None
                
            # 返回观察值的副本
            obs = self.env.observations[agent].copy()
            return obs
        except Exception as e:
            logging.warning(f"Error getting observation for agent {agent}: {str(e)}")
            return None

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        return self.env.render(mode)

    def step(self, action):
        if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
            return self._was_dead_step(action)
        agent = self.agent_selection
        if not self.action_spaces[agent].contains(action):
            raise Exception(f"Action must be Discrete")

        self.env._apply_action({agent: action})

        if self._agent_selector.is_last():#判断是否是最后一个智能体，然后集中更新
            for _ in range(self.env.delta_time):
                self.env.connection.simulationStep()
                
            self.env._update_vehicles()
            self.env.update_vehicle_colors()
            self.env._get_obs()
            self.rewards = self.env._get_rewards()####所有agent的reward
            self.env._get_truncations()
            self.infos =self.env._get_infos()
            vehicle_stats = self.env._update_vehicle_stats()
            self.infos["vehicle_stats"] = vehicle_stats
        else:
            self._clear_rewards()
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
        
        


        
        
