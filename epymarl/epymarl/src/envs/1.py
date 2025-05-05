import os, sys, sumolib,operator,random
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import src.utils.fmp_utils as fmp_utils
import src.utils.xml_utils as xml_utils
import xml.etree.ElementTree as ET
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
import numpy as np
import traci
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from .ev import ElectricVehicle
from itertools import chain
from statistics import mean
LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)#WARNING   DEBUG
 

class EV():
    def __init__(
        self,
        net_xml_file_path: str = None,
        demand_xml_file_path: str = None,
        additional_xml_file_path: str = None,
    ):
        self.__sumo_config_init(net_xml_file_path, demand_xml_file_path, additional_xml_file_path)#net.xml,rou.xml,add.xml
    def __sumo_config_init(
        self,
        net_xml_file_path: str = None,
        demand_xml_file_path: str = None,
        additional_xml_file_path: str = None,
    ):
        (   raw_vertices,  # [id (str), x_coord (float), y_coord (float)]列表（junction_id, 坐标x,坐标y）
            raw_charging_stations,  # [id, (x_coord, y_coord), edge_id, charging speed]列表（id,charge_station坐标，cs所在边edge_id，station_power, station_end_position）
            raw_electric_vehicles,  # [id (str), maximum speed (float), capacity (float)]列表（route_ev_id,max_speed,max_battery_capacity)
            raw_edges,  # [id (str), from_vertex_id (str), to_vertex_id (str)]列表（edge边id,junction始号，junction终号，边长）
            raw_departures,  # [vehicle_id, starting_edge_id]列表（车id,route起始边id)
            raw_demand,  # [junction_id, dest_vertex_id]列表（junction_id,destination_id)
        ) = xml_utils.decode_xml_fmp(
            net_xml_file_path, demand_xml_file_path, additional_xml_file_path
        )

        # `vertices` is a list of Vertex instances
        # `self.vertex_dict` is a mapping from vertex id in SUMO to idx in vertices
        vertices, self.vertex_dict = fmp_utils.convert_raw_vertices(raw_vertices)

        # `edges` is a list of Edge instances
        # `self.edge_dict` is a mapping from SUMO edge id to idx in `edges`
        # `self.edge_length_dict` is a dictionary mapping from SUMO edge id to edge length
        (   edges,
            self.edge_dict,
            self.edge_length_dict,
        ) = fmp_utils.convert_raw_edges(raw_edges, self.vertex_dict)
        
        # `electric_vehicles` is a list of ElectricVehicles instances
        # `self.ev_dict` is a mapping from ev sumo id to idx in `electric_vehicles`
        electric_vehicles, self.ev_dict = fmp_utils.convert_raw_electric_vehicles(
            raw_electric_vehicles#列表（route_ev_id,max_speed,max_battery_capacity)
        )
        
        departures = fmp_utils.convert_raw_departures(raw_departures)

        # `charging_stations` is a list of ChargingStation instances
        # `self.charging_stations_dict` is a mapping from idx in `charging_stations` to SUMO station id
        (   charging_stations,
            self.charging_stations_dict,    
        ) = fmp_utils.convert_raw_charging_stations(
            raw_charging_stations,
        )


        # `demand` is a list of Demand instances
        demands = raw_demand

        # set the FMP variables
        self.vertices = vertices
        self.edges = edges
        self.charging_stations = charging_stations
        self.electric_vehicles = electric_vehicles
        self.demands = fmp_utils.convert_raw_demand(raw_demand,self.vertex_dict)
        self.departures = departures
        self.n_demand = len(demands)
        self.n_vertex = len(self.vertices)
        self.n_edge = len(self.edges)
        self.n_vehicle = self.n_electric_vehicle = len(self.electric_vehicles)
        self.n_charging_station = len(self.charging_stations)

    def _is_valid(self):#修改判断条件
        if (
            not self.n_vertex
            or not self.n_demand
            or not self.n_edge
            or not self.n_vehicle
            or not self.n_charging_station
            or self.vertices is None
            or self.charging_stations is None
            or self.electric_vehicles is None
            or self.demands is None
            or self.edges is None
            or self.departures is None
        ):
            return False
     
        

class EVParallelEnv(ParallelEnv):  #  ParallelEnv, EzPickle
    # logger.basicConfig(level=logger.DEBUG)

    metadata = {
        "render_modes": ["human"],
    }
    EV = property(operator.attrgetter("_EV"))
    CONNECTION_LABEL = 0

    def __init__(
        self,
        net_file: str,
        sim_file: str,
        rou_file: str,
        add_file: str,
        begin_time: int = 0,
        time_limit: int = 1200,
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
        reward_scalarisation = mean,
        common_reward = True,
    ):
        super().__init__()
        self._net = net_file#net.xml
        self._sim = sim_file#sumocfg
        self._rou = rou_file#rou.xml
        self._add = add_file#add.xml
        self._EV = EV(self._net,self._rou,self._add)
        ##########################################################################################################################
        # setup Petting-Zoo environment variablesEV  需要从rou.xml文件中得到有多少个 EV agent
        self.possible_agents = list(self.EV.ev_dict.keys())# 以EV id为agent,是一个list,保存的应该是EV id 
        self.agent_name_idx_mapping = self.EV.ev_dict        
        self.agents = self.possible_agents[:]
     
        self.actions = {agent: None for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}#浅拷贝，防止删除某个agent时，possible_agent中的agent也被删除
        self.truncations = {agent: False for agent in self.agents}
        self.states = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}

        
        self.episode_data = {
            "simulation_time": 0,
            "task_success": False,
            "completion_rate": 0.0,
            "truncated_rate": 0.0,
            "average_battery": 0.0,
            "mean_reward": 0.0,
        }
        
        ############################################################################################################################
        self.enable_gui = enable_gui
        self.render_mode = render_mode
        
        # SUMO配置
        if self.enable_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")
        
        
        # 时间相关参数
        self.begin_time = begin_time
        self.sim_max_time = time_limit
        self.delta_time = delta_time
        self.max_depart_delay = max_depart_delay
        self.waiting_time_memory = waiting_time_memory
        self.time_to_teleport = time_to_teleport
        self.sumo_seed = sumo_seed
        self.sumo_warnings = sumo_warnings
        self.label = str(EVParallelEnv.CONNECTION_LABEL)
        EVParallelEnv.CONNECTION_LABEL += 1
        self.additional_sumo_cmd = additional_sumo_cmd
        self.metrics = []
        self.total_metrics = []
        self.last_saved_episode = 0  # 添加一个变量跟踪最后保存的episode
        self.output_file = output_file
        self.virtual_display = virtual_display
        self.reward_scalarisation = reward_scalarisation
        self.common_reward = common_reward
        #traci connection with sumo
        self.connection = None
        ###############################################################################初始化观察空间,准备观察空间所需参数
        
        if LIBSUMO:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net , "-a", self._add],)
            traci_connection = traci
        else:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net, "-a", self._add],
                        label="init_connection" + self.label)
            traci_connection = traci.getConnection(
                "init_connection" + self.label)
        #获得充电站编号    cs_ids
        self.cs_ids = list(traci_connection.chargingstation.getIDList())
        # logger.info(f"charge station: {self.cs_ids}")
        #获得充电站所在道路 cs_edges
        self.cs_edges = {}
        for cs_id in self.cs_ids:
            lane = traci_connection.chargingstation.getLaneID(cs_id)
            edge = traci_connection.lane.getEdgeID(lane)
            self.cs_edges[cs_id] = edge

       
        self.env_exist = False               
        self.net = sumolib.net.readNet(self._net)    
        self.completed_vehicles = set()  # 记录已完成任务的车辆ID
        self.truncated_vehicles = set()  # 记录被中断的车辆ID

        self.episode = 0
        self.curr_sim_step = 0
        self.num_steps = 0
        
        # 定义动作空间和观察空间
        self.num_stations = len(self.cs_ids)
        self.action_spaces = {
            agent: spaces.Discrete(self.num_stations + 1)  # 0~n-1表示选择对应的充电站,第n个表示无效动作
            for agent in self.agents
        }
        self.observation_spaces = {
            agent: spaces.Box(
                low=np.zeros(1 + 2 * self.num_stations, dtype=np.float32),  # 1个电量比例 + 每个充电站的距离和拥挤程度
                high=np.ones(1 + 2 * self.num_stations, dtype=np.float32)
            )
            for agent in self.agents
        }

        self.is_reset = False

        ###其他信息
        self.infos = {
            "success_rate": 0.0,
            "max_waiting_time": 0.0,
            "mean_waiting_time": 0.0,
            "truncated_rate": 0.0,
            "mean_reward": 0.0,                
        }
            
        
        self.episode_data = {
            "simulation_time": 0,
            "task_success": False,
            "completion_rate": 0.0,
            "truncated_rate": 0.0,
            "average_battery": 0.0,
            "mean_reward": 0.0,
        }
        ####################################
        self.episode_limit = 150
        self.n_agents = len(self.agents)
        
    #定义标准接口
    def action_spaces(self, agent):
        return self.action_spaces(agent)
    def observation_spaces(self, agent):
        return self.observation_spaces(agent)
                

    def _start_simulation(self):#待改，加一个info提示
        logger.info("start simulation")
        sumo_cmd = [self._sumo_binary,"-c",self._sim,"--max-depart-delay",str(self.max_depart_delay),
            "--waiting-time-memory",str(self.waiting_time_memory),"--time-to-teleport",str(self.time_to_teleport),"--no-step-log",]
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
        
                   

    def reset(self, seed: Optional[int] = None):
        logger.info("Reset")
        self.is_reset = True
        if self.episode != 0:
            self.num_steps = self.curr_sim_step
            self.close()
            # self.save_csv(self.output_file, self.episode)
        self.episode += 1
        self.metrics = []

        
        if seed is not None:
            self.sumo_seed = seed
        
        self.infos = {
            "success_rate": 0.0,
            "max_waiting_time": 0.0,
            "mean_waiting_time": 0.0,
            "truncated_rate": 0.0,
            "mean_reward": 0.0,                
        }
        
        '''
        cs数据初始化
        '''
        for i, cs in enumerate(self.EV.charging_stations):
            self.EV.charging_stations[i].n_slot = cs.n_slot or 1
            self.EV.charging_stations[i].charging_vehicle = (
                cs.charging_vehicle or list()
            )
        
        '''
        实例化EV
        '''
        self._start_simulation()
        #初始化电车的电量
        for ev_id in self.possible_agents:
            if self.connection.vehicle.getParameter(ev_id, "has.battery.device").lower() == "true":
                max_capacity = float(self.connection.vehicle.getParameter(ev_id, "device.battery.capacity"))
                current_capacity = float(self.connection.vehicle.getParameter(ev_id, "device.battery.chargeLevel"))
                if current_capacity == -1:
                    current_capacity = random.uniform(0.25, 0.45) * max_capacity
                    self.connection.vehicle.setParameter(ev_id, "device.battery.chargeLevel", current_capacity)
        
        #实例化电车
        self.vehicles = {
            ev_id: ElectricVehicle(self, ev_id, self.net, self.connection)
            for ev_id in self.possible_agents# id,str
            if self.connection.vehicle.getParameter(ev_id, "has.battery.device").lower() == "true"
        }
        '''
        初始化agent所需空间,初始化每一个智能体的观察空间
        '''
        self.agents = self.possible_agents[:]
        self.actions = {agent: None for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}#..
        self.truncations = {agent: False for agent in self.agents}#..

    
        self.observations = self._get_observations(self.is_reset)
        self.states = self.observations
       
        self.episode_data = {
            "simulation_time": 0,
            "task_success": False,
            "completion_rate": 0.0,
            "truncated_rate": 0.0,
            "average_battery": 0.0,
            "mean_reward": 0.0,
        }
 
        return self.observations   

    @property
    def sim_step(self) -> float:
        return self.connection.simulation.getTime()
    
    def step(self, actions: Dict[str, int]):
        logger.debug("Step")
        # self.agents = [agent for agent in self.agents if not self.terminations[agent] and not self.truncations[agent]]#..
        for agent in self.agents:
            # 处理智能体终止状态
            if self.terminations[agent] or self.truncations[agent]:
                self.observations[agent] = None
                self.rewards[agent] = 0.0  # 已完成或中断的智能体不再获得奖励
                continue
            if (not self.is_reset) and (self.vehicles[agent].status < self.num_stations):
                """
                如果车的当前车道与充电站a车道重合,则:
                1.可以在此时选择非充电站a,设置stop
                2.不可以在该车道重新设置a stop
                否则出现sumo 报错:too close to brake
                route can't be set
                3.也就是说一旦车辆进行充电站a 所在车道它的下一步可选择动作就锁住为a
                4.1当前车道与a 车道重合,且ation = a,那么下一步可用动作只有a
                4.2当前车道与a 车道重合,且action != a,那么下一步可用动作不为a
                
                """
                for cs in self.cs_ids:
                    vehlaneID = self.connection.vehicle.getLaneID(agent)#当前车道
                    cslaneID = self.connection.chargingstation.getLaneID(cs)
                    
                    actionlaneID = self.connection.chargingstation.getLaneID(self.EV.charging_stations_dict[actions[agent]])#action选择充电站的车道
                    vehPos = self.connection.vehicle.getLanePosition(agent)
                    csPos = self.connection.chargingstation.getEndPos(cs)
                    current_capacity = float(self.connection.vehicle.getParameter(agent, "device.battery.chargeLevel"))
                    max_capacity = float(self.connection.vehicle.getParameter(agent, "device.battery.capacity"))
                    battery_ratio = current_capacity / max_capacity
                    if (vehlaneID == cslaneID) and (actionlaneID == cslaneID) and (self.vehicles[agent].status == actions[agent]):
                        "当前车道与a 车道重合,且ation = a,那么下一步可用动作只有a"
                        "还需看上一步动作,1.如果上一步动作不为a,这一步选择a,还是会出现无法停止的情况"
                        logger.debug(f"ACTION LOCKED!--->veh:{agent} battery{battery_ratio} vehlaneID :{vehlaneID} cslaneID:{cslaneID} actionlaneID:{actionlaneID} vehPos:{vehPos} csPos:{csPos}")
                        for i, cs_id in enumerate(self.cs_ids):
                            if cs_id == cs:
                                self.vehicles[agent]._locked_cs = i
                                self.vehicles[agent]._locked_count = 1#还未蒙蔽动作

                    if (cslaneID == vehlaneID) and (actionlaneID != cslaneID):
                        "当前车道与a 车道重合,且action != a,那么下一步可用动作不为a"
                        for i, cs_id in enumerate(self.cs_ids):
                            logger.debug(f"STATION MISSED--->veh:{agent} battery{battery_ratio} vehlaneID :{vehlaneID} cslaneID:{cslaneID} actionlaneID:{actionlaneID} vehPos:{vehPos} csPos:{csPos}")
                            if cs_id == cs:
                                self.vehicles[agent]._missed_cs = i
                                self.vehicles[agent]._missed_count = 1#还未蒙蔽动作FAILURE
                    if (vehlaneID == cslaneID) and (actionlaneID == cslaneID) and (self.vehicles[agent].status != actions[agent]):
                        "设置动作失败,继续执行上一步动作"
                        logger.debug(f"ACTION SET FAILURE--->veh:{agent} battery{battery_ratio} vehlaneID :{vehlaneID} cslaneID:{cslaneID} actionlaneID:{actionlaneID} vehPos:{vehPos} csPos:{csPos}")
                        self.vehicles[agent]._failure_set = True
            # 处理正常智能体
            if  self.num_stations <= self.vehicles[agent].status < 2*self.num_stations:
                "若车辆在充电时[n_cs,2n_cs)选择新的动作,此时status仍保持上一个选择状态,因此,实际上不会做出任何动作,考虑当status[cs,2cs),可用动作为status -cs"
                self._keep_charging(agent)#充电完，会出现任务完成的情况
            elif self.vehicles[agent].status < self.num_stations:
                "选择充电站,情况1:第一次选择充电站,情况2:第二次做出了与第一次不同的选择(此时,之前的路径上已经设置了stop,因此要清理之前路径上的stop)"
                self._state_transition(agent, actions[agent])#在做决定去哪个充电站时会出现任务中止的情况
                
        # 更新所有电车路径后，开始sumo模拟
        self.connection.simulationStep()
        self.update_vehicle_colors()
        self.observations = self._get_observations(self.is_reset)
        self.rewards = self._get_rewards()
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]
        self.infos = self._get_infos()
        self._update_episode_data()            
        if all(self.terminations.values()) or all(self.truncations.values()):
            "清空智能体就会reset"
            self.agents = []
            return {},{},self.terminations, self.truncations, self.infos
        if all([self.terminations[agent] or self.truncations[agent]for agent in self.agents]):
            self.agents = []
            return {},{},self.terminations, self.truncations, self.infos
        
        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def _keep_charging(self,agent):
        """
        当agent 选择keep charging, 此时车辆 isStopped, status = n_cs +cs_id
        在充电的情况需要处理任务完成的情况,并更新观察、奖励等,打上标签is_terminated,
        之后不再更新观察、奖励。车辆停止isStop,并驶向目的地,在目的地设置stop
        """
        current_capacity = float(self.connection.vehicle.getParameter(agent, "device.battery.chargeLevel"))
        max_capacity = float(self.connection.vehicle.getParameter(agent, "device.battery.capacity"))
        battery_ratio = current_capacity / max_capacity
        if battery_ratio > 0.95:
            self.terminations[agent] = True
            self.observations[agent] = None
            self.rewards[agent]= self.vehicles[agent].get_rewards()
            self.vehicles[agent]._is_terminated = True#记录第一次完成任务,给予奖励,之后设置为False
            self.vehicles[agent].status += 2 * self.num_stations
            logger.debug(f"Sys is dealing with terminated agent :id{agent} ,there are agents in simulation:{self.connection.vehicle.getIDList()}")
            if len(self.connection.vehicle.getStops(vehID=agent, limit=1))==1:
                self.connection.vehicle.replaceStop(vehID= agent, nextStopIndex=0,edgeID="")
                logger.debug(f"vehicle{agent} can leave from station NOW!")
                terminated_route = self.connection.vehicle.getRoute(agent)#理论上和cs_id_edge一样
                terminated_destination_edge = terminated_route[-1]
                terminated_current_edge = self.connection.vehicle.getRoadID(agent)
                terminated_new_route = self.connection.simulation.findRoute(terminated_current_edge,
                                                                            terminated_destination_edge).edges
                logger.debug(f"here is the terminated new route:{terminated_new_route},sys will set stop in the destination!")
                self.connection.vehicle.setRoute(agent,terminated_new_route)
                self.connection.vehicle.setStop(agent,
                                terminated_destination_edge,
                                pos=self.connection.lane.getLength(terminated_destination_edge + "_0"),  # 在边的末端停止
                                duration=1000000)
            if self.connection.vehicle.isStopped(agent):#可能无法做到到达预定终点时电量还很大
                logger.debug(f"Terminated agent has arrived at destination ,but it is still in simulation")
        else:
            """
            继续充电,并更新观察、奖励空间
            """
            cs_idx = (self.vehicles[agent].status - self.num_stations)
            logger.debug(f"vehicle{agent} is charging at station {cs_idx}, with current battery ratio {battery_ratio},current edge: {self.connection.vehicle.getRoadID(agent)}")
            
       
    def _state_transition(self, agent, action):
        current_capacity = float(self.connection.vehicle.getParameter(agent, "device.battery.chargeLevel"))
        max_capacity = float(self.connection.vehicle.getParameter(agent, "device.battery.capacity"))
        battery_ratio = current_capacity / max_capacity
        if not self.is_reset:
            self.vehlane = self.connection.vehicle.getLaneID(agent)
            self.cslane = self.connection.chargingstation.getLaneID(self.EV.charging_stations_dict[action])
            self.vehPos = self.connection.vehicle.getLanePosition(agent)  
            self.csPos = self.connection.chargingstation.getEndPos(self.EV.charging_stations_dict[action]) 
            self.last_cs = self.vehicles[agent].status
            self.action_cs = action
            logger.debug(f"veh:{agent}status:{self.last_cs},action:{action}, vehlane: {self.vehlane},cslane: {self.cslane},vehPos: {self.vehPos},csPos:{self.csPos}")
        if battery_ratio <=0.05:
            """
            如果任务失败，则对智能体进行标记,更新此时奖励、观察等信息，之后不再对失败的智能体进行更新
            不在sumo中移除电车,防止环境报错.
            在self.agents中保留智能体,但在step循环中跳过
            在计算奖励和观察空间时,对 标记了is_truncated的智能体返回0.0和None
            !!!!
            可能不是第一次选择动作,所以要清除之前的stop,重新设置终点,防止电量为负值的电车到充电站去
            在sumo中stop只有在进入新路段的时候才能实现。所以当
            """
            if len(self.connection.vehicle.getStops(vehID=agent, limit=1))==1:
                self.connection.vehicle.replaceStop(vehID= agent, nextStopIndex=0,edgeID="")
            self.truncations[agent]= True 
            self.rewards[agent]= self.vehicles[agent].get_rewards()
            self.observations[agent] = None
            self.vehicles[agent]._is_truncated = True
            self.vehicles[agent].status += 2 * self.num_stations
            logger.debug(f"Sys is dealing with truncated agent :id{agent} ,there are agents in simulation:{self.connection.vehicle.getIDList()}")
            truncated_route = self.connection.vehicle.getRoute(agent)
            truncated_destination_edge = truncated_route[-1]  # 获取目的地边
            truncated_current_edge = self.connection.vehicle.getRoadID(agent)
            truncated_new_route = self.connection.simulation.findRoute(truncated_current_edge,
                                                                       truncated_destination_edge).edges
            logger.debug(f"here is the truncated new route:{truncated_new_route},sys will set stop in the destination!")
            self.connection.vehicle.setRoute(agent, truncated_new_route)
            self.connection.vehicle.setStop(agent,
                                truncated_destination_edge,
                                pos=self.connection.lane.getLength(truncated_destination_edge + "_0"),  # 在边的末端停止
                                duration=1000000)
            if self.connection.vehicle.isStopped(agent):
                logger.debug(f"Truncated agent has arrived at destination ,but it is still in simulation")
        else:
            """
            选择充电站且电量健康,并且更新观察，奖励等空间
            情况1:第一次选择充电站,情况2:第二次做出了与第一次不同的选择(此时,之前的路径上已经设置了stop,因此要清理之前路径上的stop,情况3:做出与上一次相同的选择)
            """
            if self.is_reset:
                "第一次选择充电站"
                cs_idx = action
                cs_id = self.EV.charging_stations_dict[cs_idx]             
                self._to_CS(agent, cs_id)
                self.vehicles[agent].status = action#设置好路线后再更新status
                logger.info(f"vehicle {agent} choose to charge ,route is set and status is updated to {self.vehicles[agent].status}")
                if self.connection.vehicle.isStoppedParking(agent):#如果到了充电站，更新status
                    self.vehicles[agent]._is_stopped = True        
                    self.vehicles[agent].status += self.num_stations
                    self.EV.charging_stations[cs_idx].charging_vehicle.append(agent)
                    logger.info(f"vehicle {agent} is charging at station {cs_id}")
            elif self.action_cs == self.last_cs:
                "情况2,不是第一次选择充电站,当前选择的充电站与上一次选择的充电站相同" 
                "出现车辆一直在同一个车道,且不改变目的地的情况" 
                logger.info(f"vehicle {agent} keep the desicion of choosing station {self.vehicles[agent].status}, no need to change the route")
                self.vehicles[agent].status = action
                cs_idx = action
                cs_id = self.EV.charging_stations_dict[cs_idx]
                logger.info(f"vehicle {agent} choose the same Charging Station {self.vehicles[agent].status},with battery{battery_ratio}")
                if self.connection.vehicle.isStoppedParking(agent):#如果到了充电站，更新status
                    self.vehicles[agent]._is_stopped = True        
                    self.vehicles[agent].status += self.num_stations
                    self.EV.charging_stations[cs_idx].charging_vehicle.append(agent)
                    logger.info(f"vehicle {agent} is charging at station {cs_id}")
            # elif (self.vehicles[agent].status != action) and (self.vehlane == self.cslane) and (self.csPos -5 <= self.vehPos):
            #     "情况3,不是第一次做选择,选择不同充电站,且当前所在道路就是动作选择充电站所在车道,且错过充电站---->不做任何动作"
            #     logger.info(f"vehicle {agent} missed the charging station {self.EV.charging_stations_dict[action]} and is rerouting to it.SYS denied the decision")
            #     "情况2,不是第一次做选择,且选择了不同的充电站。情况2.1当车辆来到某一个充电站的车道上时,它的动作选择就会被锁住.情况2.2当车辆错过充电站a后又将充电站a设置为目的地"  
            elif (self.vehicles[agent].status != action) and (not self.vehicles[agent]._failure_set):
                "其余情况,正常处理"
                if len(self.connection.vehicle.getStops(vehID=agent, limit=1))==1:
                    self.connection.vehicle.replaceStop(vehID= agent, nextStopIndex=0,edgeID="")
                logger.info(f"vehicle {agent} decides to change the station, and action is {action},status is {self.vehicles[agent].status}")
                if (self.vehicles[agent].status != action) and (not self.is_reset):
                    #说明电车重新选择了充电站
                    self.vehicles[agent]._is_rerouted = True
                    self.vehicles[agent]._route_before = self.vehicles[agent].status
                    self.vehicles[agent]._route_after = action
                cs_idx = action
                cs_id = self.EV.charging_stations_dict[cs_idx]             
                self._to_CS(agent, cs_id)
                self.vehicles[agent].status = action#设置好路线后再更新status
                vehPos = self.connection.vehicle.getLanePosition(agent)
                csPos = self.connection.chargingstation.getEndPos(cs_id)
                logger.info(f"vehicle {agent} choose to charge ,route is set and status is updated to {self.vehicles[agent].status},vehPos:{vehPos},csPos:{csPos}")
                if self.connection.vehicle.isStoppedParking(agent):#如果到了充电站，更新status
                    self.vehicles[agent]._is_stopped = True        
                    self.vehicles[agent].status += self.num_stations
                    self.EV.charging_stations[cs_idx].charging_vehicle.append(agent)
                    logger.info(f"vehicle {agent} is charging at station {cs_id}")
         
    def _to_CS(self, agent, cs_id):
        current_capacity = float(self.connection.vehicle.getParameter(agent, "device.battery.chargeLevel"))
        max_capacity = float(self.connection.vehicle.getParameter(agent, "device.battery.capacity"))
        battery_ratio = current_capacity / max_capacity                 
        target_station_id = cs_id
        station_lane_id = self.connection.chargingstation.getLaneID(target_station_id)
        station_edge_id = self.connection.lane.getEdgeID(station_lane_id)
        vehicle_route = self.connection.vehicle.getRoute(agent)
        origin_destination = vehicle_route[-1]
        if self.is_reset:
            for departure in self.EV.departures:
                if departure[0] == agent:  # 找到对应车辆的departure信息
                    current_edge_id = departure[1]  # 使用departure中的edge_id
                    # logger.debug(f"Vehicle {agent} using departure edge: {current_edge_id}")
                    break
        else:
            current_road = self.connection.vehicle.getRoadID(agent)
            logger.debug(f"current_road{current_road}")
            if current_road.startswith(':'):
                current_lane = self.connection.vehicle.getLaneID(agent)
                links = self.connection.lane.getLinks(current_lane, extended=True)
                logger.debug(f"currentlane:{current_lane},links:{links}")
                if links:
                    next_edge = links[0][0]  # 取第一个连接的出口边
                    current_edge_id = next_edge.split('_')[0]
                else:
                    current_edge_id = self.connection.vehicle.getRoute(agent)[-1]
                logger.debug(f"current_edge_id:{current_edge_id}")
            else:
                current_edge_id = current_road 
        logger.debug(f"To CS, agent {agent} reset: {self.is_reset} Route with battery{battery_ratio}: current_edge {current_edge_id}, station_edge {station_edge_id}, Original destination {origin_destination}")       
        
        route_to_station = self.connection.simulation.findRoute(
            current_edge_id, 
            station_edge_id,
        ).edges
        route_to_destination = self.connection.simulation.findRoute(
            station_edge_id,
            origin_destination,
        ).edges
        new_route = route_to_station[:-1] + route_to_destination
        if current_edge_id != route_to_station[0]:
            route_to_station = [current_edge_id] + route_to_station[1:]
            new_route = route_to_station[:-1] + route_to_destination
        self.connection.vehicle.setRoute(agent, new_route)
        station_end_pos = self.connection.chargingstation.getEndPos(target_station_id)
        self.connection.vehicle.setParkingAreaStop(
            vehID=agent,
            stopID=cs_id,#要求充电站与对应停车场的id相同
            # pos=station_end_pos,
            # laneIndex=0,
            duration= 1999999  # 充电持续时间,不能太大，否则变成浮点数了
        )

    def _get_battery_ratio(self,ev_id):#待改
        current_capacity = float(self.connection.vehicle.getParameter(
            ev_id, "device.battery.chargeLevel"))
        max_capacity = float(self.connection.vehicle.getParameter(
            ev_id, "device.battery.capacity"))
        battery_ratio = current_capacity / max_capacity
        return battery_ratio
    
    def update_vehicle_colors(self):
        vehicles = self.agents
        evs = [v for v in vehicles if self.connection.vehicle.getParameter(
            v, "has.battery.device") == "true"]

        if not LIBSUMO:
            traci.switch(self.label)
            
        for ev_id in evs:
            # 获取电池信息
            current_battery = float(self.connection.vehicle.getParameter(
                ev_id, "device.battery.chargeLevel"))
            max_battery = float(self.connection.vehicle.getParameter(
                ev_id, "device.battery.capacity"))
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

    def _get_observations(self, is_reset):
        self.observations.update(
            {ev_id: ev.get_observation(is_reset) 
             for ev_id, ev in self.vehicles.items()}
        )       
        return self.observations
    

    def _get_rewards(self):
        rewards = {agent: 0.0 for agent in self.agents}
        for ev_id, ev in self.vehicles.items():
            reward = ev.get_rewards()
            rewards[ev_id] = float(reward)
        return rewards

    def _get_infos(self):
        # 统计不同状态的智能体
        truncated_agents = [agent for agent in self.possible_agents if self.truncations[agent]]
        terminated_agents = [agent for agent in self.possible_agents if self.terminations[agent]]
        success_rate = len(terminated_agents) / len(self.possible_agents)
        truncated_rate = len(truncated_agents) / len(self.possible_agents)
        # 计算奖励相关信息
        current_total_reward = sum(self._cumulative_rewards.values())
        mean_reward = current_total_reward / len(self.possible_agents) 
        
        # 计算等待时间
        total_waiting_time = {}
        for agent in self.agents:
            total_waiting_time[agent] = self.connection.vehicle.getAccumulatedWaitingTime(agent)
        max_waiting_time = max(total_waiting_time.values()) if total_waiting_time else 0.0
        mean_waiting_time = sum(total_waiting_time.values()) / len(self.agents) if self.agents else 0.0
        
        
        self.infos.update(
            {
                "success_rate": success_rate,
                "max_waiting_time": max_waiting_time,
                "mean_waiting_time": mean_waiting_time,
                "truncated_rate": truncated_rate,
                "mean_reward": mean_reward,                
            }
        )
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
        
    def observe(self, agent):
        obs = self.observations[agent].copy()
        return obs
        
    def save_csv(self, output_file):
        if output_file is not None and hasattr(self, 'total_metrics') and self.total_metrics:
            # 只保存新的数据
            new_metrics = self.total_metrics[self.last_saved_episode:]
            if not new_metrics:
                return
                
            Path(Path(output_file).parent).mkdir(parents=True, exist_ok=True)
            file_path = output_file + "_metrics.csv"
            
            # 如果文件不存在，创建新文件并写入表头
            if not Path(file_path).exists():
                pd.DataFrame(new_metrics).to_csv(file_path, index=False)
            else:
                # 如果文件存在，追加数据（不包含表头）
                pd.DataFrame(new_metrics).to_csv(file_path, mode='a', header=False, index=False)
            
            self.last_saved_episode = len(self.total_metrics)
            logger.info(f"指标数据已追加保存至: {file_path}")

    def _update_episode_data(self):
        self.episode_data["simulation_time"] = self.connection.simulation.getTime()
        
        # 统计不同状态的智能体
        truncated_agents = [agent for agent in self.agents if self.truncations[agent]]
        terminated_agents = [agent for agent in self.agents if self.terminations[agent]]
        active_agents = [agent for agent in self.agents if not (self.terminations[agent] or self.truncations[agent])]
        
        # 更新episode数据
        self.episode_data.update({
            "simulation_time": self.connection.simulation.getTime(),
            "task_success": len(terminated_agents) == len(self.possible_agents),
            "completion_rate": len(terminated_agents) / len(self.possible_agents),
            "truncated_rate": len(truncated_agents) / len(self.possible_agents),
            "average_battery": mean([self._get_battery_ratio(agent) for agent in active_agents]) if active_agents else 0.0,
            "mean_reward": mean(self._cumulative_rewards.values()),
        })
        
        # 只在episode结束时记录数据
        if len(active_agents) == 0 or self.connection.simulation.getTime() >= self.sim_max_time:
            self.metrics = [self.episode_data.copy()]
            total_metric = { 
                "episode": self.episode,  # 直接使用self.episode作为episode编号
                "mean_reward": self.episode_data["mean_reward"],
                "completion_rate": self.episode_data["completion_rate"],
                "simulation_time": self.episode_data["simulation_time"],
                "task_success": self.episode_data["task_success"],
                "truncated_rate": self.episode_data["truncated_rate"],
                "average_battery": self.episode_data["average_battery"]
            }
            self.total_metrics.append(total_metric)
            self.save_csv(self.output_file)
        
        self.is_reset = False
        
    def get_avail_agent_actions(self, agent): 
        """
        情况1.当车辆处于充电状态,可选动作只能为当前充电站
        情况2.当车辆处于某一充电站的车道上时,可选动作只能为该充电站
        available_actions里存放idx
        """       
        # 情况1.正在充电
        available_actions = []
        if self.num_stations <= self.vehicles[agent].status <2* self.num_stations:
            available_actions = self.vehicles[agent].status - self.num_stations
            return [available_actions]
        # 情况2.
        if self.vehicles[agent]._locked_cs is not None and self.vehicles[agent]._locked_count == 1:
            "当前车道与a 车道重合，且ation = a，那么下一步可用动作只有a"
            available_actions = self.vehicles[agent]._locked_cs
            return [available_actions]
        #情况3
        if self.vehicles[agent]._missed_cs is not None and self.vehicles[agent]._missed_count == 1:
            missed = self.vehicles[agent]._missed_cs
            available_actions =[i for i in range(self.num_stations) if i != missed]
            self.vehicles[agent]._missed_count = 0#动作已经蒙蔽
            self.vehicles[agent]._missed_cs = None#清除
            return available_actions
        #情况4 完成任务或者任务失败
        if self.terminations[agent] or self.truncations[agent]:
            available_actions = self.num_stations
            return [available_actions]
        #其他情况,所有充电站都可以选择
        
        available_actions.extend(range(self.num_stations))               
        return available_actions
    
    def seed(self, seed=None):
        self._seed = seed
        np.random.seed(seed)
        return [seed]
    





