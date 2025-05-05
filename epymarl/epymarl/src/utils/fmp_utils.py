from sklearn.cluster import KMeans
import math

import src.utils.network_utils as network_utils
import numpy as np
from bisect import bisect
from collections import deque

NO_LOADING = -1
NO_CHARGING = -1
CHARGING_STATION_LENGTH = 5
IDLE_LOCATION = -1

K_MEANS_ITERATION = 10

NEAREST_CS = True
FURTHEST_CS = False

class Vertex(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.area = -1

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"vertex ({self.x}, {self.y}, {self.area})"#area 为 -1

class Edge(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return (self.start, self.end) == (other.start, other.end)

    def __lt__(self, other):
        return (self.start, self.end) < (other.start, other.end)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"edge ({self.start}, {self.end})"
    
class ChargingStation(object):
    def __init__(
        self, location, indicator, charging_speed, n_slot=None, charging_vehicle=None
    ):
        self.location = location
        self.indicator = indicator
        self.charging_speed = charging_speed
        self.n_slot = n_slot
        self.charging_vehicle = charging_vehicle

    def __eq__(self, other):
        return self.location == other.location

    def __lt__(self, other):
        return self.location < other.location

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"ChargingStation ({self.location}, {self.indicator}, {self.charging_speed})"

class ElectricVehicles(object):
    def __init__(self, id):
        self.id = id
    
class Demand(object):
    def __init__(self, departure, destination, departure_edge_id=None, destination_edge_id=None):
        self.departure = departure
        self.destination = destination
        self.departure_edge_id = departure_edge_id  # 与departure关联的edge id
        self.destination_edge_id = destination_edge_id  # 与destination关联的edge id

    def __eq__(self, other):
        return (self.departure, self.destination) == (
            other.departure,
            other.destination,
        )

    def __lt__(self, other):
        return (self.departure, other.destination) < (other.departure, other.destination)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"demand ({self.departure}, {self.destination}, departure_edge: {self.departure_edge_id}, destination_edge: {self.destination_edge_id})"
    
class Loading(object):
    def __init__(self, current=-1, target=-1):
        self.current = current
        self.target = target

    def __repr__(self):
        return f"(responding {self.current}, goto respond {self.target})"


class Charging(object):
    def __init__(self, current=-1, target=-1):
        self.current = current
        self.target = target

    def __repr__(self):
        return f"(charging {self.current}, go to charge {self.target})"


class GridAction(object):
    def __init__(self, state=None):
        self.is_loading = state.is_loading
        self.is_charging = state.is_charging
        self.location = state.location

    def __repr__(self):
        return f"({self.is_loading}, {self.is_charging}, location {self.location})"


class Metrics(object):
    def __init__(self):
        self.task_finish_time = 0
        self.total_battery_consume = 0
        self.charge_waiting_time = 0
        self.respond_failing_time = 0

    def __repr__(self):
        return f"(tft {self.task_finish_time}, tbc {self.total_battery_consume}, cwt {self.charge_waiting_time}, rft {self.respond_failing_time})"
    
    


def convert_raw_vertices(raw_vertices):#列表（junction_id, 坐标x,坐标y）
    """
    Each raw vertex is [id (str), x_coord (float), y_coord (float)]
    """
    vertices = []
    vertex_dict = {}  # vertex id in SUMO to idx in vertices
    for counter, v in enumerate(raw_vertices):#counter 代表当前元素在raw_vertices中的索引,v 代表当前元素（一个[junction, x, y] 列表）
        vertices.append(Vertex(v[1], v[2]))#创建 Vertex 对象，并将其添加到 vertices 列表中
        '''
        [   Vertex(10.5, 20.3),
            Vertex(15.2, 25.7),
            Vertex(18.0, 30.1),]
        '''
        vertex_dict[v[0]] = counter
        '''
        {   "J1": 0,
            "J2": 1,
            "J3": 2}
        '''
    return vertices, vertex_dict
    #vertices：存储所有 Vertex 对象的列表。
    #vertex_dict：映射 SUMO ID 到 vertices 索引的字典

def convert_raw_edges(raw_edges, vertex_dict):#列表（edge边id,junction始号，junction终号，边长）,vertex_dict
    """
    Each raw edge is
    [id (str), from_vertex_id (str), to_vertex_id (str), edge_length (float)]
    """
    edges = []
    edge_dict = {}  # sumo edge_id to idx in edges
    edge_length_dict = {}  # sumo edge_id to length
    for counter, e in enumerate(raw_edges):#counter 是索引，e 是 raw_edges 中的每个元素（边的信息
        new_edge = Edge(vertex_dict[e[1]], vertex_dict[e[2]])
        edges.append(new_edge)
        edge_dict[e[0]] = counter
        edge_length_dict[e[0]] = e[3]
    return edges, edge_dict, edge_length_dict


def euclidean_distance(start_x, start_y, end_x, end_y):
    """
    Compute euclidean distance between (start_x, start_y)
    and (end_x, end_y)
    """
    return (((start_x - end_x) ** 2) + ((start_y - end_y) ** 2)) ** 0.5


def convert_raw_charging_stations(raw_charging_stations):
    
    charging_station_dict = {}  # idx in charging_stations to sumo id
    charging_stations = []

    for counter, charging_station in enumerate(raw_charging_stations):
        # 只保存充电站ID映射
        charging_station_dict[counter] = charging_station[0]
        
        # 创建ChargingStation对象，使用原始edge_id作为location
        charging_stations.append(ChargingStation(
            location=charging_station[2],  # 使用原始edge_id
            indicator=220,
            charging_speed=charging_station[3]
        ))

    return charging_stations, charging_station_dict




#     return electric_vehicles, ev_dict
def convert_raw_electric_vehicles(raw_electric_vehicles):
    electric_vehicles = []
    ev_dict = {}  # ev sumo id to idx in electric_vehicles
    
    for counter, vehicle_id in enumerate(raw_electric_vehicles):
        electric_vehicles.append(ElectricVehicles(vehicle_id))
        ev_dict[vehicle_id] = counter

    return electric_vehicles, ev_dict


def convert_raw_departures(raw_departures):
    """
    Each raw departure is [vehicle_id, starting_edge_id]
    """
    return raw_departures


def convert_raw_demand(raw_demand, vertex_dict):
    """
    Each raw demand is [junction_id, dest_vertex_id, departure_edge_id, destination_edge_id]
    """
    demand = []
    for d in raw_demand:
        departure_vertex = vertex_dict[d[0]]
        destination_vertex = vertex_dict[d[1]]
        departure_edge_id = d[2] if len(d) > 2 else None
        destination_edge_id = d[3] if len(d) > 3 else None
        
        demand.append(Demand(
            departure_vertex, 
            destination_vertex,
            departure_edge_id,
            destination_edge_id
        ))
    return demand


def one_step_to_destination(vertices, edges, start_index, dest_index):
    if start_index == dest_index:
        return dest_index
    visited = [False] * len(vertices)
    bfs_queue = [dest_index]
    visited[dest_index] = True

    while bfs_queue:
        curr = bfs_queue.pop(0)
        adjacent_map = network_utils.get_adj_from_list(vertices, edges)

        for v in adjacent_map[curr]:
            if not visited[v] and v == start_index:
                return curr
            elif not visited[v]:
                bfs_queue.append(v)
                visited[v] = False


# def dist_between(vertices, edges, start_index, dest_index):#可能存在逻辑错误
#     if start_index == dest_index:
#         return 0
#     visited = [False] * len(vertices)
#     bfs_queue = [[start_index, 0]]
#     visited[start_index] = True
#     while bfs_queue:
#         curr, curr_depth = bfs_queue.pop(0)
#         adjacent_map = network_utils.get_adj_to_list(vertices, edges)

#         for v in adjacent_map[curr]:
#             if not visited[v] and v == dest_index:
#                 return curr_depth + 1
#             elif not visited[v]:
#                 bfs_queue.append([v, curr_depth + 1])
#                 visited[v] = False
def dist_between(vertices, edges, start_index, dest_index):
    if start_index == dest_index:
        return 0  #  起点 == 终点，返回 0

    adjacent_map = network_utils.get_adj_to_list(vertices, edges)  #  预计算邻接表
    if start_index not in adjacent_map or dest_index not in adjacent_map:
        return float("inf")  #  起点或终点不在图中，直接返回 ∞

    visited = set()  #  用集合代替列表，提高查找效率
    bfs_queue = deque([(start_index, 0)])  #  BFS 队列：[当前节点, 累计距离]
    visited.add(start_index)

    while bfs_queue:
        curr, curr_dist = bfs_queue.popleft()  #  O(1) 出队

        for neighbor in adjacent_map.get(curr, []):  #  避免 KeyError
            if neighbor == dest_index:
                return curr_dist + 1  #  找到目标，返回最短路径长度

            if neighbor not in visited:
                bfs_queue.append((neighbor, curr_dist + 1))
                visited.add(neighbor)  #  标记访问，防止死循环

    return float("inf")  #  目标不可达，返回 ∞


def get_hot_spot_weight(vertices, edges, demands, demand_start):
    adjacent_vertices = np.append(
        network_utils.get_adj_to_list(vertices, edges)[demand_start], demand_start
    )
    local_demands = len([d for d in demands if d.departure in adjacent_vertices])

    return local_demands / len(demands) * 100


# k as number of clusters, i.e., count of divided areas 定义一个函数 cluster_as_area，它的作用是将 vertices 按照 (x, y) 坐标进行 K-Means 聚类，将每个 vertex 的 area 属性设置为聚类的结果
def cluster_as_area(vertices, k):
    vertices_loc = [[v.x, v.y] for v in vertices]#提取 vertices 列表中每个点的 (x, y) 坐标，并将其转换为二维列表 vertices_loc
    kmeans = KMeans(
        n_clusters=k,
        init=np.asarray(_generate_initial_cluster(vertices_loc, k)),
        random_state=0,
    ).fit(vertices_loc)
    for i, v in enumerate(vertices):
        v.area = kmeans.labels_[i]

    return vertices


# get the current safe indicator#电池电量与demands博弈
def get_safe_indicator(vertices, edges, demands, charging_stations, location, battery):#待改，可优化
    dist_to_furthest_cs = max(
        get_dist_to_charging_stations(vertices, edges, charging_stations, location)
    )
    dist_to_finish_demands = get_dist_to_finish_demands(
        vertices, edges, demands, location
    )
    if battery <= min(dist_to_finish_demands) + dist_to_furthest_cs:
        return 0
    elif battery <= max(dist_to_finish_demands) + dist_to_furthest_cs:
        return 1
    else:
        return 2


# get the dist to finish all demands from the current location
def get_dist_to_finish_demands(vertices, edges, demands, start_index):
    dist_of_demands = get_dist_of_demands(vertices, edges, demands)
    return [
        dist_of_demands[i] + dist_between(vertices, edges, start_index, d.departure)
        for i, d in enumerate(demands)
    ]


# get the travel dist of demand
def get_dist_of_demands(vertices, edges, demands):
    return [dist_between(vertices, edges, d.departure, d.destination) for d in demands]


# get the dist to all cs from the current location
def get_dist_to_charging_stations(vertices, edges, charging_stations, start_index):
    return [
        dist_between(vertices, edges, start_index, cs.location)
        for cs in charging_stations
    ]


# roughly divide the map into a root x root grid map as initialization
def _generate_initial_cluster(vertices_loc, k):
    initial_clusters = []
    root = int(math.sqrt(k))

    x_sorted = sorted(vertices_loc, key=lambda x: x[0])#按 x 坐标 从小到大 排序 vertices_loc，存入 x_sorted
    x_start = x_sorted[0][0]
    x_step = (x_sorted[-1][0] - x_sorted[0][0]) / (root + 1)

    y_sorted = sorted(vertices_loc, key=lambda x: x[1])
    y_start = y_sorted[0][1]
    y_step = (y_sorted[-1][1] - y_sorted[0][1]) / (root + 1)
    for i in range(root):
        for j in range(root):
            initial_clusters.append(
                [x_start + (i + 1) * x_step, y_start + (j + 1) * y_step]
            )

    return initial_clusters