import xml.etree.ElementTree as ET
import numpy as np
from typing import Dict, Any, List, Tuple
import numpy.typing as npt
import sys

VEHICLE_XML_TAG = "vehicle"
VEHICLE_CAPACITY_TAG = "personNumber"

VERTEX_XML_TAG = "junction"
VERTEX_XML_INVALID_TYPE = "internal"
VERTEX_CUSTOMIZED_PARAM = "param"
VERTEX_DEMAND_KEY = "destination"

EDGE_XML_TAG = "edge"
EDGE_XML_INVALID_FUNC = "internal"#
EDGE_XML_PRIORITY = "-1"#


def get_electric_vehicles(flow_xml_tree):

    # Find all vehicle types
    vtypes = flow_xml_tree.findall("vType")
    ev_type_ids = []
    
    # Find which vehicle types are electric (have battery capacity parameter)
    for vtype in vtypes:
        vtype_params = vtype.findall("param")
        for vtype_param in vtype_params:
            if vtype_param.attrib["key"] == "has.battery.device" and vtype_param.attrib["value"] == "true":
                ev_type_ids.append(vtype.attrib["id"])
                break

    # Find all vehicles of electric types
    ev_lst = []
    vehicles = flow_xml_tree.findall(VEHICLE_XML_TAG)
    for vehicle in vehicles:
        if vehicle.attrib["type"] in ev_type_ids:
            ev_lst.append(vehicle.attrib["id"])
            
    return ev_lst

def get_charging_stations(additional_xml_tree, net_xml_tree):
    """
    Helper function for decode_xml_fmp


    Returns charging stations

    Each charging station is [id, (x_coord, y_coord), edge_id, charging speed]
    """
    cs_lst = []
    stations = additional_xml_tree.findall("chargingStation")
    for station in stations:

        if "shadow" not in station.attrib["id"]:

            # get approximate location
            # x_coord, y_coord = station.findall("param")[0].attrib["value"].split()#charging station approximate location(自己加的属性)
            # get edge_id
            lane_id = station.attrib["lane"]
            edge_id = None
            # get all edges in net.xml
            edges = net_xml_tree.findall("edge")
            for edge in edges:
                lanes = edge.findall("lane")
                lanes = [
                    lane.attrib["id"] for lane in lanes if lane.attrib["id"] == lane_id
                ]
                if len(lanes) == 1:
                    edge_id = edge.attrib["id"]#返回charging station 所在边的edge id#实际上edge_id 就是lane_id,不用检索
                    break
            cs_lst.append(
                (
                    station.attrib["id"],
                    # (float(x_coord), float(y_coord)),
                    edge_id,
                    float(station.attrib["power"]),
                    float(station.attrib["endPos"]),
                )
            )

    return cs_lst


def get_vertices(net_xml_tree):
    """
    Helper function for decode_xml_fmp

    Returns vertices
    Each vertex is [id (str), x_coord (float), y_coord (float)]
    """
    vtx_lst = []
    junctions = net_xml_tree.findall(VERTEX_XML_TAG)

    for junction in junctions:
        if junction.attrib["type"] == VERTEX_XML_INVALID_TYPE:
            continue
        vtx_lst.append(
            [
                junction.attrib["id"],
                float(junction.attrib["x"]),
                float(junction.attrib["y"]),
            ]
        )

    return vtx_lst


def get_edges(net_xml_tree):
    """
    Helper function for decode_xml_fmp

    Returns edges

    Each edge is [id (str), from_vertex_id (str),
                  to_vertex_id (str),
                  edge_length (float)]
    """
    edge_lst = []
    edges = net_xml_tree.findall(EDGE_XML_TAG)
    for e in edges:
        if "function" in e.attrib and e.attrib["function"] == EDGE_XML_INVALID_FUNC:
            continue
        # Edge lengths are given by lane lengths. Each edge has at
        # least one lane and all lanes of an edge have
        # the same length
        lane = e.findall("lane")[0]
        edge_lst.append(
            [
                e.attrib["id"],
                e.attrib["from"],
                e.attrib["to"],
                float(lane.attrib["length"]),
            ]
        )
    return edge_lst


def get_departures(flow_xml_tree):
    """
    Helper function for decode_xml_fmp

    Returns departures for each vehicle

    Each departure is [vehicle_id, starting_edge_id]
    and should be defined for all vehicles
    """
    departures = []  # id, start_index tuple
    for vehicle in flow_xml_tree.findall(VEHICLE_XML_TAG):
        route = vehicle.findall("route")[0]  # findall should return one
        edges = route.attrib["edges"]  # space separated list of edge ids
        start_edge = edges.split()[0]
        departures.append([vehicle.attrib["id"], start_edge])

    return departures


def get_demand(net_xml_tree):
    """
    Some junctions are customer nodes and this is represented by
    a junction having param destination
    Each demand is [junction_id, dest_vertex_id, departure_edge_id, destination_edge_id]
    """
    # 首先获取所有非internal类型的edge
    valid_edges = {}
    for edge in net_xml_tree.findall(EDGE_XML_TAG):
        if "function" not in edge.attrib or edge.attrib["function"] != EDGE_XML_INVALID_FUNC:
            valid_edges[edge.attrib["id"]] = {
                "from": edge.attrib["from"],
                "to": edge.attrib["to"]
            }
    
    demand = []
    for junction in net_xml_tree.findall(VERTEX_XML_TAG):
        if junction.get("type") != VERTEX_XML_INVALID_TYPE:
            junction_id = junction.get("id")
            for customized_params in junction.findall(VERTEX_CUSTOMIZED_PARAM):
                if customized_params.get("key") == VERTEX_DEMAND_KEY:
                    # 获取incLanes属性，它包含了进入该junction的lanes
                    inc_lanes = junction.get("incLanes", "")
                    # 从incLanes中提取edge ids
                    departure_edge_id = None
                    destination_edge_id = None
                    
                    if inc_lanes:
                        # incLanes格式为"edge_id_lane_index edge_id_lane_index ..."，例如"gneE18_0 234726492#3_0"
                        for lane in inc_lanes.split():
                            # 提取edge id，例如从"gneE18_0"提取"gneE18"
                            potential_edge_id = lane.split("_")[0]
                            
                            # 检查是否是有效的edge（非internal类型）
                            if potential_edge_id in valid_edges:
                                edge_info = valid_edges[potential_edge_id]
                                # 检查edge的终点是否是当前junction
                                if edge_info["to"] == junction_id:
                                    departure_edge_id = potential_edge_id
                                    break
                        
                        # 如果找不到终点是当前junction的非internal类型edge，尝试使用起点是当前junction的非internal类型edge
                        if departure_edge_id is None:
                            for lane in inc_lanes.split():
                                potential_edge_id = lane.split("_")[0]
                                if potential_edge_id in valid_edges:
                                    edge_info = valid_edges[potential_edge_id]
                                    # 检查edge的起点是否是当前junction
                                    if edge_info["from"] == junction_id:
                                        departure_edge_id = potential_edge_id
                                        break
                    
                    # 获取destination的edge id
                    dest_junction_id = customized_params.get("value")
                    if dest_junction_id:
                        dest_junction = next((j for j in net_xml_tree.findall(VERTEX_XML_TAG) 
                                           if j.get("id") == dest_junction_id), None)
                        if dest_junction:
                            dest_inc_lanes = dest_junction.get("incLanes", "")
                            if dest_inc_lanes:
                                for lane in dest_inc_lanes.split():
                                    potential_edge_id = lane.split("_")[0]
                                    if potential_edge_id in valid_edges:
                                        edge_info = valid_edges[potential_edge_id]
                                        if edge_info["to"] == dest_junction_id:
                                            destination_edge_id = potential_edge_id
                                            break
                                
                                if destination_edge_id is None:
                                    for lane in dest_inc_lanes.split():
                                        potential_edge_id = lane.split("_")[0]
                                        if potential_edge_id in valid_edges:
                                            edge_info = valid_edges[potential_edge_id]
                                            if edge_info["from"] == dest_junction_id:
                                                destination_edge_id = potential_edge_id
                                                break
                    
                    demand.append([
                        junction_id, 
                        customized_params.get("value"),
                        departure_edge_id,
                        destination_edge_id
                    ])

    return demand


def decode_xml_fmp(
    net_xml_file_path: str = None,
    flow_xml_file_path: str = None,
    additional_xml_path: str = None,
):
    """
    Parse files generated from SUMO and return a FMP instance

    Returns vertices, charging_stations, electric_vehicles,
    edges, departures, and demands
    """
    net_xml_tree = ET.parse(net_xml_file_path)
    flow_xml_tree = ET.parse(flow_xml_file_path)
    additional_xml_tree = ET.parse(additional_xml_path)
    vertices = get_vertices(net_xml_tree)
    charging_stations = get_charging_stations(additional_xml_tree, net_xml_tree)
    electric_vehicles = get_electric_vehicles(flow_xml_tree)
    edges = get_edges(net_xml_tree)
    departures = get_departures(flow_xml_tree)
    demand = get_demand(net_xml_tree)
    return vertices, charging_stations, electric_vehicles, edges, departures, demand


def decode_xml(
    net_xml_file_path: str = None, flow_xml_file_path: str = None
) -> Tuple[npt.NDArray[Any]]:
    """
    Parse the net.xml and rou.xml generated from SUMO and read into VRP initialization environments.
    Return objects: vertices, demand, edges, departures, capacity for VRP class
    """

    net_xml_source = open(net_xml_file_path)
    flow_xml_source = open(flow_xml_file_path)

    vertices, demand, edge_id_map, edges = _parse_network_xml(net_xml_source)
    departures, capacity = _parse_flow_xml(flow_xml_source, edge_id_map, edges)

    net_xml_source.close()
    flow_xml_source.close()

    return (
        np.asarray(vertices),
        np.asarray(demand),
        np.asarray(edges),
        np.asarray(departures),
        np.asarray(capacity),
    )


def _parse_flow_xml(flow_file_path: str, edge_id_map: Dict[str, int], edges: Any):
    """
    :param flow_file_path:      file path of rou.xml
    :param edge_id_map:         sample structure: {'genE0': 0, 'genE1': 1}
    :param edges:               tuple of [edge_from_vertex, edge_to_vertex], sample structure: [[0,4], [1,3], [7,5]]
    """
    flow_tree = ET.parse(flow_file_path)
    flow_xml_root = flow_tree.getroot()

    departures = []  # int array
    capacity = []  # float array
    for vehicle_trip in flow_xml_root.findall(VEHICLE_XML_TAG):
        departure_edge = vehicle_trip.get("from")
        departures.append(edges[edge_id_map[departure_edge]][0])

        capacity_value = vehicle_trip.get(VEHICLE_CAPACITY_TAG) or 20.0
        capacity.append(float(capacity_value))

    return departures, capacity


def _parse_network_xml(network_file_path: str):
    """
    :param network_file_path:     file path of net.xml
    """
    network_tree = ET.parse(network_file_path)
    network_xml_data = network_tree.getroot()

    vertices_id_map = {}  # sample structure: {'genJ1': 0, 'genJ10': 1}交叉点id-count索引
    vertices = []  # tuple of x,y position of each vertex
    demand = []  # float array
    vertex_count = 0
    for junction in network_xml_data.findall(VERTEX_XML_TAG):
        if junction.get("type") != VERTEX_XML_INVALID_TYPE:#记录不是internal的junction
            vertices.append([float(junction.get("x")), float(junction.get("y"))])
            vertices_id_map[junction.get("id")] = vertex_count

            demand_value = 0.0
            for customized_params in junction.findall(VERTEX_CUSTOMIZED_PARAM):
                if customized_params.get("key") == VERTEX_DEMAND_KEY:
                    demand_value = float(customized_params.get("value"))#待改，应该是整型
            demand.append(demand_value)#有dest,保存dest,没有destination，则保存0

            vertex_count += 1

    edge_id_map = {}  # sample structure: {'genE0': 0, 'genE1': 1}，priority =-1的(edge_id,count)
    edges = (
        []
    )  # tuple of [edge_from_vertex, edge_to_vertex], sample structure: [[0,4], [1,3], [7,5]]
    edge_count = 0
    for edge in network_xml_data.findall(EDGE_XML_TAG):
        if edge.get("priority") == EDGE_XML_PRIORITY:#priority = -1的边
            edges.append(
                [vertices_id_map[edge.get("from")], vertices_id_map[edge.get("to")]]
            )
            edge_id_map[edge.get("id")] = edge_count
            edge_count += 1

    return vertices, demand, edge_id_map, edges
