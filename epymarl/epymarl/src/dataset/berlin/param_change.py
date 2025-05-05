import xml.etree.ElementTree as ET
from xml.dom import minidom

# 读取并解析 XML
tree = ET.parse('src/dataset/berlin/berlin_ev.rou.xml')  # 原始文件路径
root = tree.getroot()

# 遍历所有 <vehicle> 元素
for vehicle in root.findall('vehicle'):
    vehicle.set('depart', '0.00')  # 将 depart 改为 0

    # 查找其子元素 <route>
    route = vehicle.find('route')
    if route is not None:
        # 在 route 后添加 <param key="actualBatteryCapacity" value="-1"/>
        param = ET.Element('param')
        param.set('key', 'actualBatteryCapacity')
        param.set('value', '-1')
        vehicle.insert(list(vehicle).index(route) + 1, param)

# 使用 minidom 美化格式
rough_string = ET.tostring(root, 'utf-8')
reparsed = minidom.parseString(rough_string)

# 去除多余空行再写入文件
pretty_xml_as_string = '\n'.join(
    [line for line in reparsed.toprettyxml(indent="  ").split('\n') if line.strip()]
)

with open('src/dataset/berlin/berlin_ev0.rou.xml', 'w', encoding='utf-8') as f:
    f.write('<?xml version="1.0" ?>\n' + pretty_xml_as_string)