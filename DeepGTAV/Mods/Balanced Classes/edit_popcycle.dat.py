import os
from itertools import chain
import re

from utils.BoundingBoxes import getLabelFromObjectName
import lxml.etree as etree

# There are:

# VEH_AIRPORT
# VEH_ARMY
# VEH_ARMY_MP
# VEH_BEACH_BIKE
# VEH_BEACH_BIKE_MP
# VEH_BIKES
# VEH_BIKES_MP
# VEH_BOATS
# VEH_BOATS_FREEWAY
# VEH_BOATS_FREEWAY_MP
# VEH_BOATS_MP
# VEH_COPCAR
# VEH_COUNTRYCOACH
# VEH_COUNTRYSIDE_OFFROAD
# VEH_COUNTRYSIDE_OFFROAD_MP
# VEH_COUNTRYSIDE_ONROAD
# VEH_COUNTRYSIDE_ONROAD_MP
# VEH_FREEWAY
# VEH_FREEWAY_MP
# VEH_HAULAGE
# VEH_HAULAGE_MP
# VEH_LARGE_CITY
# VEH_LARGE_CITY_MP
# VEH_LOST
# VEH_LOST_MP
# VEH_MID
# VEH_MID_MP
# VEH_POOR
# VEH_POOR_MP
# VEH_RICH
# VEH_RICH_MP
# VEH_SALT
# VEH_SALT_MP
# VEH_TAXI
# VEH_TRANSPORT
# VEH_TRANSPORT_MP
# VEH_UTILITY
# VEH_UTILITY_MP
# VEH_YANKTON

# + MP
# VEH_AIRPORT, VEH_COUNTRYCOACH, VEH_BOATS


# It seems the sum of VEH_... is 100
# and the sum of Peds is 200


# POPCYCLE_PATH = "E:\\Google Drive Meine Ablage\\Studium\\SS 20\Bachelorarbeit\\Scripts_TO_MOVE\\Popcycle.dat Editing\\popcycle.dat"
POPCYCLE_PATH = "C:\\Users\\Surfer\\Google Drive Meine Ablage\\Studium\\SS 20\\Bachelorarbeit\\Scripts_TO_MOVE\\Popcycle.dat Editing\\popcycle.dat"


def find_all_names(popcycle_path):
    with open(popcycle_path, 'r') as file:
        txt = file.read()
    txt = txt.split("\n")
    txt = [t.split(" ") for t in txt]
    txt = list(set(chain(*txt)))
    txt.sort()
    txt = "\n".join(txt)
    # txt = set(txt)
    return(txt)

# names = find_all_names("C:\\Users\\Surfer\\Google Drive Meine Ablage\\Studium\\SS 20\\Bachelorarbeit\\Scripts_TO_MOVE\\Popcycle.dat Editing\\popcycle.dat")
# print(names)


def find_all_pedtypes(popcycle_path):
    with open(popcycle_path, 'r') as file:
        txt = file.read()
    txt = txt.split("\n")
    txt_new = []
    for t in txt:
        if "peds" in t and "cars" in t:
            t = t.split("peds")[1].split("cars")[0]
            t = t.split(" ")
            t = [i for i in t if not i.isdigit()]
            txt_new = txt_new + t
    txt = list(set(txt_new))
    txt.sort()
    txt = "\n".join(txt)
    # txt = set(txt)
    return(txt)



# names = find_all_pedtypes("C:\\Users\\Surfer\\Google Drive Meine Ablage\\Studium\\SS 20\\Bachelorarbeit\\Scripts_TO_MOVE\\Popcycle.dat Editing\\popcycle.dat")
# names = find_all_pedtypes(POPCYCLE_PATH)
# print(names)

# Edit Beginning
# //  #Peds  #Scenario #Cars  #prkdcrs #lowprkdcrs    PercCopCars PercCopPed  MaxScenPeds MaxScenVehs MaxPreAssignedParked
#       50     100        85      20         30           1           1           3          4                   1           

# Edit VEH:
# search after " cars "
# Make a balanced composition and replace


# Edit Pedestrian types
# search after " peds ", before " cars "
# make a balanced standard composition?

def edit_popcycle(popcycle_path, constpeds = 50, constscenario = 100, perccopcars = 1, perccopped = 1): #, VEH_BIKES, VEH_TRANSPORT):
    with open(popcycle_path, 'r') as file:
        txt = file.read()
    txt = txt.split("\n")
    
    def applyreg(t):
        t = re.sub("\d+", str(constscenario), t, 2)
        t = re.sub("\d+", str(constpeds), t, 1)
        return t
    txt = [applyreg(t) for t in txt]
    txt = "\n".join(txt)
    with open(popcycle_path, 'w') as file:
        file.write(txt)

# POPCYCLE_PATH = "C:\\Users\\Surfer\\Google Drive Meine Ablage\\Studium\\SS 20\\Bachelorarbeit\\Scripts_TO_MOVE\\Popcycle.dat Editing\\popcycle.dat"
# edit_popcycle(POPCYCLE_PATH, constpeds=1000,constscenario=1000)


NORMSTART = "      1000     1000        85      20         30           1           1           3          4                   1           "
NORMPEDS = "peds  DeepGTAV_peds 200  "
# NORMVEH = "cars  VEH_BIKES 100"
# NORMVEH = "cars  VEH_BEACH_BIKE 25  VEH_BEACH_BIKE_MP 25  VEH_BIKES 25  VEH_BIKES_MP 25"
NORMVEH = "cars  VEH_DEEPGTAV_CAR 20  VEH_DEEPGTAV_VAN 20  VEH_DEEPGTAV_TRUCK 20  VEH_DEEPGTAV_BUS 20  VEH_DEEPGTAV_MOTOR 20"

NORMLINE = NORMSTART + NORMPEDS + NORMVEH

def edit_popcycle_cars(popcycle_path, normcarstring = NORMVEH):
    pass



def edit_popcycle_replace_full_line(popcycle_path, normline = NORMLINE):
    with open(popcycle_path, 'r') as file:
        txt = file.read()
    txt = txt.split("\n")
    
    def applyreg(t):
        if re.search("\d+", t) != None:
            t = normline
        return t
    txt = [applyreg(t) for t in txt]
    txt = "\n".join(txt)
    with open(popcycle_path, 'w') as file:
        file.write(txt)



edit_popcycle_replace_full_line(POPCYCLE_PATH)






# XML_PATH = "E:\\Google Drive Meine Ablage\\Studium\\SS 20\Bachelorarbeit\\Scripts_TO_MOVE\\Popcycle.dat Editing\\popgroups.ymt"
XML_PATH = "C:\\Users\\Surfer\\Google Drive Meine Ablage\\Studium\\SS 20\\Bachelorarbeit\\Scripts_TO_MOVE\\Popcycle.dat Editing\\popgroups.ymt"
def load_xml(xml_path = XML_PATH):
    tree = etree.parse(xml_path)
    return tree

tree = load_xml()
root = tree.getroot()
depth = 0
pedTypes = []
vehTypes = []
# print(depth*" ", root.tag, root.attrib)
for c1 in tree.getroot():
    depth=1
    # print(depth * " ", c1.tag, c1.attrib, c1.text)
    if c1.tag == "pedGroups":
        for c2 in c1:
            depth=2
            # print(depth * " ", c2.tag, c2.attrib, c2.text)
            for c3 in c2:
                depth = 3
                # print(depth*" ", c3.tag, c3.attrib, c3.text)
                for c4 in c3:
                    depth=4
                    # print(depth*" ", c4.tag, c4.attrib, c4.text)
                    for c5 in c4:
                        depth = 5
                        # print(depth*" ", c5.tag, c5.attrib, c5.text)
                        if c5.tag == "Name":
                            pedTypes.append(c5.text)
    if c1.tag == "vehGroups":
        for c2 in c1:
            depth=2
            # print(depth * " ", c2.tag, c2.attrib, c2.text)
            for c3 in c2:
                depth = 3
                # print(depth*" ", c3.tag, c3.attrib, c3.text)
                for c4 in c3:
                    depth=4
                    # print(depth*" ", c4.tag, c4.attrib, c4.text)
                    for c5 in c4:
                        depth = 5
                        # print(depth*" ", c5.tag, c5.attrib, c5.text)
                        if c5.tag == "Name":
                            vehTypes.append(c5.text)

pedTypes = set(pedTypes)
vehTypes = set(vehTypes)


pedestrian = []
people = []
bicycle = []
car = []
van = []
truck = []
bus = []
motor = []

for veh in vehTypes:
    lab = getLabelFromObjectName(veh)
    if lab == "car":
        car.append(veh)
    if lab == "van":
        van.append(veh)
    if lab == "truck":
        truck.append(veh)
    if lab == "bus":
        bus.append(veh)
    if lab == "motor":
        motor.append(veh)

car
van
truck
bus
motor

def makeXML(names, categoryname):
    txt = ""
    txt = "<Item>\n"
    txt = txt + "  <Name>" + categoryname + "</Name>\n"
    txt = txt + "  <models>\n"
    depth = 2
    for name in names:
        txt = txt + depth*"  " + "<Item>\n"
        txt = txt + (depth+1)*"  " + "<Name>" + name + "</Name>\n"
        txt = txt + (depth+1)*"  " + '<Variations type="NULL"/>\n'
        txt = txt + depth*"  " + "</Item>\n"
    txt = txt + "  </models>\n"
    txt = txt + "  <flags>POPGROUP_SCENARIO POPGROUP_AMBIENT</flags>\n"
    txt = txt + "</Item>"
    return txt


txt = makeXML(car, "VEH_DEEPGTAV_CAR") + "\n"
txt += makeXML(van, "VEH_DEEPGTAV_VAN") + "\n"
txt += makeXML(truck, "VEH_DEEPGTAV_TRUCK") + "\n"
txt += makeXML(bus, "VEH_DEEPGTAV_BUS") + "\n"
txt += makeXML(motor, "VEH_DEEPGTAV_MOTOR") + "\n"

print(txt)


txt = makeXML(pedTypes, "DeepGTAV_peds")
print(txt)