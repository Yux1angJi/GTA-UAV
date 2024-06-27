# Install Instructions for Mods

In this Readme the the installation instructions for the Modifications that I have made to GTAV to improve the capturing process are listed. Those modifications should easily work with GTAV version 1.0.2060.1. 
If you face difficulties with newer versions of GTAV, try to find newer versions of the mods listed here. 

In general there are mostly two different ways of mod installations. Either the scripts have to be copied to the `GTAV/` root folder, or the game files have to be overwritten usign `OpenIV` (https://openiv.com/). 
For those installations where `OpenIV` is required I recommend using the dedicated `mods` folder. Detailed instructions for the use of `OpenIV` can be found online (https://www.lcpdfr.com/tutorials/index.action/grand-theft-auto-v/42_script-data-file-modification-tutorials/how-to-install-openivasi-r127/).

## No chromatic aberration lens distortion
This mod removes the post processing effects for chromatic aberration and lens distortion. Only with this modifications do the bounding boxes at the edges of the screen fit correctly.
https://de.gta5-mods.com/misc/no-chromatic-aberration-lens-distortion-1-41

## Heap Limit Adjuster
This mod increases the heap size of GTAV used. The files can be found here (https://de.gta5-mods.com/tools/heap-limit-adjuster-600-mb-of-heap). 
To install, copy the file `GTAV.HeapAdjuster.asi` into the GTAV root directory. 


## Simple Increase Traffic and Pedestrian
This mod increases the number of spawned objects (pedestrians, vehicles and animals). The files can be found at (https://de.gta5-mods.com/misc/simple-increase-traffic-and-pedestrian).
To install, run `x10 (Heavy).oiv` using OpenIV


## Balanced Classes of spawned objects

### Installation
Install this modificatino after `Simple Increase Traffic and Pedestrian`
To install this modification use `OpenIV` to overwrite 

    update/update.rpf/common/data/levels/gta5/popcycle.dat

and 

    update/update.rpf/x64/levels/gta5/popgroups.ymt

with the ones in 

    Mods/Balanced Classes/Popcycle.dat Editing/Overwrite

### In depth explanation
GTAV basically uses two files to manage the spawned objects: `popgroups.ymt` and `popcycle.dat`. 
`popgroups.ymt` defines population groups, which are groups of different objects that are spawned in the same way. 
An example of a population group caleld `VEH_DEEPGTAV_BUS` containing the ingame objects `bus`, `airbus` and `coach` is shown below. 

    <Item>
      <Name>VEH_DEEPGTAV_BUS</Name>
      <models>
        <Item>
          <Name>bus</Name>
          <Variations type="NULL"/>
        </Item>
        <Item>
          <Name>airbus</Name>
          <Variations type="NULL"/>
        </Item>
        <Item>
          <Name>coach</Name>
          <Variations type="NULL"/>
        </Item>
      </models>
      <flags>POPGROUP_SCENARIO POPGROUP_AMBIENT</flags>
    </Item>

The file `popcycle.dat` defines the proportions of objects from different population groups, that are spawned at different ingame locations at different times of day and weekdays. An excerpt is shown below. The first columns are self explanatory. The population groups and numbers after `peds` describe the proportion of pedestrian population groups, the total amount has to add up to 200. The population groups after `cars` define the proportions of different vehicle population groups, the total number has to add up to 100. Some more explanation is shown in comments at the beginning of the `popcycle.dat` file. 

    //  #Peds  #Scenario #Cars  #prkdcrs #lowprkdcrs    PercCopCars PercCopPed  MaxScenPeds MaxScenVehs MaxPreAssignedParked
        1000     1000        85      20         30           1           1           3          4                   1           peds  DeepGTAV_peds 200  cars  VEH_DEEPGTAV_CAR 20  VEH_DEEPGTAV_VAN 20  VEH_DEEPGTAV_TRUCK 20  VEH_DEEPGTAV_BUS 20  VEH_DEEPGTAV_MOTOR 20
    [...]
    END_POP_SCHEDULE

`edit_popcycle.dat.py` is a script to automate the production of the `popgroups.ymt` and `popcycle.dat` files.
It has to be noted, that those modifications do not perfectly produce the specified proportions of ingame objects. At this point it is not clear why this is the case, but it probably can be  mitigated with some modifications. 


## Graphics Settings
The folders in `Graphics Settings` contain screenshots from GeForce Experience that document the graphics settings that were used in our works. 