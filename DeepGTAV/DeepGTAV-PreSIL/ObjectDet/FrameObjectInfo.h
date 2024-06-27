#pragma once

#include <map>

struct ObjEntity {
    int entityID;
    int classID;

    //Consistent with kitti coordinate system (relative to camera)
    Vector3 location;
    Vector3 worldPos;//To reduce calculations. Center of object at it's bottom
    float speed;
    float heading;
    float height;
    float width;
    float length;
    Vector3 dim;
    Vector3 offcenter;

    float rotation_y;
    float alpha;
    
    float distance;
    BBox2D bbox2d;
    BBox2D bbox2dUnprocessed;
    
    int pointsHit3D;
    int pointsHit2D;
    float truncation;
    float occlusion;

    Hash model;
    std::string modelString;
    std::string objType;

    int trackFirstFrame; //First frame (of sequence) it is seen in

    //These are in world coords right now (probably needs to be changed)
    float pitch;
    float roll;

    bool isPedInV = false;
    int vPedIsIn;


    //The following are saved to reduce calculations for checking if each pixels
    //resides in the 3D bounding box of an entity
    //Vectors which transform from vehicle into world coordinates
    Vector3 xVector;
    Vector3 yVector;
    Vector3 zVector;
    Vector3 u;
    Vector3 v;
    Vector3 w;

    Vector3 rearBotLeft;
    Vector3 frontBotLeft;
    Vector3 rearTopLeft;
    Vector3 rearBotRight;
    Vector3 rearMiddleLeft;
    Vector3 rearThirdLeft;
    Vector3 rearTopExactLeft;

    ObjEntity(int _entityID) : entityID(_entityID) {};
    ObjEntity() {};
};

typedef std::pair<int, ObjEntity> EntityMapEntry;
typedef std::map<int, ObjEntity> EntityMap;

struct FrameObjectInfo
{
    int instanceIdx;
    int seriesIdx;


    //In world coordinates
    Vector3 forwardVec;
    Vector3 rightVec;
    Vector3 upVec;
    Vector3 camPos;
    Vector3 kittiWorldPos;//Bottom center (after correcting for offset) in GTA world coords


    float focalLen;

    //These are maps for when we do 4 cameras for full LiDAR
    //It will enable quicker checking to see if entity is already discovered
    EntityMap vehicles;
    EntityMap peds;

    std::vector<Vector3> groundPlanePoints;
};