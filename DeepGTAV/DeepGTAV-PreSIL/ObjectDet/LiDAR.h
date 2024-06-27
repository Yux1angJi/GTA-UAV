/*

Original source code taken from:
https://github.com/gdpinchina/A-virtual-LiDAR-for-DeepGTAV

*/

#pragma once
#include "..\ObjectDetIncludes.h"
#include <unordered_map>
#include <Eigen/Core>
#include "CamParams.h"

#define _LIDAR_NOT_INIT_YET_ 0
#define _LIDAR_INIT_AS_2D_ 1
#define _LIDAR_INIT_AS_3D_ 2

const int MAX_POINTS = 5000000;

struct HitLidarEntity {
    int pointsHit;
    Vector3 forward;
    Vector3 position;
    float maxFront;
    float maxBack;

    HitLidarEntity(Vector3 _forward, Vector3 _position) {
        pointsHit = 1;
        forward = _forward;
        position = _position;
        maxFront = 0;
        maxBack = 0;
    }
};

struct Hit2DDepth {
    Vector3 target;
    Eigen::Vector2f target2D;
    float groundDist;
    float rayCastDepth;
};

class LiDAR
{
public:
    LiDAR();
    ~LiDAR();

public:
    //Note: polar-coordinate definition below: 
    //1. Horizontal angle ranges from 0 to 360 degree.
    //	[horizRiLimit, horizLeLimit), anti-clock, namely from rightside to leftside.
    //	horizRiLimit:[180, 360), horizLeLimit:[0, 180). 
    //	For example, shown as defaults in API Init2DLiDAR_SmplNum, right:[270, 360), left:[0, 90).
    //
    //2. Vertical angle ranges from 0 to 180 degree.
    //	[vertiUnLimit, vertiUpLimit), namely from underside to upside. 
    //	vertiUnLimit:[180, 90), vertiUpLimit:[90, 0).
    //	For example, shown as defaults in API Init3DLiDAR_SmplNum, downside:[135, 90), upside:[90, 45).
    //
    //3. One single frame contains :
    //	2D: HorizSmplNum ranges. 
    //	3D: VertiSmplNum * horizSmplNum ranges. 
    //
    //4. By defaults:
    //	Position: The LiDAR device is set to the same position of camera.


    void Init2DLiDAR_SmplNum(float maxRange = 100.0, int horizSmplNum = 180, float horizLeLimit = 90.0, float horizRiLimit = 270.0);

    void Init3DLiDAR_SmplNum(float maxRange = 100.0, int horizSmplNum = 180, float horizLeLimit = 90.0, float horizRiLimit = 270.0,
        int vertiSmplNum = 9, float vertiUpLimit = 45, float vertiUnLimit = 135.0);

    void Init2DLiDAR_FOV(float maxRange = 100.0, float horizFOV = 180.0, float horizAngResolu = 1.0);

    void Init3DLiDAR_FOV(float maxRange = 100.0, float horizFOV = 180.0, float horizAngResolu = 1.0, float vertiFOV = 90.0, float vertiAngResolu = 10.0, float vertiUpLimit = 2.0);

    void AttachLiDAR2Camera(Cam camera, Entity ownCar);

    void DestroyLiDAR();

    float* GetPointClouds(int &size, std::unordered_map<int, HitLidarEntity*> *entitiesHit, int param, float* depthMap, uint32_t* pInstanceSeg, Entity perspectiveVehicle = -1);
    float* Get2DPoints(int &size);
    float* GetRaycastPointcloud(int & size);
    float* UpdatePointCloud(int &size, float* depthMap);
    int getTotalSmplNum();
    int getVertiSmplNum();
    int getHorizSmplNum();
    int getCurType();

    void updateCurrentPosition(Vector3 cameraForwardVec, Vector3 cameraRightVector, Vector3 cameraUpVector);

    //Print stats on disparity between depth calculations/raycast distance
    std::string printDepthStats();

private:

    void GenerateSinglePoint(float phi, float theta, float *p);
    void GenerateHorizPointClouds(float phi, float *p);
    void calcDCM();
    void addToHitEntities(const Eigen::Vector2f &target2D);


private:

    float* m_pPointClouds;
    float* m_pRaycastPointCloud;
    int m_pointsHit;
    int m_depthMapPoints;
    int m_raycastPoints;
    float m_maxRange;//meter
    float m_vertiUpLimit;//deg, the upside limit of zenith direction, namely the min vertical angle, 0 <= up <= phiUp < 90
    float m_vertiUnLimit;//deg, the underside limit of ground direction, namely the max vertical angle, 90 <= phiLo <= un <= 180
    float m_horizLeLimit;//deg, the left limit of horizontal direction, if no limits, set to 180, 0 <= thetaLe < le < 180
    float m_horizRiLimit;//deg, the right limit of horizontal direction, if no limits, set to 180, 180 <= ri <= thetaRi < 360
    int m_vertiSmplNum;
    int m_horizSmplNum;
    float m_vertiResolu;//deg, vertical angle resolution
    float m_horizResolu;//deg, horizontal angle resolution

    Cam m_camera;
    Entity m_lidarVehicle;
    float m_quaterion[4];
    float m_rotDCM[9];//convert n-coord to b-coord
    int m_initType;
    bool m_isAttach;

    Vector3 cameraForwardVec;
    Vector3 cameraUpVec;
    Vector3 cameraRightVec;

    float m_max_dist;
    float m_min_dist;

    float* m_lidar2DPoints;
    int m_beamCount;

    std::unordered_map<int, HitLidarEntity*>* m_entitiesHit;
    uint32_t* m_pInstanceSeg;
    int native_param = 7;

    //Depth map variables
    float * m_depthMap;
    Vector3 adjustEndCoord(Vector3 pos, Vector3 relPos);
    float depthFromNDC(int x, int y, float screenX = 0.0f, float screenY = 0.0f);
    float getDepthFromScreenPos(float screenX, float screenY);

    //Updating at a later time with the new depth map
    int m_updatedPointCount;
    float * m_updatedPointCloud;
    Vector3 get3DFromDepthTarget(Vector3 target, Eigen::Vector2f target2D);
    std::vector<Hit2DDepth> m_hitDepthPoints;
};