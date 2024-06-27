#pragma once

#include <stdlib.h>
#include <ctime>

#include "LiDAR.h"
#include "Functions.h"
#include "CamParams.h"
#include "FrameObjectInfo.h"
#include <opencv2\opencv.hpp>
#include <boost/shared_ptr.hpp>

//#define DEBUG 1

struct BufferSizes {
	int DepthBufferSize;
	int StencilBufferSize;
};


struct WorldObject {
    Entity e;
    std::string type;
    Hash model;
};

static Vector3 createVec3(float x, float y, float z) {
    Vector3 vec;
    vec.x = x;
    vec.y = y;
    vec.z = z;
    return vec;
}

//struct SubsetInfo {
//    Vector3 kittiPos;
//    float kittiHeight;
//    float kittiWidth;
//    float kittiLength;
//    float beta_kitti;
//    float rot_y;
//    float alpha_kitti;
//};

class ObjectDetection {
public:
    ObjectDetection();
    ~ObjectDetection();
private:
    FrameObjectInfo m_curFrame;
    bool m_initialized = false;
    bool m_eve = false;

    Vehicle m_vehicle = NULL;
    Vehicle m_ownVehicle = NULL;
    Ped ped = NULL;
    Cam camera = NULL;

    bool pointclouds;

    std::string baseFolder;

    Vector3 currentPos;
    Vector3 vehicleForwardVector;
    Vector3 vehicleUpVector;
    Vector3 vehicleRightVector;

    Vector3 m_camForwardVector;
    Vector3 m_camRightVector;
    Vector3 m_camUpVector;


    //LiDAR variables
    LiDAR lidar;
    bool lidar_initialized = false;
    std::unordered_map<int, HitLidarEntity*> m_entitiesHit;
    int lidar_param = 7;

	float m_max_lidar_dist;

    //Perspective variables
    int m_vPerspective = -1;//Entity ID of perspective vehicle (-1 if self)

    //Depth Map variables
    float* m_pDepth = NULL;
    uint8_t* m_pStencil = NULL;
	//unsigned char* m_pColor = NULL;


    float* m_pDMPointClouds = NULL;
    uint16_t* m_pDMImage = NULL;
    uint8_t* m_pStencilImage = NULL;
    unsigned char* color_buf = NULL;
    uint8_t* m_pStencilSeg = NULL;
    int m_stencilSegLength = 0;
    uint32_t* m_pInstanceSeg = NULL;
    int m_instanceSegLength = 0;
    uint8_t* m_pInstanceSegImg = NULL;
    int m_instanceSegImgLength = 0;
    uint8_t* m_pOcclusionImage = NULL;
    uint8_t* m_pUnusedStencilImage = NULL;
    uint8_t* m_pGroundPointsImage = NULL;

    cv::Mat m_depthMat = cv::Mat::zeros(cv::Size(s_camParams.width, s_camParams.height), CV_32FC1);


    //For tracking: first frame in a series that an entity appears
    std::unordered_map<int, int> trackFirstFrame;
    //For calculating real speed
    Vector3 m_trackLastPos;
    float m_trackRealSpeed;
    float m_trackLastRealSpeed;
    float m_trackDist;
    float m_trackDistVar;
    float m_trackDistErrorTotal;
    float m_trackDistErrorTotalVar;
    float m_trackDistErrorTotalCount;
    int m_trackLastIndex = -1;
    int m_trackLastSeqIndex = 0;
    std::string m_timeTrackFile;
    //std::string m_usedPixelFile;

    //Camera intrinsic parameters
    float intrinsics[3];

    bool m_vLookupInit = false;
    std::unordered_map<std::string, std::string> m_vLookup; //Vehicle lookup

    std::unordered_map<Vehicle, std::vector<Ped>> m_pedsInVehicles;

    //Map for tracking which entities are possible for each point which is in multiple 3D boxes
    std::unordered_map<int, std::vector<ObjEntity*>> m_overlappingPoints;

public:
    void initCollection(UINT camWidth, UINT camHeight, bool exportEVE = true, int startIndex = 0, float maxLidarDist = 120.0f);
    //void setCamParams(float* forwardVec = NULL, float* rightVec = NULL, float* upVec = NULL);
    void setOwnVehicleObject();

	// TODO remove
    //FrameObjectInfo setDepthAndStencil(bool prevDepth = false, float* pDepth = NULL, uint8_t* pStencil = NULL);
    //Depth buffer fn/var needs to be accessed by server
    //void setDepthBuffer(bool prevDepth = false);

	//void ObjectDetection::passDepthStencilEntity(float* pDepth, uint8_t* pStencil, int entityID = -1);
	void ObjectDetection::passEntity(int entityID = -1);
	BufferSizes ObjectDetection::setDepthAndStencil();


	FrameObjectInfo generateMessage();
	void refreshBuffers();

	std::string exportDetectionsString(FrameObjectInfo fObjInfo, ObjEntity * vPerspective = NULL);
	std::string exportDetectionsStringUnprocessed(FrameObjectInfo fObjInfo, ObjEntity * vPerspective = NULL);

	// TODO remove
	//void exportDetections(FrameObjectInfo fObjInfo, ObjEntity* vPerspective = NULL);
    std::string exportImage(BYTE* data, int imageType = CV_8UC3);
    void increaseIndex();
    std::string getStandardFilename(std::string subDir, std::string extension);


	// TODO look if those variables are still used (they were not in Scenario)
	int instance_index = 0;
    int series_index = 0;
    std::string series_string = "0000";
    std::string instance_string;

    //Tracking variables
    bool collectTracking;
    //# of instances in one series
    const int trSeriesLength = 500;
    //# of seconds between series
    const int trSeriesGapTime = 30;
    //Used for keeing track of when to add the gap
    bool trSeriesGap = false;

    //Other vehicle detection labels
    std::vector<ObjEntity> m_nearbyVehicles;
    ObjEntity m_ownVehicleObj;


	void printSegImage();
	std::string outputOcclusion();
	std::string outputUnusedStencilPixels();



	// TODO new image export functions
	std::string ObjectDetection::exportSegmentationImage();
	std::string ObjectDetection::printInstanceSegmentationImage();
	std::string ObjectDetection::printInstanceSegmentationImageColor();
	std::string ObjectDetection::exportLiDAR();
	std::string ObjectDetection::exportLiDARRaycast();

	std::string ObjectDetection::export2DPointmap();
	std::string ObjectDetection::exportSome2DPointmapText();
	std::string ObjectDetection::exportLidarDepthStats();

	std::string ObjectDetection::exportStencilBuffer();
	std::string ObjectDetection::exportStencilImage();
	std::string ObjectDetection::exportIndividualStencilImages();

	std::string ObjectDetection::exportDepthBuffer();

private:
    void setVehiclesList();
    void setPedsList();

	// TODO remove those (this can not be done right now, because they set m_curFrame
    // void setSpeed();
    // void setYawRate();
    // void setTime();
    
	void setupLiDAR();
    void setIndex();
    void calcCameraIntrinsics();
    void setFocalLength();
    bool getEntityVector(ObjEntity &entity, int entityID, Hash model, int classid, std::string type, std::string modelString, bool isPedInV, Vehicle vPedIsIn, bool &nearbyVehicle);
    void setPosition();
    Vector3 correctOffcenter(Vector3 position, Vector3 min, Vector3 max, Vector3 forwardVector, Vector3 rightVector, Vector3 upVector, Vector3 &offcenter);
    float observationAngle(Vector3 position);
    void drawVectorFromPosition(Vector3 vector, int blue, int green);
    Vector3 depthToCamCoords(float depth, float screenX, float screenY);
    //void outputRealSpeed();

    BBox2D BBox2DFrom3DObject(Vector3 position, Vector3 dim, Vector3 forwardVector, Vector3 rightVector, Vector3 upVector, bool &success, float &truncation);
    bool in3DBox(Vector3 point, Vector3 objPos, Vector3 dim, Vector3 yVector, Vector3 xVector, Vector3 zVector);
    bool in3DBox(ObjEntity *e, Vector3 point, bool &upperHalf);
    bool in2DBoxUnprocessed(const int &i, const int &j, ObjEntity* e);
    bool checkDirection(Vector3 unit, Vector3 point, Vector3 min, Vector3 max);
    //Process pixel instance segmentations with two different methods
    //2D uses only 2D segmentation techniques whereas 3D uses depth buffer
    //Depth buffer hits vehicle windows whereas stencil buffer does not
    void processSegmentation2D();
    void processSegmentation3D();
	std::vector<ObjEntity*> ObjectDetection::pointInside3DEntities(const Vector3 &worldPos, std::vector<ObjEntity*> possibleEntities);
    void processOverlappingPoints();
    void setEntityBBoxParameters(ObjEntity *e);
    void processStencilPixel3D(const uint8_t &stencilVal, const int &j, const int &i, const Vector3 &xVectorCam, const Vector3 &yVectorCam, const Vector3 &zVectorCam);
    void addSegmentedPoint3D(int i, int j, ObjEntity *e);
    void addPointToSegImages(int i, int j, int entityID);

    void update3DPointsHit();
    void update3DPointsHit(ObjEntity* e);

    void processOcclusion();
    void processOcclusionForEntity(ObjEntity *e, const Vector3 &xVectorCam, const Vector3 &yVectorCam, const Vector3 &zVectorCam);

    void getRollAndPitch(Vector3 rightVector, Vector3 forwardVector, Vector3 upVector, float &pitch, float &roll);

    bool hasLOSToEntity(Entity entityID, Vector3 position, Vector3 dim, Vector3 forwardVector, Vector3 rightVector, Vector3 upVector, bool useOrigin = false, Vector3 origin = createVec3(0,0,0));

    void initVehicleLookup();
    bool isPointOccluding(Vector3 worldPos, ObjEntity* e);

    //Export functions
    void exportEntity(ObjEntity e, std::ostringstream& oss, bool unprocessed, bool augmented, bool checkbbox2d = true, const int &maxDist = -1, const int &min2DPoints = -1, const int &min3DPoints = -1);
    void exportEntities(EntityMap entMap, std::ostringstream& oss, bool unprocessed = false, bool augmented = false, bool checkbbox2d = true, const int &maxDist = -1, const int &min2DPoints = -1, const int &min3DPoints = -1);
    std::string exportCalib();
    std::string exportPosition();
    //void exportEgoObject(ObjEntity vPerspective);

    //Ground plane points
    Vector3 getGroundPoint(Vector3 point, Vector3 yVectorCam, Vector3 xVectorCam, Vector3 zVectorCam);
    std::string setGroundPlanePoints();

    //Other vehicle detections
    //void checkEntity(Vehicle p, WorldObject e, Vector3 pPos, std::ostringstream& oss);
    //SubsetInfo getObjectInfoSubset(Vector3 position, Vector3 forwardVector, Vector3 dim);
    Vector3 getVehicleDims(Entity e, Hash model, Vector3 &min, Vector3 &max);


	// Drawing functionality for debugging
	void ObjectDetection::drawEntityBBox3D(ObjEntity e);
	void ObjectDetection::draw_line(Vector3 a, Vector3 b);
};