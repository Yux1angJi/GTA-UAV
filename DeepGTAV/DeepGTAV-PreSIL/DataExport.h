
#include <stdlib.h>
#include <ctime>

#include "lib/script.h"
#include "lib/utils.h"

#include "lib/rapidjson/document.h"
#include "lib/rapidjson/stringbuffer.h"

#include "ScreenCapturer.h"
#include "Rewarders\Rewarder.h"
#include "LiDAR.h"
#include "Functions.h"
#include "CamParams.h"
#include <memory>
#include "ObjectDetection.h"
#include "Constants.h"




using namespace rapidjson;


/*
	The DataExport Class is used to aggregate all the data that will be sent over a TCP-Connection in a JSON Document. 
	To do this the data is generated (mostly from GTAV functions or in ObjectDet
*/
class DataExport {
private:

	// TODO why is this a unique pointer? this should be an object, then the -> should be replaced with .
	std::unique_ptr<ObjectDetection> m_pObjDet = NULL;
	Rewarder* rewarder;

	bool direction;
	bool reward;
	bool throttle;
	bool brake;
	bool steering;
	bool speed;
	bool yawRate;
	bool location;
	bool time;
	bool exportBBox2D;
	bool exportBBox2DUnprocessed;
	bool occlusionImage;
	bool unusedStencilIPixelmage;
	bool segmentationImage;
	bool instanceSegmentationImage;
	bool instanceSegmentationImageColor;
	bool exportLiDAR;
	bool exportLiDARRaycast;
	float maxLidarDist;
	bool export2DPointmap;
	bool exportSome2DPointmapText;
	bool exportLiDARDepthStats;
	bool exportStencliBuffer;
	bool exportStencilImage;
	bool exportIndividualStencilImages;
	bool exportDepthBuffer;


	Document d;

	//void setDirection();
	void exportReward();
	void exportCameraPosition();
	void exportCameraAngle();
	void exportThrottle();
	void exportBrake();
	void exportSteering();
	void exportSpeed();
	void exportYawRate();
	void exportLocation();
	void exportTime();
	void exportHeightAboveGround();


	void setRenderingCam(Vehicle v);
	void capture();


	bool recording_active = false;


	Cam camera = NULL;

	// Persistent Camera offsets
	Vector3 cameraPositionOffset = { 0, 0, 0 };
	Vector3 cameraRotationOffset = { 0, 0, 0 };

	int instance_index = 0;



public:
	StringBuffer generateMessage();
	void parseDatasetConfig(const Value& dc, bool setDefaults);
	void buildJSONObject();

	void setRecording_active(bool x);

	// TODO make private, make camera fully owned by DataExport
	//Cam * camera;
	//Vector3 * cameraPositionOffset;
	//Vector3 * cameraRotationOffset;
	Vehicle * m_ownVehicle;


	//void generateSecondaryPerspectives();
	//void generateSecondaryPerspective(ObjEntity vInfo);


	// TODO make private, after having camera fully in DataExport ownership
	void setCamParams();

	// TODO move to private
	ScreenCapturer* screenCapturer;

	void initialize();

	//TODO move to private
	void setCameraPositionAndRotation(float x, float y, float z, float rot_x, float rot_y, float rot_z);



};