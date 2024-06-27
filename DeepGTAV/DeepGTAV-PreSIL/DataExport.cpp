#include "DataExport.h"

#include "lib/utils.h"
#include "lib/rapidjson/writer.h"
#include "Rewarders\GeneralRewarder.h"
#include "Rewarders\LaneRewarder.h"
#include "Rewarders\SpeedRewarder.h"
#include "defaults.h"
#include <time.h>
#include <fstream>
#include <string>
#include <sstream>
#include "Functions.h"
#include "Constants.h"
#include <Eigen/Core>
#include <sstream>
#include "AreaRoaming.h"





const float VERT_CAM_FOV = 59; //In degrees




void DataExport::initialize() {
	Vector3 rotation;

	rotation = ENTITY::GET_ENTITY_ROTATION(*m_ownVehicle, 0);
	CAM::DESTROY_ALL_CAMS(TRUE);
	camera = CAM::CREATE_CAM("DEFAULT_SCRIPTED_CAMERA", TRUE);
	//if (strcmp(_vehicle, "packer") == 0) CAM::ATTACH_CAM_TO_ENTITY(camera, vehicle, 0, 2.35, 1.7, TRUE);
	//else CAM::ATTACH_CAM_TO_ENTITY(camera, vehicle, 0, CAM_OFFSET_FORWARD, CAM_OFFSET_UP, TRUE);
	CAM::SET_CAM_FOV(camera, VERT_CAM_FOV);
	CAM::SET_CAM_ACTIVE(camera, TRUE);


	// TODO CANGED
	CAM::SET_CAM_ROT(camera, rotation.x, rotation.y, rotation.z, 0);
	//CAM::SET_CAM_ROT(camera, rotation.x, rotation.y, 2, 0);

	// pos = CAM::GET_CAM_COORD()
	//Vector3 camPos;
	//camPos = CAM::GET_CAM_COORD(camera);
	//CAM::SET_CAM_COORD(camera, camPos.x + 40.0, camPos.y + 40.0, camPos.z + 40.0);


	CAM::SET_CAM_INHERIT_ROLL_VEHICLE(camera, TRUE);


}


//void DataExport::setCapturedData() {
//
//}







void DataExport::parseDatasetConfig(const Value& dc, bool setDefaults) {
	log("DataExport::parseDatasetConifg");


	if (!dc["frame"].IsNull()) {
		if (!dc["frame"][0].IsNull()) s_camParams.width = dc["frame"][0].GetInt();
		else if (setDefaults) s_camParams.width = _DEFAULT_CAMERA_WIDTH_;

		if (!dc["frame"][1].IsNull()) s_camParams.height = dc["frame"][1].GetInt();
		else if (setDefaults) s_camParams.height = _DEFAULT_CAMERA_HEIGHT_;
	}
	else if (setDefaults) {
		s_camParams.width = _DEFAULT_CAMERA_WIDTH_;
		s_camParams.height = _DEFAULT_CAMERA_HEIGHT_;
	}

	if (!dc["screenResolution"].IsNull()) {
		if (!dc["screenResolution"][0].IsNull()) s_camParams.screenWidth = dc["screenResolution"][0].GetInt();
		else if (setDefaults) s_camParams.screenWidth = _DEFAULT_SCREEN_WIDHT_;

		if (!dc["screenResolution"][1].IsNull()) s_camParams.screenHeight = dc["screenResolution"][1].GetInt();
		else if (setDefaults) s_camParams.screenHeight = _DEFAULT_SCREEN_HEIGHT_;
	}
	else if (setDefaults){
		s_camParams.screenWidth = _DEFAULT_SCREEN_WIDHT_;
		s_camParams.screenHeight = _DEFAULT_SCREEN_HEIGHT_;
	}

	//Need to reset camera params when dataset config is received
	s_camParams.init = false;

	if (dc["reward"].IsArray()) {
		if (dc["reward"][0].IsFloat() && dc["reward"][1].IsFloat()) {
			rewarder = new GeneralRewarder((char*)(GetCurrentModulePath() + "paths.xml").c_str(), dc["reward"][0].GetFloat(), dc["reward"][1].GetFloat());
			reward = true;
		}
		else if (setDefaults) reward = _REWARD_;
	}
	else if (setDefaults) reward = _REWARD_;


	if (!dc["startIndex"].IsNull()) {
		instance_index = dc["startIndex"].GetInt();
	}
	if (!dc["throttle"].IsNull()) throttle = dc["throttle"].GetBool();
	else if (setDefaults) throttle = _THROTTLE_;
	if (!dc["brake"].IsNull()) brake = dc["brake"].GetBool();
	else if (setDefaults) brake = _BRAKE_;
	if (!dc["steering"].IsNull()) steering = dc["steering"].GetBool();
	else if (setDefaults) steering = _STEERING_;
	if (!dc["speed"].IsNull()) speed = dc["speed"].GetBool();
	else if (setDefaults) speed = _SPEED_;
	if (!dc["yawRate"].IsNull()) yawRate = dc["yawRate"].GetBool();
	else if (setDefaults) yawRate = _YAW_RATE_;
	if (!dc["location"].IsNull()) location = dc["location"].GetBool();
	else if (setDefaults) location = _LOCATION_;
	if (!dc["time"].IsNull()) time = dc["time"].GetBool();
	else if (setDefaults) time = _TIME_;


	if (!dc["exportBBox2D"].IsNull()) exportBBox2D = dc["exportBBox2D"].GetBool();
	else if (setDefaults) exportBBox2D = _EXPORT_BBOX_2D_;
	if (!dc["exportBBox2DUnprocessed"].IsNull()) exportBBox2DUnprocessed = dc["exportBBox2DUnprocessed"].GetBool();
	else if (setDefaults) exportBBox2DUnprocessed = _EXPORT_BBOX_2D_UNPROCESSED_;
	if (!dc["occlusionImage"].IsNull()) occlusionImage = dc["occlusionImage"].GetBool();
	else if (setDefaults) occlusionImage = _OCCLUSION_IMAGE_;
	if (!dc["unusedStencilIPixelmage"].IsNull()) unusedStencilIPixelmage = dc["unusedStencilIPixelmage"].GetBool();
	else if (setDefaults) unusedStencilIPixelmage = _UNUSED_STENCIL_IMAGE_;
	if (!dc["segmentationImage"].IsNull()) segmentationImage = dc["segmentationImage"].GetBool();
	else if (setDefaults) segmentationImage = _SEGMENTATION_IMAGE_;
	if (!dc["instanceSegmentationImage"].IsNull()) instanceSegmentationImage = dc["instanceSegmentationImage"].GetBool();
	else if (setDefaults) instanceSegmentationImage = _INSTANCE_SEGMENTATION_IMAGE_;
	if (!dc["instanceSegmentationImageColor"].IsNull()) instanceSegmentationImageColor = dc["instanceSegmentationImageColor"].GetBool();
	else if (setDefaults) instanceSegmentationImageColor = _INSTANCE_SEGMENTATION_IMAGE_COLOR_;
	if (!dc["exportLiDAR"].IsNull()) exportLiDAR = dc["exportLiDAR"].GetBool();
	else if (setDefaults) exportLiDAR = _EXPORT_LIDAR_;
	if (!dc["exportLiDARRaycast"].IsNull()) exportLiDARRaycast = dc["exportLiDARRaycast"].GetBool();
	else if (setDefaults) exportLiDARRaycast = _EXPORT_LIDAR_RAYCAST_;
	if (!dc["maxLidarDist"].IsNull()) maxLidarDist = dc["maxLidarDist"].GetFloat();
	else if (setDefaults) maxLidarDist = _MAX_LIDAR_DIST_;
	if (!dc["export2DPointmap"].IsNull()) export2DPointmap = dc["export2DPointmap"].GetBool();
	else if (setDefaults) export2DPointmap = _EXPORT_2D_POINTMAP_;
	if (!dc["exportSome2DPointmapText"].IsNull()) exportSome2DPointmapText = dc["exportSome2DPointmapText"].GetBool();
	else if (setDefaults) exportSome2DPointmapText = _EXPORT_SOME_2D_POINTMAP_TEXT_;
	if (!dc["exportLiDARDepthStats"].IsNull()) exportLiDARDepthStats = dc["exportLiDARDepthStats"].GetBool();
	else if (setDefaults) exportLiDARDepthStats = _EXPORT_LIDAR_DEPTH_STATS_;
	if (!dc["exportStencliBuffer"].IsNull()) exportStencliBuffer = dc["exportStencliBuffer"].GetBool();
	else if (setDefaults) exportStencliBuffer = _EXPORT_STENCIL_BUFFER_;
	if (!dc["exportStencilImage"].IsNull()) exportStencilImage = dc["exportStencilImage"].GetBool();
	else if (setDefaults) exportStencilImage = _EXPORT_STENCIL_IMAGE_;
	if (!dc["exportIndividualStencilImages"].IsNull()) exportIndividualStencilImages = dc["exportIndividualStencilImages"].GetBool();
	else if (setDefaults) exportIndividualStencilImages = _EXPORT_INDIVIDUAL_STENCIL_IMAGE_;
	if (!dc["exportDepthBuffer"].IsNull()) exportDepthBuffer = dc["exportDepthBuffer"].GetBool();
	else if (setDefaults) exportDepthBuffer = _EXPORT_DEPTH_BUFFER_;


	screenCapturer = new ScreenCapturer(s_camParams.screenWidth, s_camParams.screenHeight);

	buildJSONObject();

}




void DataExport::buildJSONObject() {
	log("DataExport::buildJSONObject");
	d.SetObject();
	Document::AllocatorType& allocator = d.GetAllocator();
	Value a(kArrayType);


	//TODO rename those settings properly (export_...)

	if (direction) d.AddMember("direction", a, allocator);
	if (reward) d.AddMember("reward", 0.0, allocator);
	if (throttle) d.AddMember("throttle", 0.0, allocator);
	if (brake) d.AddMember("brake", 0.0, allocator);
	if (steering) d.AddMember("steering", 0.0, allocator);
	if (speed) d.AddMember("speed", 0.0, allocator);
	if (yawRate) d.AddMember("yawRate", 0.0, allocator);
	if (location) d.AddMember("location", a, allocator);
	if (time) d.AddMember("time", a, allocator);

	// TODO add setting for those (test for those should also be made below)
	// TODO remove unnecessary ones
	d.AddMember("index", 0, allocator);
	d.AddMember("focalLen", 0.0, allocator);
	d.AddMember("curPosition", a, allocator);
	d.AddMember("seriesIndex", a, allocator);
	d.AddMember("HeightAboveGround", 0.0, allocator);
	d.AddMember("CameraAngle", a, allocator);
	d.AddMember("CameraPosition", a, allocator);


	// Add empty fields. This is used to have the fields None in the JSON to prevent client errors
	if (exportBBox2D) d.AddMember("bbox2d", a, allocator);
	if (exportBBox2DUnprocessed) d.AddMember("bbox2dUnprocessed", a, allocator);
	if (occlusionImage) d.AddMember("occlusionImage", a, allocator);
	if (unusedStencilIPixelmage) d.AddMember("unusedStencilIPixelmage", a, allocator);
	if (segmentationImage) d.AddMember("segmentationImage", a, allocator);
	if (instanceSegmentationImage) d.AddMember("instanceSegmentationImage", a, allocator);
	if (instanceSegmentationImageColor) d.AddMember("instanceSegmentationImageColor", a, allocator);
	if (exportLiDAR) d.AddMember("LiDAR", a, allocator);
	if (exportLiDARRaycast) d.AddMember("LiDARRaycast", a, allocator);
	if (export2DPointmap) d.AddMember("2DPointmap", a, allocator);
	if (exportSome2DPointmapText) d.AddMember("Some2DPointmapText", a, allocator);
	if (exportLiDARDepthStats) d.AddMember("LiDARDepthStats", a, allocator);
	if (exportStencliBuffer) d.AddMember("StencilBuffer", a, allocator);
	if (exportStencilImage) d.AddMember("StencilImage", a, allocator);
	if (exportIndividualStencilImages) d.AddMember("IndividualStencilImage", a, allocator);
	if (exportDepthBuffer) d.AddMember("DepthBuffer", a, allocator);

	

}


void DataExport::setRenderingCam(Vehicle v) {
	log("DataExport::setRenderingCam");
	Vector3 position;
	Vector3 fVec, rVec, uVec;
	Vector3 rotation = ENTITY::GET_ENTITY_ROTATION(v, 0);
	ENTITY::GET_ENTITY_MATRIX(v, &fVec, &rVec, &uVec, &position);

	Vector3 offsetWorld = camToWorld(cameraPositionOffset, fVec, rVec, uVec);
	//Since it's offset need to subtract the cam position
	offsetWorld.x -= s_camParams.pos.x;
	offsetWorld.y -= s_camParams.pos.y;
	offsetWorld.z -= s_camParams.pos.z;

	GAMEPLAY::SET_TIME_SCALE(0.0f);
	GAMEPLAY::SET_GAME_PAUSED(false);
	GAMEPLAY::SET_TIME_SCALE(0.0f);

	//TODO fix pointer billiard, after making DataExport the owner of the camera
	CAM::SET_CAM_COORD(camera, position.x + offsetWorld.x, position.y + offsetWorld.y, position.z + offsetWorld.z);
	CAM::SET_CAM_ROT(camera, rotation.x + cameraRotationOffset.x, rotation.y + cameraRotationOffset.y, rotation.z + cameraRotationOffset.z, 0);
	
	// TODO this was added for simplicity, its ownership should be restrucutred.
	s_camParams.cameraRotationOffset = cameraRotationOffset;


	scriptWait(0);
	GAMEPLAY::SET_GAME_PAUSED(true);

	//std::ostringstream oss;
	//oss << "EntityID/rotation/position: " << v << "\n" <<
	//	position.x << ", " << position.y << ", " << position.z <<
	//	"\n" << rotation.x << ", " << rotation.y << ", " << rotation.z <<
	//	"\nOffset: " << offset.x << ", " << offset.y << ", " << offset.z <<
	//	"\nOffsetworld: " << offsetWorld.x << ", " << offsetWorld.y << ", " << offsetWorld.z;
	//log(oss.str());
}



void DataExport::setCameraPositionAndRotation(float x, float y, float z, float rot_x, float rot_y, float rot_z) {
	cameraPositionOffset.x = x;
	cameraPositionOffset.y = y;
	cameraPositionOffset.z = z;
	cameraRotationOffset.x = rot_x;
	cameraRotationOffset.y = rot_y;
	cameraRotationOffset.z = rot_z;
}



StringBuffer DataExport::generateMessage() {
	log("DataExport::GenerateMessage");

	buildJSONObject();

	//StringBuffer buffer(0, 131072);
	StringBuffer buffer;
	buffer.Clear();
	Writer<StringBuffer> writer(buffer);


	GAMEPLAY::SET_GAME_PAUSED(true);
	GAMEPLAY::SET_TIME_SCALE(0.0f);

	setRenderingCam((*m_ownVehicle));

	////Can check whether camera and vehicle are aligned
	//Vector3 camRot2 = CAM::GET_CAM_ROT(camera, 0);
	//std::ostringstream oss1;
	//oss1 << "entityRotation X: " << rotation.x << " Y: " << rotation.y << " Z: " << rotation.z <<
	//    "\n camRot X: " << camRot.x << " Y: " << camRot.y << " Z: " << camRot.z <<
	//    "\n camRot2 X: " << camRot2.x << " Y: " << camRot2.y << " Z: " << camRot2.z;
	//std::string str1 = oss1.str();
	//log(str1);


	// Setting bufffers for the Server
	// Those were the original commands from DeepGTAV
	// They each need Scenario::setVehicleList(), Scenario::setPedsList() etc.
	// Those commands would set the JSON document, e.g. d["vehicles"] = _vehicles;

	// Those functionalities have been somehow moved to ObjectDetection.cpp, e.g. ObjectDetection::setVehiclesList()
	// To fix the JSON / Server those have to be implemented again to correctly set e.g. d["vehicles"] = _vehicles
	// A quick fix would be to copy the old Definitions from DeepGTAV, but I don't know if there would be side effects
	// The correct solution would be to integrate the functionalities of e.g. Scenario::setVehiclesList() and ObjectDetection::setVehiclesList()
	// into one.
	//
	// For now i only implement the messages I need and have the rest commented out.

	// if (vehicles) setVehiclesList();
	// if (peds) setPedsList();
	// if (trafficSigns); //TODO
	//if (direction) setDirection(); // TODO add again
	if (reward) exportReward();
	if (throttle) exportThrottle();
	if (brake) exportBrake();
	if (steering) exportSteering();
	if (speed) exportSpeed();
	if (yawRate) exportYawRate();
	if (location) exportLocation();
	if (time) exportTime();
	exportHeightAboveGround();
	exportCameraPosition();
	exportCameraAngle();



	// TODO legacy functions from ObjectDetection: 
	//exportPosition();
	//exportCalib();
	//setGroundPlanePoints();


	if (!m_pObjDet) {
		m_pObjDet.reset(new ObjectDetection());
		m_pObjDet->initCollection(s_camParams.width, s_camParams.height, false, instance_index, maxLidarDist);
		
	}


	if (recording_active) {
		capture();

		setCamParams();
		//setColorBuffer();
		BufferSizes bufferSizes = m_pObjDet->setDepthAndStencil();

		// check if the buffer sizes are actually correct, if not skip this export
		// They could be wrong due to errors in NVIDIA DSR
		if (bufferSizes.DepthBufferSize == s_camParams.width * s_camParams.height * 4
			&& bufferSizes.StencilBufferSize == s_camParams.width * s_camParams.height) {

			// TODO this was for secondary perspective capture, it could be removed
			m_pObjDet->passEntity();






			////Create vehicles if it is a stationary scenario
			//createVehicles();

			//if (GENERATE_SECONDARY_PERSPECTIVES) {
			//	generateSecondaryPerspectives();
			//}

			//For testing to ensure secondary ownvehicle aligns with main perspective
			//generateSecondaryPerspective(m_pObjDet->m_ownVehicleObj);

			FrameObjectInfo fObjInfo = m_pObjDet->generateMessage();

			// TODO remove?
			d["index"] = fObjInfo.instanceIdx;

			Document::AllocatorType& allocator = d.GetAllocator();

			if (exportBBox2D) d.AddMember("bbox2d", m_pObjDet->exportDetectionsString(fObjInfo), allocator);
			if (exportBBox2DUnprocessed) d.AddMember("bbox2dUnprocessed", m_pObjDet->exportDetectionsStringUnprocessed(fObjInfo), allocator);
			if (occlusionImage) d.AddMember("occlusionImage", m_pObjDet->outputOcclusion(), allocator);
			if (unusedStencilIPixelmage) d.AddMember("unusedStencilIPixelmage", m_pObjDet->outputUnusedStencilPixels(), allocator);

			// TODO this is not clean right now, make this better later
			// Export different Segmentation images:
			if (segmentationImage) d.AddMember("segmentationImage", m_pObjDet->exportSegmentationImage(), allocator);
			if (instanceSegmentationImage) d.AddMember("instanceSegmentationImage", m_pObjDet->printInstanceSegmentationImage(), allocator);
			if (instanceSegmentationImageColor) d.AddMember("instanceSegmentationImageColor", m_pObjDet->printInstanceSegmentationImageColor(), allocator);
			if (exportLiDAR) d.AddMember("LiDAR", m_pObjDet->exportLiDAR(), allocator);
			if (exportLiDARRaycast) d.AddMember("LiDARRaycast", m_pObjDet->exportLiDARRaycast(), allocator);
			if (export2DPointmap) d.AddMember("2DPointmap", m_pObjDet->export2DPointmap(), allocator);
			if (exportSome2DPointmapText) d.AddMember("Some2DPointmapText", m_pObjDet->exportSome2DPointmapText(), allocator);
			if (exportLiDARDepthStats) d.AddMember("LiDARDepthStats", m_pObjDet->exportLidarDepthStats(), allocator);
			if (exportStencliBuffer) d.AddMember("StencilBuffer", m_pObjDet->exportStencilBuffer(), allocator);
			if (exportStencilImage) d.AddMember("StencilImage", m_pObjDet->exportStencilImage(), allocator);
			if (exportIndividualStencilImages) d.AddMember("IndividualStencilImage", m_pObjDet->exportIndividualStencilImages(), allocator);
			if (exportDepthBuffer) d.AddMember("DepthBuffer", m_pObjDet->exportDepthBuffer(), allocator);
		}

		m_pObjDet->refreshBuffers();
		m_pObjDet->increaseIndex();
	}

	d.Accept(writer);

	//log("Message JSON");
	//log(buffer.GetString());
	//log("End Message");

	GAMEPLAY::SET_GAME_PAUSED(false);
	GAMEPLAY::SET_TIME_SCALE(1.0f);

	return buffer;
}



void DataExport::setCamParams() {
	log("DataExport::setCamParams");
	//These values stay the same throughout a collection period
	if (!s_camParams.init) {
		s_camParams.nearClip = CAM::GET_CAM_NEAR_CLIP(camera);
		s_camParams.farClip = CAM::GET_CAM_FAR_CLIP(camera);
		s_camParams.fov = CAM::GET_CAM_FOV(camera);
		s_camParams.ncHeight = 2 * s_camParams.nearClip * tan(s_camParams.fov / 2. * (PI / 180.)); // field of view is returned vertically
		s_camParams.ncWidth = s_camParams.ncHeight * GRAPHICS::_GET_SCREEN_ASPECT_RATIO(false);
		s_camParams.init = true;

		//if (m_recordScenario) {
		//	float gameFC = CAM::GET_CAM_FAR_CLIP(camera);
		//	std::ostringstream oss;
		//	oss << "NC, FC (gameFC), FOV: " << s_camParams.nearClip << ", " << s_camParams.farClip << " (" << gameFC << "), " << s_camParams.fov;
		//	std::string str = oss.str();
		//	log(str, true);
		//}
	}

	//These values change frame to frame
	s_camParams.theta = CAM::GET_CAM_ROT(camera, 0);
	s_camParams.pos = CAM::GET_CAM_COORD(camera);

	//std::ostringstream oss1;
	//oss1 << "\ns_camParams.pos X: " << s_camParams.pos.x << " Y: " << s_camParams.pos.y << " Z: " << s_camParams.pos.z <<
	//	"\nvehicle.pos X: " << currentPos.x << " Y: " << currentPos.y << " Z: " << currentPos.z <<
	//	"\nfar: " << s_camParams.farClip << " nearClip: " << s_camParams.nearClip << " fov: " << s_camParams.fov <<
	//	"\nrotation gameplay: " << s_camParams.theta.x << " Y: " << s_camParams.theta.y << " Z: " << s_camParams.theta.z <<
	//	"\n AspectRatio: " << GRAPHICS::_GET_SCREEN_ASPECT_RATIO(false);
	//std::string str1 = oss1.str();
	//log(str1);

	//For optimizing 3d to 2d and unit vector to 2d calculations
	s_camParams.eigenPos = Eigen::Vector3f(s_camParams.pos.x, s_camParams.pos.y, s_camParams.pos.z);
	s_camParams.eigenRot = Eigen::Vector3f(s_camParams.theta.x, s_camParams.theta.y, s_camParams.theta.z);
	s_camParams.eigenTheta = (PI / 180.0) * s_camParams.eigenRot;
	s_camParams.eigenCamDir = rotate(WORLD_NORTH, s_camParams.eigenTheta);
	s_camParams.eigenCamUp = rotate(WORLD_UP, s_camParams.eigenTheta);
	s_camParams.eigenCamEast = rotate(WORLD_EAST, s_camParams.eigenTheta);
	s_camParams.eigenClipPlaneCenter = s_camParams.eigenPos + s_camParams.nearClip * s_camParams.eigenCamDir;
	s_camParams.eigenCameraCenter = -s_camParams.nearClip * s_camParams.eigenCamDir;

	//For measuring height of camera (LiDAR) to ground plane
	/*float groundZ;
	GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(s_camParams.pos.x, s_camParams.pos.y, s_camParams.pos.z, &(groundZ), 0);

	std::ostringstream oss;
	oss << "LiDAR height: " << s_camParams.pos.z - groundZ;
	std::string str = oss.str();
	log(str);*/
	log("DataExport::setCamParams END");
}



void DataExport::capture() {
	log("DataExport::capture");
	//Time synchronization seems to be correct with 2 render calls
	CAM::RENDER_SCRIPT_CAMS(TRUE, FALSE, 0, FALSE, FALSE);
	scriptWait(0);
	CAM::RENDER_SCRIPT_CAMS(TRUE, FALSE, 0, FALSE, FALSE);
	scriptWait(0);
	CAM::RENDER_SCRIPT_CAMS(TRUE, FALSE, 0, FALSE, FALSE);
	scriptWait(0);
	screenCapturer->capture();
}


void DataExport::setRecording_active(bool x) {
	recording_active = x;
}


////Generate a secondary perspective for all nearby vehicles
//void DataExport::generateSecondaryPerspectives() {
//	for (ObjEntity v : m_pObjDet->m_nearbyVehicles) {
//		if (VEHICLE::IS_THIS_MODEL_A_CAR(v.model)) {
//			generateSecondaryPerspective(v);
//		}
//	}
//	m_pObjDet->m_nearbyVehicles.clear();
//}
//
//void DataExport::generateSecondaryPerspective(ObjEntity vInfo) {
//	setRenderingCam(vInfo.entityID, vInfo.height, vInfo.length);
//
//	//GAMEPLAY::SET_GAME_PAUSED(true);
//	capture();
//
//	setCamParams();
//	setDepthBuffer();
//	setStencilBuffer();
//
//	FrameObjectInfo fObjInfo = m_pObjDet->generateMessage(depth_map, m_stencilBuffer, vInfo.entityID);
//	m_pObjDet->exportDetections(fObjInfo, &vInfo);
//	std::string filename = m_pObjDet->getStandardFilename("image_2", ".png");
//	m_pObjDet->exportImage(screenCapturer->pixels, filename);
//
//	//GAMEPLAY::SET_GAME_PAUSED(false);
//}




/*
 ********************************************************************
 Set Individual JSON fields
 ********************************************************************
*/

void DataExport::exportThrottle() {
	d["throttle"] = getFloatValue(*m_ownVehicle, 0x92C);
}

void DataExport::exportBrake() {
	d["brake"] = getFloatValue(*m_ownVehicle, 0x930);
}

void DataExport::exportSteering() {
	d["steering"] = -getFloatValue(*m_ownVehicle, 0x924) / 0.6981317008;
}

void DataExport::exportSpeed() {
	d["speed"] = ENTITY::GET_ENTITY_SPEED(*m_ownVehicle);
}

void DataExport::exportYawRate() {
	Vector3 rates = ENTITY::GET_ENTITY_ROTATION_VELOCITY(*m_ownVehicle);
	d["yawRate"] = rates.z*180.0 / 3.14159265359;
}

void DataExport::exportLocation() {
	Document::AllocatorType& allocator = d.GetAllocator();
	Vector3 pos = ENTITY::GET_ENTITY_COORDS(*m_ownVehicle, false);
	Value location(kArrayType);
	location.PushBack(pos.x, allocator).PushBack(pos.y, allocator).PushBack(pos.z, allocator);
	d["location"] = location;
}

void DataExport::exportTime() {
	Document::AllocatorType& allocator = d.GetAllocator();
	Value time(kArrayType);
	time.PushBack(TIME::GET_CLOCK_HOURS(), allocator).PushBack(TIME::GET_CLOCK_MINUTES(), allocator).PushBack(TIME::GET_CLOCK_SECONDS(), allocator);
	d["time"] = time;
}

void DataExport::exportHeightAboveGround() {
	Vector3 pos = ENTITY::GET_ENTITY_COORDS(*m_ownVehicle, false);
	float waterZ;
	WATER::GET_WATER_HEIGHT(pos.x, pos.y, pos.z, &waterZ);
	float heightAboveWater = pos.z - waterZ;
	float height = std::min(heightAboveWater, ENTITY::GET_ENTITY_HEIGHT_ABOVE_GROUND(*m_ownVehicle));

	d["HeightAboveGround"] = height;
}

//void DataExport::setDirection() {
//	int direction;
//	float distance;
//	Vehicle temp_vehicle;
//	Document::AllocatorType& allocator = d.GetAllocator();
//	PATHFIND::GENERATE_DIRECTIONS_TO_COORD(dir.x, dir.y, dir.z, TRUE, &direction, &temp_vehicle, &distance);
//	Value _direction(kArrayType);
//	_direction.PushBack(direction, allocator).PushBack(distance, allocator);
//	d["direction"] = _direction;
//}

void DataExport::exportReward() {
	d["reward"] = rewarder->computeReward(*m_ownVehicle);
}

void DataExport::exportCameraPosition() {
	Document::AllocatorType& allocator = d.GetAllocator();
	Vector3 pos = CAM::GET_CAM_COORD(camera);
	Value position(kArrayType);
	position.PushBack(pos.x, allocator).PushBack(pos.y, allocator).PushBack(pos.z, allocator);
	d["CameraPosition"] = position;
}

void DataExport::exportCameraAngle() {
	Document::AllocatorType& allocator = d.GetAllocator();
	Vector3 ang = CAM::GET_CAM_ROT(camera, 0);
	Value angles(kArrayType);
	angles.PushBack(ang.x, allocator).PushBack(ang.y, allocator).PushBack(ang.z, allocator);
	d["CameraAngle"] = angles;
}

//void DataExport::exportWeather() {
//	d["Weather"] = ...;
//}




