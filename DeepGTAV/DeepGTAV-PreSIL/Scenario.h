#pragma once

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

#include "DataExport.h"

#include <time.h>

using namespace rapidjson;

//#define DEBUG 1

struct SpawnedPed {
	Ped ped;
	time_t spawntime;
};

struct SpawnedVehicle {
	Vehicle vehicle;
	time_t spawntime;
};


class Scenario {
private:
	static char* weatherList[14];
	static char* vehicleList[3];


	Vehicle m_ownVehicle = NULL;
	Player player = NULL;
	Ped ped = NULL;
	Vector3 dir;

	float x, y, z;
    float startHeading;
	int hour, minute;
	const char* _weather;
	const char* _vehicle;

    bool offscreen;
    bool showBoxes;
    bool stationaryScene;

	float currentThrottle = 0.0;
	float currentBrake = 0.0;
	float currentSteering = 0.0;


	std::clock_t lastSafetyCheck;
	int _drivingMode;
	float _setSpeed;

	bool running = false;

    int m_startArea = 1; //Downtown (see s_locationBounds)
    std::vector<std::vector<char>> m_polyGrid;


    bool vehicles_created = false;
    std::vector<VehicleToCreate> vehiclesToCreate;
    std::vector<PedToCreate> pedsToCreate;

	std::vector<SpawnedPed> spawnedPeds;
	std::vector<SpawnedVehicle> spawnedVehicles;
	double spawnedEntitiesDespawnSeconds;


public:
	float rate;

	void start(const Value& sc, const Value& dc);
	void stop();
	void config(const Value& sc, const Value& dc);
	void setCommands(float throttle, float brake, float steering);
	void run();

	// TODO remove
	StringBuffer generateMessage();
	void setRecording_active(bool x);
	
	void goToLocation(float x, float y, float z, float setSpeed);
	void teleportToLocation(float x, float y, float z);

	// TODO remove
	void setCameraPositionAndRotation(float x, float y, float z, float rot_x, float rot_y, float rot_z);


    //Tracking variables
    bool collectTracking;
    //# of instances in one series
    const int trSeriesLength = 500;
    //# of seconds between series
    const int trSeriesGapTime = 30;
    //Used for keeing track of when to add the gap
    bool trSeriesGap = false;


	void createVehicle(const char* model, float relativeForward, float relativeRight, float heading, int color, int color2, bool placeOnGround, bool withLifeJacketPed);
	void createPed(const char* model, float relativeForward, float relativeRight, float relativeUp, float heading, bool placeOnGround, const char* animDict, const char* animName);
	
	void setWeather(const char * weather);
	void setClockTime(int hour, int minute, int second);

	//TODO move to private
	DataExport exporter;


private:
	void parseScenarioConfig(const Value& sc, bool setDefaults);
	void parseDatasetConfig(const Value& dc, bool setDefaults);
	void buildScenario();

    void drawBoxes(Vector3 BLL, Vector3 FUR, Vector3 dim, Vector3 upVector, Vector3 rightVector, Vector3 forwardVector, Vector3 position, int colour);
    void createVehicles();
    //void setPosition();

	void despawnSpawnedObjectsAfterTime();


};