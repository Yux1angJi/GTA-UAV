//File for camera parameters
#pragma once

struct CamParams {
    bool firstInit = false;
    bool init; //initialized
    int width;
    int height;
	int screenWidth;
	int screenHeight;
    float nearClip;
    float farClip;
    float fov;

    //near clip width/height
    float ncHeight;
    float ncWidth;

    //These values change between frames
    Vector3 pos; //In world coordinates
    //Theta is in degrees (functions which use it need to convert to radians
    Vector3 theta; //Camera rotation from world coordinates

    //For optimizing 3d to 2d and unit vector to 2d calculations
    Eigen::Vector3f eigenPos;
    Eigen::Vector3f eigenRot;
    Eigen::Vector3f eigenTheta;
    Eigen::Vector3f eigenCamDir;
    Eigen::Vector3f eigenCamUp;
    Eigen::Vector3f eigenCamEast;
    Eigen::Vector3f eigenClipPlaneCenter;
    Eigen::Vector3f eigenCameraCenter;


	// TODO this was added for simplicity, its ownership should be restrucutred.
	Vector3 cameraRotationOffset;

};

//Global variable to be used by scenario and LiDAR
//Ensures the calculation for camera parameters is the same across files
extern CamParams s_camParams;