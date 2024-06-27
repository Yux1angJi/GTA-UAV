#include <Eigen/Core>
#include <Eigen/LU>
#include "Constants.h"
#include "CamParams.h"
#include "..\ObjectDetIncludes.h"
#include "Functions.h"

#pragma once

const bool LOG_RAGE = false;

//Rockstar Advanced Game Engine (RAGE) transformations between coordinate systems
static bool rage_initialized = false;
static Eigen::Matrix4f rage_CamToWorld;
static Eigen::Matrix4f rage_worldToCam;
static Eigen::Matrix4f rage_NDCToCam;
static Eigen::Matrix4f rage_CamToNDC;
static Vector3 rage_CamPos;
static Vector3 rage_CamRot;

//Values constant for a session
static float rage_nc;
static float rage_fc;
static float rage_fov;
static int rage_width;
static int rage_height;

//Far clip Value calculated by Racinsky Matej which is far clip used in NDC calculation
static const float rage_FC = 10003.814;
static const float rage_NC = 1.500225;

static float max_x_ndc = -FLT_MAX;
static float min_x_ndc = FLT_MAX;
static float max_y_ndc = -FLT_MAX;
static float min_y_ndc = FLT_MAX;

static float max_x = -FLT_MAX;
static float min_x = FLT_MAX;
static float max_y = -FLT_MAX;
static float min_y = FLT_MAX;
static float max_z = -FLT_MAX;
static float min_z = FLT_MAX;
static float max_z2 = -FLT_MAX;
static float min_z2 = FLT_MAX;

static void rageResetValues() {
    max_x_ndc = -FLT_MAX;
    min_x_ndc = FLT_MAX;
    max_y_ndc = -FLT_MAX;
    min_y_ndc = FLT_MAX;

    max_x = -FLT_MAX;
    min_x = FLT_MAX;
    max_y = -FLT_MAX;
    min_y = FLT_MAX;
    max_z = -FLT_MAX;
    min_z = FLT_MAX;
    max_z2 = -FLT_MAX;
    min_z2 = FLT_MAX;
}

static void ragePrintValues() {
    std::ostringstream oss;
    oss << "Max/min ndc x: " << max_x_ndc << ", " << min_x_ndc <<
        "\ny: " << max_y_ndc << ", " << min_y_ndc <<
        "\nMax/min_x: " << max_x << ", " << min_x <<
        "\ny: " << max_y << ", " << min_y <<
        "\nMax/min_z: " << max_z << ", " << min_z <<
        "\nz2: " << max_z2 << ", " << min_z2;
    std::string str = oss.str();
    log(str, true);
}

static void rageInitialize(float near_clip, float far_clip, float fov, int width, int height) {
    rage_nc = near_clip;//Trying this out near_clip;
    rage_fc = far_clip;//Far clip from game natives
    rage_fc = rage_FC;//reverse engineered by racinmat
    rage_fov = fov * PI / 180;
    std::ostringstream oss;
    oss << "RageInitialize: Field of View: " << fov << " Rage fov: " << rage_fov <<
        "\nNear Clip: " << near_clip << "Far clip: " << far_clip;
    std::string str = oss.str();
    log(str, true);
    rage_width = width;
    rage_height = height;
    rage_initialized = true;
}

//Call this function before starting to convert a depth map
//Input current camera position and rotation
static void rageNewDepthMap(Vector3 pos, Vector3 rot) {
    if (!rage_initialized) {
        rageInitialize(s_camParams.nearClip, s_camParams.farClip, s_camParams.fov, s_camParams.width, s_camParams.height);
    }
    //Convert rotation to radians
    rot.x = rot.x * PI / 180;
    rot.y = rot.y * PI / 180;
    rot.z = rot.z * PI / 180;

    //World to Cam and inverse calculations
    Eigen::Matrix4f WorldToCamA;
    Eigen::Matrix4f WorldToCamB;
    Eigen::Matrix4f WorldToCamC;
    Eigen::Matrix4f WorldToCamD;
    Eigen::Matrix4f WorldToCam;
    WorldToCamA << 1, 0, 0, 0,
                   0, sin(rot.x), cos(rot.x), 0,
                   0, cos(rot.x), -sin(rot.x), 0,
                   0, 0, 0, 1;

    WorldToCamB << cos(rot.y), 0, -sin(rot.y), 0,
                   0, 1, 0, 0,
                   sin(rot.y), 0, cos(rot.y), 0,
                   0, 0, 0, 1;

    WorldToCamC << cos(rot.z), sin(rot.z), 0, 0,
                   sin(rot.z), -cos(rot.z), 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1;

    WorldToCamD << 1, 0, 0, -pos.x,
                   0, 1, 0, -pos.y,
                   0, 0, 1, -pos.z,
                   0, 0, 0, 1;

    //Ordering works correctly with Eigen
    WorldToCam = WorldToCamA * WorldToCamB * WorldToCamC * WorldToCamD;
    rage_worldToCam = WorldToCam;
    rage_CamToWorld = WorldToCam.inverse();

    //Cam to NDC and inverse calculations
    Eigen::Matrix4f CamToNDC;
    /*CamToNDC << rage_height / (rage_width * tan(rage_fov / 2)), 0, 0, 0,
        0, 1 / tan(rage_fov / 2), 0, 0,
        0, 0, -rage_fc / (rage_nc - rage_fc), (-rage_fc * rage_nc) / (rage_nc - rage_fc),
        0, 0, -1, 0;*/
    float Q = rage_fc / (rage_fc - rage_nc);
    CamToNDC << rage_height / (rage_width * tan(rage_fov / 2)), 0, 0, 0,
        0, 1 / tan(rage_fov / 2), 0, 0,
        0, 0, Q, 1,
        0, 0, -Q * rage_nc, 0;

    /*CamToNDC << 1.210067, 0, 0, -0.000004,
    0, 2.144507, 0, 0.000002,
    0, 0, 0.00015, 1.500225,
    0, 0, -1, 0;*/
    rage_CamToNDC = CamToNDC;
    rage_NDCToCam = CamToNDC.inverse();

    std::ostringstream oss;
    oss << "Cam rotation: " << rot.x << ", " << rot.y << ", " << rot.z;
    oss << "\nCam position: " << pos.x << ", " << pos.y << ", " << pos.z;
    oss << "\nCos: " << cos(rot.x) << ", " << cos(rot.y) << ", " << cos(rot.z);
    oss << "\nsin: " << sin(rot.x) << ", " << sin(rot.y) << ", " << sin(rot.z);
    oss << "\n FOV and tan(fov/2): " << rage_fov << ", " << tan(rage_fov / 2);
    oss << "\nWorldToCam:\n" << WorldToCam << "\nCamToNDC:\n" << CamToNDC;
    oss << "\nCamToWorld:\n" << rage_CamToWorld << "\nNDCToCAm:\n" << rage_NDCToCam;
    std::string str = oss.str();
    log(str, true);
}

//Transforms from NDC (depth buffer) coordinates to GTA world coordinates
static Vector3 rageNDCToWorld(float depth, int x, int y) {
    Vector3 temp;
    if (!rage_initialized) {
        rageInitialize(s_camParams.nearClip, s_camParams.farClip, s_camParams.fov, s_camParams.width, s_camParams.height);
    }
    float x_ndc = (2 * (float)x / rage_width) - 1;
    float y_ndc = -((2 * (float)y / rage_height) - 1);

    float ncX = x_ndc * s_camParams.ncWidth / 2;
    float ncY = x_ndc * s_camParams.ncHeight / 2;

    //Distance to near clip (hypotenus)
    float d2nc = sqrt(s_camParams.nearClip * s_camParams.nearClip + ncX * ncX + ncY * ncY);
    float worldDepth = d2nc / depth;

    Eigen::Vector4f ndc(x_ndc, y_ndc, worldDepth, 1);

    Eigen::Vector4f camCoords = rage_NDCToCam * ndc;
    Eigen::Vector4f worldCoords = rage_CamToWorld * camCoords;

    if (LOG_RAGE) {
        std::ostringstream oss;
        oss << "Depth: " << depth;
        oss << "\nx/y NDC: " << x_ndc << ", " << y_ndc;
        oss << "\nCam coords: " << camCoords(0) << ", " << camCoords(1) << ", " << camCoords(2) << ", " << camCoords(3);
        oss << "\nWorld coords: " << worldCoords(0) << ", " << worldCoords(1) << ", " << worldCoords(2) << ", " << worldCoords(3);
        std::string str = oss.str();
        log(str);
    }

    temp.x = worldCoords(0);
    temp.y = worldCoords(1);
    temp.z = worldCoords(2);
    return temp;
}

static Vector3 rageNDCToCam(float depth, float screenX, float screenY, bool logRage) {
    Vector3 result;

    float x_ndc = 2 * screenX / float(s_camParams.width - 1) - 1.0f;
    float y_ndc = 2 * screenY / float(s_camParams.height - 1) - 1.0f;

    if (x_ndc > max_x_ndc) max_x_ndc = x_ndc;
    if (x_ndc < min_x_ndc) min_x_ndc = x_ndc;
    if (y_ndc > max_y_ndc) max_y_ndc = y_ndc;
    if (y_ndc < min_y_ndc) min_y_ndc = y_ndc;

    Eigen::Vector4f P_ndc(x_ndc, y_ndc, depth, 1);

    Eigen::Vector4f camCoords = rage_NDCToCam * P_ndc;
    

    result.x = camCoords(0);
    result.y = camCoords(1);
    result.z = camCoords(2);

    if (camCoords(0) > max_x) max_x = camCoords(0);
    if (camCoords(0) < min_x) min_x = camCoords(0);
    if (camCoords(1) > max_y) max_y = camCoords(1);
    if (camCoords(1) < min_y) min_y = camCoords(1);
    if (camCoords(2) > max_z) max_z = camCoords(2);
    if (camCoords(2) < min_z) min_z = camCoords(2);
    if (camCoords(3) > max_z2) max_z2 = camCoords(3);
    if (camCoords(3) < min_z2) min_z2 = camCoords(3);

    if (logRage) {
        std::ostringstream oss;
        oss << "Depth: " << depth;
        oss << "\nx/y NDC: " << x_ndc << ", " << y_ndc;
        oss << "\nCam coords: " << camCoords(0) << ", " << camCoords(1) << ", " << camCoords(2) << ", " << camCoords(3);
        std::string str = oss.str();
        log(str, true);
    }
    return result;
}

static Vector3 rageCamToWorld(Vector3 relPos, bool logRage) {
    Vector3 world;

    Eigen::Vector4f camCoords(relPos.x, relPos.z, -relPos.y, 1);
    Eigen::Vector4f worldCoords = rage_CamToWorld * camCoords;

    if (logRage) {
        std::ostringstream oss;
        oss << "\nCam coords: " << camCoords(0) << ", " << camCoords(1) << ", " << camCoords(2) << ", " << camCoords(3);
        oss << "\nWorld coords: " << worldCoords(0) << ", " << worldCoords(1) << ", " << worldCoords(2) << ", " << worldCoords(3);
        oss << "\nRage_camtoworld: \n" << rage_CamToWorld;
        std::string str = oss.str();
        log(str, true);
    }

    world.x = worldCoords(0);
    world.y = worldCoords(1);
    world.z = worldCoords(2);
    return world;
}

static Vector3 rageWorldToCam(Vector3 worldPos, bool logRage) {
    Vector3 relPos;

    Eigen::Vector4f worldCoords(worldPos.x, worldPos.y, worldPos.z, 1);
    Eigen::Vector4f camCoords = rage_worldToCam * worldCoords;

    if (logRage) {
        std::ostringstream oss;
        oss << "World to Cam *********************************************************************";
        oss << "\nCam coords: " << camCoords(0) << ", " << camCoords(1) << ", " << camCoords(2) << ", " << camCoords(3);
        oss << "\nWorld coords: " << worldCoords(0) << ", " << worldCoords(1) << ", " << worldCoords(2) << ", " << worldCoords(3);
        oss << "\nRage_camtoworld: \n" << rage_CamToWorld;
        std::string str = oss.str();
        log(str, true);
    }

    relPos.x = camCoords(0);
    relPos.y = -camCoords(2);
    relPos.z = camCoords(1);
    return relPos;
}

static float rageCamToNDC(Vector3 camPos, bool logRage) {
    float ndcVal;

    Eigen::Vector4f camCoords(camPos.x, camPos.z, -camPos.y, 1);
    Eigen::Vector4f result = rage_CamToNDC * camCoords;

    if (logRage) {
        float logResult = result(2) * (2.0 * log(abs(result(2)) / rage_nc) / log(rage_fc / rage_nc) - 1);
        float logResult2 = camPos.z * (2.0 * log(abs(camPos.z) / rage_nc) / log(rage_fc / rage_nc) - 1);
        float logResult3 = -camPos.y * (2.0 * log(abs(-camPos.y) / rage_nc) / log(rage_fc / rage_nc) - 1);
        float logResult4 = -camPos.y * (2.0 * log(abs(-camPos.y) + 1) / log(rage_fc + 1) - 1);
        float logResult5 = result(2) * (2.0 * log(abs(result(2)) + 1) / log(rage_fc + 1) - 1);
        std::ostringstream oss;
        oss << "\nCam coords: " << camCoords(0) << ", " << camCoords(1) << ", " << camCoords(2) << ", " << camCoords(3);
        oss << "\n result: " << result(0) << ", " << result(1) << ", " << result(2) << ", " << result(3);
        oss << "\n logResult: " << logResult << ", " << logResult2 << ", " << logResult3 << ", " << logResult4 << ", " << logResult5;
        std::string str = oss.str();
        log(str, true);
    }
    return result(2);
}