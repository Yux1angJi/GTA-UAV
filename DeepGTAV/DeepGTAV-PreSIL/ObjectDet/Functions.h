#include "..\ObjectDetIncludes.h"
#include <Eigen/Core>
#include <Eigen/Geometry> 

#include "Constants.h"

#include <chrono>

#include <math.h>

#pragma once

static char* logDir = getenv("DEEPGTAV_LOG_FILE");

// TODO using such a static variable is not nice, but I don't see an easier solution right now
static auto lasttime = std::chrono::high_resolution_clock::now();
static void log(std::string str, bool override = false) {
    if ((override || DEBUG_LOGGING) && logDir != NULL) {
        FILE* f = fopen(logDir, "a");
		if (DEBUG_LOG_WITH_TIME) {
			auto thistime = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(thistime - lasttime);
			fprintf(f, "[+%s ms] %s", std::to_string(duration.count()), str.c_str());
			lasttime = thistime;

		}
		else {
			fprintf(f, str.c_str());
		}
        fprintf(f, "\n");
        fclose(f);
    }
}

// Converts a vector 'vec' into the coordinate system with the specified unit vectors in terms of the original coordinate system
static Vector3 convertCoordinateSystem(Vector3 vec, Vector3 forwardVector, Vector3 rightVector, Vector3 upVector) {
    Vector3 newVec;

    newVec.x = vec.x*rightVector.x + vec.y*rightVector.y + vec.z*rightVector.z;
    newVec.y = vec.x*forwardVector.x + vec.y*forwardVector.y + vec.z*forwardVector.z;
    newVec.z = vec.x*upVector.x + vec.y*upVector.y + vec.z*upVector.z;

    return newVec;
}


static Vector3 convertCoordinateSystem2(Vector3 vec, Vector3 b1, Vector3 b2, Vector3 b3) {
	// (b1, b2, b3) * vec should be the correct transformation, where
	// (b1, b2, b3) is a matrix with b_i as column vectors

	Vector3 ret;

	ret.x = vec.x * b1.x + vec.y * b2.x + vec.z * b3.x;
	ret.y = vec.x * b1.y + vec.y * b2.y + vec.z * b3.y;
	ret.z = vec.x * b1.z + vec.y * b2.z + vec.z * b3.z;
	
	return ret;
}

struct BBox2D {
    float left;
    float top;
    float right;
    float bottom;

    float width() {
        return right - left;
    }

    float height() {
        return bottom - top;
    }

    float posX() {
        return left + width() / 2;
    }

    float posY() {
        return top + height() / 2;
    }

    BBox2D() :
        left(s_camParams.width),
        right(0),
        top(s_camParams.height),
        bottom(0)
    {};
};

struct VehicleToCreate {
    std::string model;
    float forward;
    float right;
    float heading;
    int color;
    int color2;

    VehicleToCreate(std::string _model, float _forward, float _right, float _heading,
        int _color, int _color2) :
        model(_model),
        forward(_forward),
        right(_right),
        heading(_heading),
        color(_color),
        color2(_color2)
    {
    }

    VehicleToCreate(){
    }
};

struct PedToCreate {
    int model;
    float forward;
    float right;
    float heading;
};

static Vector3 subtractVector(Vector3 first, Vector3 subtract) {
    Vector3 diff;
    diff.x = first.x - subtract.x;
    diff.y = first.y - subtract.y;
    diff.z = first.z - subtract.z;
    return diff;
}

static Vector3 addVector(Vector3 a, Vector3 b) {
	Vector3 res;
	res.x = a.x + b.x;
	res.y = a.y + b.y;
	res.z = a.z + b.z;
	return res;
}

static Eigen::Vector3f rotate(Eigen::Vector3f a, Eigen::Vector3f theta)
{
    Eigen::Vector3f d;

    d(0) = (float)cos((double)theta(2))*((float)cos((double)theta(1))*a(0) + (float)sin((double)theta(1))*((float)sin((double)theta(0))*a(1) + (float)cos((double)theta(0))*a(2))) - (float)sin((double)theta(2))*((float)cos((double)theta(0))*a(1) - (float)sin((double)theta(0))*a(2));
    d(1) = (float)sin((double)theta(2))*((float)cos((double)theta(1))*a(0) + (float)sin((double)theta(1))*((float)sin((double)theta(0))*a(1) + (float)cos((double)theta(0))*a(2))) + (float)cos((double)theta(2))*((float)cos((double)theta(0))*a(1) - (float)sin((double)theta(0))*a(2));
    d(2) = -(float)sin((double)theta(1))*a(0) + (float)cos((double)theta(1))*((float)sin((double)theta(0))*a(1) + (float)cos((double)theta(0))*a(2));

    return d;
}

static Eigen::Vector2f get_2d_from_3d(const Eigen::Vector3f& vertex, bool draw_debug = false) {
    // Inspired by Artur Filopowicz: Video Games for Autonomous Driving: https://github.com/arturf1/GTA5-Scripts/blob/master/Annotator.cs#L379

    if (draw_debug)
    {
        auto cam_dir_line_end = s_camParams.eigenCamDir + s_camParams.eigenPos;

        GRAPHICS::DRAW_LINE(s_camParams.eigenPos.x(), s_camParams.eigenPos.y(), s_camParams.eigenPos.z(), cam_dir_line_end.x(), cam_dir_line_end.y(), cam_dir_line_end.z(), 0, 255, 0, 200);
    }

    const Eigen::Vector3f near_clip_to_target = vertex - s_camParams.eigenClipPlaneCenter; // del

    Eigen::Vector3f camera_to_target = near_clip_to_target - s_camParams.eigenCameraCenter; // Total distance - subtracting a negative to add clip distance

    if (draw_debug)
    {
        auto cam_up_end_line = s_camParams.eigenCamUp + s_camParams.eigenPos;
        GRAPHICS::DRAW_LINE(s_camParams.eigenPos.x(), s_camParams.eigenPos.y(), s_camParams.eigenPos.z(), cam_up_end_line.x(), cam_up_end_line.y(), cam_up_end_line.z(), 100, 100, 255, 200);
        auto cam_east_end_line = s_camParams.eigenCamUp + s_camParams.eigenPos;
        GRAPHICS::DRAW_LINE(s_camParams.eigenPos.x(), s_camParams.eigenPos.y(), s_camParams.eigenPos.z(), cam_east_end_line.x(), cam_east_end_line.y(), cam_east_end_line.z(), 100, 100, 255, 200);
        auto del_draw = s_camParams.eigenPos + near_clip_to_target;
        GRAPHICS::DRAW_LINE(s_camParams.eigenClipPlaneCenter.x(), s_camParams.eigenClipPlaneCenter.y(), s_camParams.eigenClipPlaneCenter.z(), del_draw.x(), del_draw.y(), del_draw.z(), 255, 255, 100, 255);
        auto viewerDistDraw = s_camParams.eigenPos + camera_to_target;
        GRAPHICS::DRAW_LINE(s_camParams.eigenPos.x(), s_camParams.eigenPos.y(), s_camParams.eigenPos.z(), viewerDistDraw.x(), viewerDistDraw.y(), viewerDistDraw.z(), 255, 100, 100, 255);
    }
    Eigen::Vector3f camera_to_target_unit_vector = camera_to_target * (1. / camera_to_target.norm()); // Unit vector in direction of plane / line intersection

    double view_plane_dist = s_camParams.nearClip / s_camParams.eigenCamDir.dot(camera_to_target_unit_vector);

    Eigen::Vector3f up3d, forward3d, right3d;
    up3d = rotate(WORLD_UP, s_camParams.eigenRot);
    right3d = rotate(WORLD_EAST, s_camParams.eigenRot);
    forward3d = rotate(WORLD_NORTH, s_camParams.eigenRot);
    Eigen::Vector3f new_origin = s_camParams.eigenClipPlaneCenter + (s_camParams.ncHeight / 2.) * s_camParams.eigenCamUp - (s_camParams.ncWidth / 2.) * s_camParams.eigenCamEast;

    if (draw_debug)
    {
        auto top_right = new_origin + s_camParams.ncWidth * right3d;

        GRAPHICS::DRAW_LINE(new_origin.x(), new_origin.y(), new_origin.z(), top_right.x(), top_right.y(), top_right.z(), 100, 255, 100, 255);

        auto bottom_right = top_right - s_camParams.ncHeight * up3d;

        GRAPHICS::DRAW_LINE(bottom_right.x(), bottom_right.y(), bottom_right.z(), top_right.x(), top_right.y(), top_right.z(), 100, 255, 100, 255);

        auto bottom_left = bottom_right - s_camParams.ncWidth * right3d;

        GRAPHICS::DRAW_LINE(bottom_right.x(), bottom_right.y(), bottom_right.z(), bottom_left.x(), bottom_left.y(), bottom_left.z(), 100, 255, 100, 255);
        GRAPHICS::DRAW_LINE(bottom_left.x(), bottom_left.y(), bottom_left.z(), new_origin.x(), new_origin.y(), new_origin.z(), 100, 255, 100, 255);
    }
    Eigen::Vector2f ret;

    bool use_artur_method = true;

    if (use_artur_method)
    {
        Eigen::Vector3f view_plane_point = view_plane_dist * camera_to_target_unit_vector + s_camParams.eigenCameraCenter;
        view_plane_point = (view_plane_point + s_camParams.eigenClipPlaneCenter) - new_origin;
        double viewPlaneX = view_plane_point.dot(s_camParams.eigenCamEast) / s_camParams.eigenCamEast.dot(s_camParams.eigenCamEast);
        double viewPlaneZ = view_plane_point.dot(s_camParams.eigenCamUp) / s_camParams.eigenCamUp.dot(s_camParams.eigenCamUp);
        double screenX = viewPlaneX / (double)s_camParams.ncWidth;
        double screenY = -viewPlaneZ / (double)s_camParams.ncHeight;
        ret = { screenX, screenY };
    }
    else
    {
        auto intersection = s_camParams.eigenPos + view_plane_dist * camera_to_target_unit_vector;
        auto center_to_intersection = s_camParams.eigenClipPlaneCenter - intersection;
        auto x_dist = center_to_intersection.dot(right3d);
        auto z_dist = center_to_intersection.dot(up3d);
        auto screen_x = 1. - (s_camParams.ncWidth / 2. + x_dist) / s_camParams.ncWidth;
        auto screen_y = (s_camParams.ncHeight / 2. + z_dist) / s_camParams.ncHeight;
        ret = { screen_x, screen_y };
    }
    return ret;
}

static const std::vector<Eigen::Vector3f> coefficients = {
    { -0.5, -0.5,-0.5 },
{ 0.5, -0.5,-0.5 },
{ 0.5,  0.5, 0.5 },
{ -0.5,  0.5,-0.5 },
{ -0.5, -0.5, 0.5 },
{ 0.5, -0.5, 0.5 },
{ 0.5,  0.5, 0.5 },
{ -0.5,  0.5, 0.5 }
};

static Vector3 getUnitVector(Vector3 vector) {
    float distance = sqrt(SYSTEM::VDIST2(0, 0, 0, vector.x, vector.y, vector.z));
    Vector3 unitVec;
    unitVec.x = vector.x / distance;
    unitVec.y = vector.y / distance;
    unitVec.z = vector.z / distance;
    return unitVec;
}

static Vector3 subtractVecs(Vector3 first, Vector3 subtract) {
    Vector3 difference;
    difference.x = first.x - subtract.x;
    difference.y = first.y - subtract.y;
    difference.z = first.z - subtract.z;
    return difference;
}

static float dotProd(Vector3 first, Vector3 sec) {
    return (first.x * sec.x + first.y * sec.y + first.z * sec.z);
}

static Vector3 camToWorld(Vector3 relPos, Vector3 camForward, Vector3 camRight, Vector3 camUp) {
    Vector3 worldX; worldX.x = 1; worldX.y = 0; worldX.z = 0;
    Vector3 worldY; worldY.x = 0; worldY.y = 1; worldY.z = 0;
    Vector3 worldZ; worldZ.x = 0; worldZ.y = 0; worldZ.z = 1;
    Vector3 xVectorCam = convertCoordinateSystem(worldX, camForward, camRight, camUp);
    Vector3 yVectorCam = convertCoordinateSystem(worldY, camForward, camRight, camUp);
    Vector3 zVectorCam = convertCoordinateSystem(worldZ, camForward, camRight, camUp);

    Vector3 worldPos = convertCoordinateSystem(relPos, yVectorCam, xVectorCam, zVectorCam);
    worldPos.x += s_camParams.pos.x;
    worldPos.y += s_camParams.pos.y;
    worldPos.z += s_camParams.pos.z;

    return worldPos;
}


static void drawPoint(Vector3 worldPos, float size) {
	GRAPHICS::DRAW_LINE(worldPos.x - size, worldPos.y, worldPos.z, worldPos.x + size, worldPos.y, worldPos.z, 200, 40, 40, 200);
	GRAPHICS::DRAW_LINE(worldPos.x, worldPos.y - size, worldPos.z, worldPos.x, worldPos.y + size, worldPos.z, 200, 40, 40, 200);
	GRAPHICS::DRAW_LINE(worldPos.x, worldPos.y, worldPos.z - size, worldPos.x, worldPos.y, worldPos.z + size, 200, 40, 40, 200);
}


static void drawBoxes(Vector3 position, Vector3 dim, Vector3 forwardVector, Vector3 rightVector, Vector3 upVector, int colour) {
    
	Vector3 FUR, BLL;
	FUR.x = position.x + dim.y*forwardVector.x + dim.x*rightVector.x + dim.z*upVector.x;
	FUR.y = position.y + dim.y*forwardVector.y + dim.x*rightVector.y + dim.z*upVector.y;
	FUR.z = position.z + dim.y*forwardVector.z + dim.x*rightVector.z + dim.z*upVector.z;
	//GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(FUR.x, FUR.y, FUR.z, &(FUR.z), 0);
	//FUR.z += 2 * dim.z;

	BLL.x = position.x - dim.y*forwardVector.x - dim.x*rightVector.x - dim.z*upVector.x;
	BLL.y = position.y - dim.y*forwardVector.y - dim.x*rightVector.y - dim.z*upVector.y;
	BLL.z = position.z - dim.y*forwardVector.z - dim.x*rightVector.z - dim.z*upVector.z;
	//GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(BLL.x, BLL.y, 1000.0, &(BLL.z), 0);
	
	
	Vector3 edge1 = BLL;
    Vector3 edge2;
    Vector3 edge3;
    Vector3 edge4;
    Vector3 edge5 = FUR;
    Vector3 edge6;
    Vector3 edge7;
    Vector3 edge8;

    int green = colour * 255;
    int blue = abs(colour - 1) * 255;

    edge2.x = edge1.x + 2 * dim.y*forwardVector.x;
    edge2.y = edge1.y + 2 * dim.y*forwardVector.y;
    edge2.z = edge1.z + 2 * dim.y*forwardVector.z;

    edge3.x = edge2.x + 2 * dim.z*upVector.x;
    edge3.y = edge2.y + 2 * dim.z*upVector.y;
    edge3.z = edge2.z + 2 * dim.z*upVector.z;

    edge4.x = edge1.x + 2 * dim.z*upVector.x;
    edge4.y = edge1.y + 2 * dim.z*upVector.y;
    edge4.z = edge1.z + 2 * dim.z*upVector.z;

    edge6.x = edge5.x - 2 * dim.y*forwardVector.x;
    edge6.y = edge5.y - 2 * dim.y*forwardVector.y;
    edge6.z = edge5.z - 2 * dim.y*forwardVector.z;

    edge7.x = edge6.x - 2 * dim.z*upVector.x;
    edge7.y = edge6.y - 2 * dim.z*upVector.y;
    edge7.z = edge6.z - 2 * dim.z*upVector.z;

    edge8.x = edge5.x - 2 * dim.z*upVector.x;
    edge8.y = edge5.y - 2 * dim.z*upVector.y;
    edge8.z = edge5.z - 2 * dim.z*upVector.z;


	GRAPHICS::DRAW_LINE(edge1.x, edge1.y, edge1.z, edge2.x, edge2.y, edge2.z, 0, green, blue, 200);
	GRAPHICS::DRAW_LINE(edge1.x, edge1.y, edge1.z, edge4.x, edge4.y, edge4.z, 0, green, blue, 200);
	GRAPHICS::DRAW_LINE(edge2.x, edge2.y, edge2.z, edge3.x, edge3.y, edge3.z, 0, green, blue, 200);
	GRAPHICS::DRAW_LINE(edge3.x, edge3.y, edge3.z, edge4.x, edge4.y, edge4.z, 0, green, blue, 200);

	GRAPHICS::DRAW_LINE(edge5.x, edge5.y, edge5.z, edge6.x, edge6.y, edge6.z, 0, green, blue, 200);
	GRAPHICS::DRAW_LINE(edge5.x, edge5.y, edge5.z, edge8.x, edge8.y, edge8.z, 0, green, blue, 200);
	GRAPHICS::DRAW_LINE(edge6.x, edge6.y, edge6.z, edge7.x, edge7.y, edge7.z, 0, green, blue, 200);
	GRAPHICS::DRAW_LINE(edge7.x, edge7.y, edge7.z, edge8.x, edge8.y, edge8.z, 0, green, blue, 200);

	GRAPHICS::DRAW_LINE(edge1.x, edge1.y, edge1.z, edge7.x, edge7.y, edge7.z, 0, green, blue, 200);
	GRAPHICS::DRAW_LINE(edge2.x, edge2.y, edge2.z, edge8.x, edge8.y, edge8.z, 0, green, blue, 200);
	GRAPHICS::DRAW_LINE(edge3.x, edge3.y, edge3.z, edge5.x, edge5.y, edge5.z, 0, green, blue, 200);
	GRAPHICS::DRAW_LINE(edge4.x, edge4.y, edge4.z, edge6.x, edge6.y, edge6.z, 0, green, blue, 200);
}




static void getVehicleVectorsFromTransform(Vector3 xVector, Vector3 yVector, Vector3 zVector, Vector3 &forwardVector, Vector3 &rightVector, Vector3 &upVector) {
	// TODO this relies on having
	//Vector3 worldX; worldX.x = 1; worldX.y = 0; worldX.z = 0;
	//Vector3 worldY; worldY.x = 0; worldY.y = 1; worldY.z = 0;
	//Vector3 worldZ; worldZ.x = 0; worldZ.y = 0; worldZ.z = 1;
	
	// In general this could/should be replaced with basis transformations
	// In a sense this is the inversion of the call to convertCoordinateSystem()
	
	forwardVector.x = xVector.y;
	forwardVector.y = yVector.y; 
	forwardVector.z = zVector.y; 

	rightVector.x = xVector.x;
	rightVector.y = yVector.x;
	rightVector.z = zVector.x;

	upVector.x = xVector.z;
	upVector.y = yVector.z;
	upVector.z = zVector.z;
}


static Vector3 rotationAroundAngles(Vector3 vec, Vector3 rotation) {
	// TODO there is an error in here, don't use this function

	// This implements a rotation of a vector by the x,y,z axis around the angles specified by rotation. 
	// TODO In principle this implements a rotation matrix, in the future it should be done with a rotation matrix
	// General roation given by https://www.symbolab.com/solver/step-by-step/%5Cbegin%7Bpmatrix%7D1%260%260%5C%5C%200%26cos%5Cleft(a%5Cright)%26-sin%5Cleft(a%5Cright)%5C%5C%200%26sin%5Cleft(a%5Cright)%26cos%5Cleft(a%5Cright)%5Cend%7Bpmatrix%7D%5Cbegin%7Bpmatrix%7Dcos%5Cleft(b%5Cright)%260%26sin%5Cleft(b%5Cright)%5C%5C%200%261%260%5C%5C%20-sin%5Cleft(b%5Cright)%260%26cos%5Cleft(b%5Cright)%5Cend%7Bpmatrix%7D%5Cbegin%7Bpmatrix%7Dcos%5Cleft(c%5Cright)%26-sin%5Cleft(c%5Cright)%260%5C%5C%20sin%5Cleft(c%5Cright)%26cos%5Cleft(c%5Cright)%260%5C%5C%200%260%261%5Cend%7Bpmatrix%7D

	Vector3 ret;

	float a = rotation.x * D2R;
	float b = rotation.y * D2R;
	float c = rotation.z * D2R;

	ret.x = cos(b) * cos(c) * vec.x + -cos(b) * sin(c) * vec.y + sin(b) * vec.z;
	ret.y = (sin(a) * sin(b) * cos(c) + cos(a) * sin(c)) * vec.x + (cos(a) * cos(c) - sin(a) * sin(b) * sin(c)) * vec.y + (-sin(a) * cos(b)) * vec.z;
	ret.z = (sin(a) * sin(c) - cos(a) * sin(b) * cos(c)) * vec.x + (cos(a) * sin(b) * sin(c) + sin(a) * cos(c)) * vec.y + cos(a) * cos(b) * vec.z;

	log("rotationAroundAngles");
	log("a,b,c (Deg): " + std::to_string(rotation.x) + ", " + std::to_string(rotation.y) + ", " + std::to_string(rotation.z));
	log("a,b,c: " + std::to_string(a) + ", " + std::to_string(b) + ", " + std::to_string(c));
	log("vec: " + std::to_string(vec.x) + ", " + std::to_string(vec.y) + ", " + std::to_string(vec.z));
	log("ret: " + std::to_string(ret.x) + ", " + std::to_string(ret.y) + ", " + std::to_string(ret.z));


	return ret;

}

static Vector3 rotationAroundX(Vector3 vec, Vector3 rotation) {
	float a = rotation.x * D2R;

	Vector3 ret;
	ret.x = vec.x;
	ret.y = cos(a) * vec.y - sin(a) * vec.z;
	ret.z = sin(a) * vec.y + cos(a) * vec.z;

	log("rotationAroundX");
	log("a: " + std::to_string(rotation.x));
	log("sin(a): " + std::to_string(sin(a)));
	log("cos(a): " + std::to_string(cos(a)));
	log("vec: " + std::to_string(vec.x) + ", " + std::to_string(vec.y) + ", " + std::to_string(vec.z));
	log("ret: " + std::to_string(ret.x) + ", " + std::to_string(ret.y) + ", " + std::to_string(ret.z));

	return ret;
}



static Vector3 rotateVectorAroundAxis(Vector3 vec, Vector3 axis, float degree) {
	Eigen::Vector3f v(vec.x, vec.y, vec.z);
	Eigen::Vector3f a(axis.x, axis.y, axis.z);
	a.normalize();

	float d = degree * D2R;

	//Eigen::Quaternion<float> q(w, x, y, z);
	Eigen::Matrix3f m;
	m = Eigen::AngleAxisf(d, a);
	
	Eigen::Vector3f rotated = m * v;


	Vector3 ret; ret.x = rotated.x(); ret.y = rotated.y(); ret.z = rotated.z();
	return ret;
}