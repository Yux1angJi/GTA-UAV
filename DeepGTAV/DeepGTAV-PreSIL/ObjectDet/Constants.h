#pragma once

#define PI 3.1415926535898
#define D2R PI/180.0

static const Eigen::Vector3f WORLD_NORTH(0.0, 1.0, 0.0);
static const Eigen::Vector3f WORLD_UP(0.0, 0.0, 1.0);
static const Eigen::Vector3f WORLD_EAST(1.0, 0.0, 0.0);

//Floats per point in the pointcloud (x, y, z, reflectance)
// TODO actually I believe that it is (x, y, z, segmentationInstance), to see this examine the line 
//		*(p + 3) = m_pInstanceSeg[s_camParams.width * j + i];
// in LiDAR::GenerateSinglePoint
const int FLOATS_PER_POINT = 4;
const int OBJECT_MAX_DIST = 0; // 200; //in metres (label_aug will have objects past this value)


//const bool SET_PED_BOXES = true;
//const float PED_BOX_WIDTH = 1.2f;
//const float PED_BOX_LENGTH = 1.0f;
//const float PED_BOX_WALKING_LEN = 1.44f;
//
//const float BBOX_ADJUSTMENT_FACTOR = 1.1f;


// TODO settings when using the unprocessed bounding boxes as output bounding boxes 
const bool SET_PED_BOXES = true;
const float PED_BOX_WIDTH = 1.2f;
const float PED_BOX_LENGTH = 1.0f;
const float PED_BOX_WALKING_LEN = 1.44f;

const float BBOX_ADJUSTMENT_FACTOR = 1.1f;



//TODO: These should be moved to a settings file so don't need to rebuild when only changing settings

//Use the same time of day throughout the collection process
const bool SAME_TIME_OF_DAY = false;

//Some settings for testing pointcloud generation
//This prints the 2D map of where lidar beams hit on the screen (not consistent when moving)
const bool GENERATE_2D_POINTMAP = true;
//Outputs secondary pointcloud with raycast points
// TODO remove
const bool OUTPUT_RAYCAST_POINTS = true;
//Uses ray casting then transforms 3D point back to 2D plane to use depth buffer value
const bool USE_RAYCASTING = false;

//Outputs pointcloud with 1:1 ratio of pixels in image < max_lidar_dist
const bool OUTPUT_DM_POINTCLOUD = false;
//If OUTPUT_DM_POINTCLOUD, outputs all points, even those past max_lidar_dist (good for testing)
const bool OUTPUT_FULL_DM_POINTCLOUD = false;
//Output offset pointclouds (for testing alignment)
const bool OUTPUT_OFFSET_POINTCLOUDS = false;

//Sends animals to client. Check client also outputs them
const bool RETAIN_ANIMALS = true;

//For testing
const bool OUTPUT_OCCLUSION_IMAGE = true;
const bool OUTPUT_UNUSED_PIXELS_IMAGE = true;

//Can be used to get general outline of some objects with raycasting
//WARNING: NOT ALL VEHICLES ARE HIT WITH RAYCASTING
const bool OBTAIN_RAY_POINTS_HIT = false;
//Overlays adjusted points on top of pointcloud (using raycasting + depth map)
const bool OUTPUT_ADJUSTED_POINTS = false;

//2d points will be shrunk with stencil cull
const bool CORRECT_2D_POINTS_BEHIND_CAMERA = false;

const bool LIDAR_GAUSSIAN_NOISE = true;
const double DEPTH_NOISE_STDDEV = 0.006;//3 standard deviations is approximately 2cm
const double DEPTH_NOISE_MEAN = 0.0;

//Extends bboxes/segmentation for bike type vehicles with rider (ped) information
const bool PROCESS_PEDS_ON_BIKES = true;
const bool TESTING_PEDS_ON_BIKES = true;

// Whether to add peds in vehicles to their vehicles or seperate them from the vehicles
const bool DISTINGUISH_PEDS_IN_VEHICLES = true;

//Outputs stats on lidar vs depth map conversion (used for testing)
const bool OUTPUT_DEPTH_STATS = true;

//Maximum distance from ground to be considered ground point:
const float GROUND_POINT_MAX_DIST = 0.1;//in metres
//Prints out image in groundPointsImg for testing
const bool OUTPUT_GROUND_PIXELS = true;

//Outputs separate stencil segmentation images for each stencil value
const bool OUTPUT_SEPARATE_STENCILS = true;

//Outputs stencil image which shows stencil of some classes (for debugging)
const bool OUTPUT_STENCIL_IMAGE = true;

//Needs OUTPUT_SEPARATE_STENCILS to be true, only outputs log messages for values which are unknown 
const bool ONLY_OUTPUT_UNKNOWN_STENCILS = false;

////If set to true DeepGTAV outputs all information from nearby vehicles within specified range
//const bool GENERATE_SECONDARY_PERSPECTIVES = false;
//const int SECONDARY_PERSPECTIVE_RANGE = 100;
////Only generates secondary perspectives from occupied vehicles (prevents capturing parking garages)
////This should be set to true except for stationary scenarios
//const bool ONLY_OCCUPIED_VEHICLES = true;

const bool TRUPERCEPT_SCENARIO = false;

//Outputs self location (For finding spots for stationary scenes)
const bool OUTPUT_SELF_LOCATION = false;

//Outputs unprocessed labels file (for testing)
const bool OUTPUT_UNPROCESSED_LABELS = false;

////Processes overlapping points for segmentation images
////Warning!!!! There is a memory leak in here that needs to be investigated
//const bool PROCESS_OVERLAPPING_POINTS = true;

//Outputs all vehicles within range in augmented labels
const bool AUGMENT_ALL_VEHICLES_IN_RANGE = true;

//Only collects and writes to file system image data and bounding boxes to improve speed for this task
const bool ONLY_COLLECT_IMAGE_AND_BBOXES = false;

const bool DONT_COLLECT_IMAGE_AND_BBOXES_TO_FILE = false;


// Use this to print Debugging information to logfile at getenv("DEEPGTAV_LOG_FILE")
const bool DEBUG_LOGGING = true;

// Print Timestamps in LOG (for time testing)
const bool DEBUG_LOG_WITH_TIME = true;