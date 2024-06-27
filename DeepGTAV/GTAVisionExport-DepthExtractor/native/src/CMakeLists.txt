


find_package(Eigen3 REQUIRED)
set(GTAVision_src 
    main.cpp
    export.h
    export.cpp
    main.h
    enums.h
    nativeCaller.h
    natives.h
    types.h
)

add_library(GTAVisionNative SHARED ${GTAVision_src})
target_link_libraries(GTAVisionNative PUBLIC ${PROJECT_SOURCE_DIR}/deps/ScriptHookV.lib)
TARGET_INCLUDE_DIRECTORIES(GTAVisionNative PRIVATE ${EIGEN3_INCLUDE_DIR})
set_target_properties(GTAVisionNative PROPERTIES SUFFIX ".asi")

