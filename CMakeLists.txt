SET(PROJECT_NAME ScaViSLAM)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/CMakeModules/")

SET (OpenCV_DIR ${CMAKE_PREFIX_PATH}/share/OpenCV)

SET (CMAKE_VERBOSE_MAKEFILE ON)

IF(NOT CUDA_SUPPORT)
  SET(CUDA_SUPPORT ON)
ENDIF()

IF( NOT PCL_SUPPORT)
  SET(PCL_SUPPORT OFF)
ENDIF()


IF(CUDA_SUPPORT)
  MESSAGE ("CUDA SUPPORT ON")
  ADD_DEFINITIONS(-DSCAVISLAM_CUDA_SUPPORT)
ELSE()
  MESSAGE ("CUDA SUPPORT OFF")
ENDIF()

IF(PCL_SUPPORT)
  MESSAGE ("PCL SUPPORT ON")
  ADD_DEFINITIONS(-DSCAVISLAM_PCL_SUPPORT)
  FIND_PACKAGE(PCL 1.2 REQUIRED)
  ADD_DEFINITIONS(${PCL_DEFINITIONS})
ELSE()
  MESSAGE ("PCL SUPPORT OFF")
ENDIF()

IF( NOT CMAKE_BUILD_TYPE )
  SET( CMAKE_BUILD_TYPE Release )
ENDIF()

IF (CMAKE_COMPILER_IS_GNUCXX )
  SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")

  IF( NOT SSE4_1_SUPPORT )
    SET(SSE4_1_SUPPORT OFF)
  ENDIF()
  IF(SSE4_1_SUPPORT)
    SET(SSE4_1_SUPPORT ON)
    MESSAGE ("SSE 4.1 SUPPORT ON")
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -msse4.1 -g")
  ELSE()
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")
    MESSAGE ("SSE 4.1  SUPPORT OFF")
  ENDIF()
  ADD_DEFINITIONS("-Wall -Werror -Wno-unused-variable
                   -Wno-unused-but-set-variable -Wno-unknown-pragmas")
ENDIF()

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(SuiteSparse REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
LIST(APPEND INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR} 
                         ${CSPARSE_INCLUDE_DIR})

LIST(APPEND CMAKE_LIBRARY_PATH EXTERNAL/VisionTools/svs_build/)

SET (LIB_NAMES GL pangolin glut g2o_stuff g2o_core g2o_solver_csparse csparse
               g2o_csparse_extension  VisionTools Sophus
               boost_system boost_thread boost_filesystem boost_regex
               opencv_core opencv_features2d opencv_flann opencv_gpu
               opencv_nonfree
               opencv_highgui opencv_imgproc opencv_calib3d)

IF(CUDA_SUPPORT)
  LIST(APPEND INCLUDE_DIRS "EXTERNAL/opencv2/modules/gpu/include")
  FIND_PACKAGE(CUDA REQUIRED)
  SET(CUDA_NVCC_FLAGS  "-arch=sm_20" "--use_fast_math" "-O3"
                       "--ptxas-options=--verbose" "-keep"  )
  ADD_DEFINITIONS(-DCUDA_BUILD -DBT_USE_DOUBLE_PRECISION)
ENDIF()


FOREACH(lib ${LIB_NAMES})
  FIND_LIBRARY(LIB_${lib} ${lib})
  IF (LIB_${lib})
    MESSAGE(STATUS "found library '${lib}': ${LIB_${lib}}")
  ELSE (LIB_${lib})
    MESSAGE(FATAL_ERROR "cannot find library '${lib}' - this will not work ...")
  ENDIF (LIB_${lib})
  LIST(APPEND LIBS ${LIB_${lib}})
ENDFOREACH(lib)

#Add class XYZ here which consits of two files: XYZ.cpp and XYZ.h
SET (CLASSES  placerecognizer
              maths_utils
              ransac_models
              homography
              g2o_types/anchored_points
              stereo_camera
              backend
              fast_grid
              stereo_frontend
              dense_tracking
              filegrabber)


SET (TEMPLATE ransac
              frame_grabber
              matcher
              slam_graph )

SET (SOURCE_DIR "scavislam")
SET (SOURCES ${SOURCE_DIR}/draw_items.h
             ${SOURCE_DIR}/global.h   
             ${SOURCE_DIR}/keyframes.h
             ${SOURCE_DIR}/quadtree.h
             ${SOURCE_DIR}/transformations.h
             ${SOURCE_DIR}/data_structures.h
             ${SOURCE_DIR}/pose_optimizer.h)

IF(PCL_SUPPORT)
  LIST(APPEND CLASSES rgbd_grabber)
ENDIF()

IF(CUDA_SUPPORT)
  LIST(APPEND SOURCES ${SOURCE_DIR}/gpu/dense_tracking.cuh
                      ${SOURCE_DIR}/gpu/dense_tracking.cu)
ENDIF()


FOREACH(class ${CLASSES})
  LIST(APPEND SOURCES ${SOURCE_DIR}/${class}.cpp ${SOURCE_DIR}/${class}.h)
ENDFOREACH(class)

FOREACH(template ${TEMPLATE})
  LIST(APPEND SOURCES ${SOURCE_DIR}/${template}.cpp   
                      ${SOURCE_DIR}/${template}.hpp
                      ${SOURCE_DIR}/${template}-impl.cpp )
ENDFOREACH(template)

LINK_DIRECTORIES(${LINK_DIRS} ${PCL_LIBRARY_DIRS})
LINK_LIBRARIES (${PROJECT_NAME} ${LIBS} ${PCL_LIBRARIES})

IF(CUDA_SUPPORT)
  LIST(APPEND INCLUDE_DIRS ${CUDA_SDK_ROOT_DIR}/common/inc ${CUDA_INCLUDE_DIRS})
  CUDA_ADD_EXECUTABLE(stereo_slam ${SOURCE_DIR}/stereo_slam.cpp)
  CUDA_ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCES})
ELSE()
  ADD_EXECUTABLE(stereo_slam ${SOURCE_DIR}/stereo_slam.cpp)
  ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCES})
ENDIF()
ADD_EXECUTABLE(create_dictionary ${SOURCE_DIR}/create_dictionary.cpp)
IF(PCL_SUPPORT)
  ADD_EXECUTABLE(dump_rgbd ${SOURCE_DIR}/dump_rgbd.cpp)
ENDIF()

INCLUDE_DIRECTORIES(${INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
