SET(PROJECT_NAME ScaViSLAM)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/CMakeModules/")

SET (CMAKE_VERBOSE_MAKEFILE ON)

SET(CUDA_SUPPORT ON)
SET(PCL_SUPPORT OFF)

IF(CUDA_SUPPORT)
  ADD_DEFINITIONS(-DSCAVISLAM_CUDA_SUPPORT)
ENDIF()

IF(PCL_SUPPORT)
  ADD_DEFINITIONS(-DSCAVISLAM_PCL_SUPPORT)
  FIND_PACKAGE(PCL 1.2 REQUIRED)
  ADD_DEFINITIONS(${PCL_DEFINITIONS})
ENDIF()

IF( NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE Release )
ENDIF()

IF (CMAKE_COMPILER_IS_GNUCXX )
   SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")
   SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -g ")

  ADD_DEFINITIONS("-Wall -Werror -Wno-unused-variable
                   -Wno-unused-but-set-variable -Wno-unknown-pragmas ")
ENDIF()


SET (INCLUDE_DIRS "../EXTERNAL/eigen3/"
                  "../EXTERNAL/opencv2/include/opencv"
                  "../EXTERNAL/opencv2/include/"
                  "../EXTERNAL/opencv2/modules/core/include"
                  "../EXTERNAL/opencv2/modules/imgproc/include"
                  "../EXTERNAL/opencv2/modules/video/include"
                  "../EXTERNAL/opencv2/modules/features2d/include"
                  "../EXTERNAL/opencv2/modules/flann/include"
                  "../EXTERNAL/opencv2/modules/objdetect/include"
                  "../EXTERNAL/opencv2/modules/calib3d/include"
                  "../EXTERNAL/opencv2/modules/ml/include"
                  "../EXTERNAL/opencv2/modules/highgui/include"
                  "../EXTERNAL/opencv2/modules/legacy/include"
                  "../EXTERNAL/opencv2/modules/contrib/include"
                  "../EXTERNAL/g2o/svs_build"
                  "../EXTERNAL/g2o/"
                  "../EXTERNAL/g2o/EXTERNAL/csparse/"
                  "../EXTERNAL/Pangolin/"
                  "../EXTERNAL/Pangolin/svs_build"
                  "../EXTERNAL/Sophus/src"
                  "../EXTERNAL/Sophus/svs_build"
                  "../EXTERNAL/VisionTools/src"
                  "../EXTERNAL/VisionTools/svs_build")

 LIST(APPEND CMAKE_LIBRARY_PATH ../EXTERNAL/g2o/lib
                                ../EXTERNAL/opencv2/svs_build/lib/
                                ../EXTERNAL/Pangolin/svs_build/pangolin
                                ../EXTERNAL/Sophus/svs_build/
                                ../EXTERNAL/VisionTools/svs_build/)

SET (LINK_DIRS )
SET (LIB_NAMES GL pangolin glut g2o_stuff g2o_core g2o_solver_csparse
               VisionTools Sophus
               boost_system boost_thread boost_filesystem boost_regex
               opencv_core opencv_features2d opencv_flann opencv_gpu
               opencv_highgui opencv_imgproc opencv_calib3d)

IF(CUDA_SUPPORT)
  LIST(APPEND INCLUDE_DIRS "../EXTERNAL/opencv2/modules/gpu/include")
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
              dense_tracking)


SET (TEMPLATE ransac
              frame_grabber
              matcher
              slam_graph )

SET (SOURCES draw_items.h
             global.h   
             keyframes.h
             quadtree.h
             transformations.h
             data_structures.h
             pose_optimizer.h)

IF(PCL_SUPPORT)
  LIST(APPEND CLASSES rgbd_grabber)
ENDIF()

IF(CUDA_SUPPORT)
  LIST(APPEND SOURCES gpu/dense_tracking.cuh
                      gpu/dense_tracking.cu)
ENDIF()


FOREACH(class ${CLASSES})
  LIST(APPEND SOURCES ${class}.cpp ${class}.h)
ENDFOREACH(class)

FOREACH(template ${TEMPLATE})
  LIST(APPEND SOURCES ${template}.cpp   ${template}.hpp
                      ${template}-impl.cpp )
ENDFOREACH(template)

LINK_DIRECTORIES(${LINK_DIRS} ${PCL_LIBRARY_DIRS})
LINK_LIBRARIES (${PROJECT_NAME} ${LIBS} ${PCL_LIBRARIES})

IF(CUDA_SUPPORT)
  LIST(APPEND INCLUDE_DIRS ${CUDA_SDK_ROOT_DIR}/common/inc ${CUDA_INCLUDE_DIRS})
  CUDA_ADD_EXECUTABLE(stereo_slam stereo_slam.cpp)
  CUDA_ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCES})
ELSE()
  ADD_EXECUTABLE(stereo_slam stereo_slam.cpp)
  ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCES})
ENDIF()

INCLUDE_DIRECTORIES(${INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
