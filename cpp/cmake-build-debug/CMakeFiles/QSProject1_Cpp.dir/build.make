# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/QSProject1_Cpp.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/QSProject1_Cpp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/QSProject1_Cpp.dir/flags.make

CMakeFiles/QSProject1_Cpp.dir/main.cpp.o: CMakeFiles/QSProject1_Cpp.dir/flags.make
CMakeFiles/QSProject1_Cpp.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/QSProject1_Cpp.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/QSProject1_Cpp.dir/main.cpp.o -c /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/main.cpp

CMakeFiles/QSProject1_Cpp.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/QSProject1_Cpp.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/main.cpp > CMakeFiles/QSProject1_Cpp.dir/main.cpp.i

CMakeFiles/QSProject1_Cpp.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/QSProject1_Cpp.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/main.cpp -o CMakeFiles/QSProject1_Cpp.dir/main.cpp.s

CMakeFiles/QSProject1_Cpp.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/QSProject1_Cpp.dir/main.cpp.o.requires

CMakeFiles/QSProject1_Cpp.dir/main.cpp.o.provides: CMakeFiles/QSProject1_Cpp.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/QSProject1_Cpp.dir/build.make CMakeFiles/QSProject1_Cpp.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/QSProject1_Cpp.dir/main.cpp.o.provides

CMakeFiles/QSProject1_Cpp.dir/main.cpp.o.provides.build: CMakeFiles/QSProject1_Cpp.dir/main.cpp.o


CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.o: CMakeFiles/QSProject1_Cpp.dir/flags.make
CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.o: ../featureGeneration.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.o -c /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/featureGeneration.cpp

CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/featureGeneration.cpp > CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.i

CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/featureGeneration.cpp -o CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.s

CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.o.requires:

.PHONY : CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.o.requires

CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.o.provides: CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.o.requires
	$(MAKE) -f CMakeFiles/QSProject1_Cpp.dir/build.make CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.o.provides.build
.PHONY : CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.o.provides

CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.o.provides.build: CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.o


CMakeFiles/QSProject1_Cpp.dir/models.cpp.o: CMakeFiles/QSProject1_Cpp.dir/flags.make
CMakeFiles/QSProject1_Cpp.dir/models.cpp.o: ../models.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/QSProject1_Cpp.dir/models.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/QSProject1_Cpp.dir/models.cpp.o -c /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/models.cpp

CMakeFiles/QSProject1_Cpp.dir/models.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/QSProject1_Cpp.dir/models.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/models.cpp > CMakeFiles/QSProject1_Cpp.dir/models.cpp.i

CMakeFiles/QSProject1_Cpp.dir/models.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/QSProject1_Cpp.dir/models.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/models.cpp -o CMakeFiles/QSProject1_Cpp.dir/models.cpp.s

CMakeFiles/QSProject1_Cpp.dir/models.cpp.o.requires:

.PHONY : CMakeFiles/QSProject1_Cpp.dir/models.cpp.o.requires

CMakeFiles/QSProject1_Cpp.dir/models.cpp.o.provides: CMakeFiles/QSProject1_Cpp.dir/models.cpp.o.requires
	$(MAKE) -f CMakeFiles/QSProject1_Cpp.dir/build.make CMakeFiles/QSProject1_Cpp.dir/models.cpp.o.provides.build
.PHONY : CMakeFiles/QSProject1_Cpp.dir/models.cpp.o.provides

CMakeFiles/QSProject1_Cpp.dir/models.cpp.o.provides.build: CMakeFiles/QSProject1_Cpp.dir/models.cpp.o


CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.o: CMakeFiles/QSProject1_Cpp.dir/flags.make
CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.o: ../dataset.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.o -c /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/dataset.cpp

CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/dataset.cpp > CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.i

CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/dataset.cpp -o CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.s

CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.o.requires:

.PHONY : CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.o.requires

CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.o.provides: CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.o.requires
	$(MAKE) -f CMakeFiles/QSProject1_Cpp.dir/build.make CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.o.provides.build
.PHONY : CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.o.provides

CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.o.provides.build: CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.o


# Object files for target QSProject1_Cpp
QSProject1_Cpp_OBJECTS = \
"CMakeFiles/QSProject1_Cpp.dir/main.cpp.o" \
"CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.o" \
"CMakeFiles/QSProject1_Cpp.dir/models.cpp.o" \
"CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.o"

# External object files for target QSProject1_Cpp
QSProject1_Cpp_EXTERNAL_OBJECTS =

QSProject1_Cpp: CMakeFiles/QSProject1_Cpp.dir/main.cpp.o
QSProject1_Cpp: CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.o
QSProject1_Cpp: CMakeFiles/QSProject1_Cpp.dir/models.cpp.o
QSProject1_Cpp: CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.o
QSProject1_Cpp: CMakeFiles/QSProject1_Cpp.dir/build.make
QSProject1_Cpp: /usr/local/lib/libopencv_gapi.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_stitching.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_aruco.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_bgsegm.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_bioinspired.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_ccalib.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_dnn_objdetect.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_dpm.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_face.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_freetype.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_fuzzy.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_hfs.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_img_hash.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_line_descriptor.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_quality.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_reg.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_rgbd.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_saliency.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_sfm.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_stereo.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_structured_light.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_superres.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_surface_matching.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_tracking.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_videostab.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_xfeatures2d.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_xobjdetect.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_xphoto.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_shape.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_datasets.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_plot.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_text.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_dnn.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_ml.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_phase_unwrapping.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_optflow.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_ximgproc.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_video.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_objdetect.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_calib3d.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_features2d.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_flann.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_highgui.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_videoio.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_imgcodecs.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_photo.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_imgproc.4.1.0.dylib
QSProject1_Cpp: /usr/local/lib/libopencv_core.4.1.0.dylib
QSProject1_Cpp: CMakeFiles/QSProject1_Cpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable QSProject1_Cpp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/QSProject1_Cpp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/QSProject1_Cpp.dir/build: QSProject1_Cpp

.PHONY : CMakeFiles/QSProject1_Cpp.dir/build

CMakeFiles/QSProject1_Cpp.dir/requires: CMakeFiles/QSProject1_Cpp.dir/main.cpp.o.requires
CMakeFiles/QSProject1_Cpp.dir/requires: CMakeFiles/QSProject1_Cpp.dir/featureGeneration.cpp.o.requires
CMakeFiles/QSProject1_Cpp.dir/requires: CMakeFiles/QSProject1_Cpp.dir/models.cpp.o.requires
CMakeFiles/QSProject1_Cpp.dir/requires: CMakeFiles/QSProject1_Cpp.dir/dataset.cpp.o.requires

.PHONY : CMakeFiles/QSProject1_Cpp.dir/requires

CMakeFiles/QSProject1_Cpp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/QSProject1_Cpp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/QSProject1_Cpp.dir/clean

CMakeFiles/QSProject1_Cpp.dir/depend:
	cd /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/cmake-build-debug /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/cmake-build-debug /Users/quan/Desktop/2019-semester/2/MachineLearning/QSProject1-Cpp/cmake-build-debug/CMakeFiles/QSProject1_Cpp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/QSProject1_Cpp.dir/depend

