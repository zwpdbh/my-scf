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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.10.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.10.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/zw/code/C_and_C++_Projects/parallel-kNN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/zw/code/C_and_C++_Projects/parallel-kNN/build

# Include any dependencies generated for this target.
include CMakeFiles/matrix_multiplication.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/matrix_multiplication.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/matrix_multiplication.dir/flags.make

CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.o: CMakeFiles/matrix_multiplication.dir/flags.make
CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.o: ../pthread_demo/matrix_multiplication.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zw/code/C_and_C++_Projects/parallel-kNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.o"
	/usr/local/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.o -c /Users/zw/code/C_and_C++_Projects/parallel-kNN/pthread_demo/matrix_multiplication.cpp

CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.i"
	/usr/local/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zw/code/C_and_C++_Projects/parallel-kNN/pthread_demo/matrix_multiplication.cpp > CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.i

CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.s"
	/usr/local/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zw/code/C_and_C++_Projects/parallel-kNN/pthread_demo/matrix_multiplication.cpp -o CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.s

CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.o.requires:

.PHONY : CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.o.requires

CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.o.provides: CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.o.requires
	$(MAKE) -f CMakeFiles/matrix_multiplication.dir/build.make CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.o.provides.build
.PHONY : CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.o.provides

CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.o.provides.build: CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.o


# Object files for target matrix_multiplication
matrix_multiplication_OBJECTS = \
"CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.o"

# External object files for target matrix_multiplication
matrix_multiplication_EXTERNAL_OBJECTS =

../bin/matrix_multiplication: CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.o
../bin/matrix_multiplication: CMakeFiles/matrix_multiplication.dir/build.make
../bin/matrix_multiplication: CMakeFiles/matrix_multiplication.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/zw/code/C_and_C++_Projects/parallel-kNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/matrix_multiplication"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matrix_multiplication.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/matrix_multiplication.dir/build: ../bin/matrix_multiplication

.PHONY : CMakeFiles/matrix_multiplication.dir/build

CMakeFiles/matrix_multiplication.dir/requires: CMakeFiles/matrix_multiplication.dir/pthread_demo/matrix_multiplication.cpp.o.requires

.PHONY : CMakeFiles/matrix_multiplication.dir/requires

CMakeFiles/matrix_multiplication.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/matrix_multiplication.dir/cmake_clean.cmake
.PHONY : CMakeFiles/matrix_multiplication.dir/clean

CMakeFiles/matrix_multiplication.dir/depend:
	cd /Users/zw/code/C_and_C++_Projects/parallel-kNN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/zw/code/C_and_C++_Projects/parallel-kNN /Users/zw/code/C_and_C++_Projects/parallel-kNN /Users/zw/code/C_and_C++_Projects/parallel-kNN/build /Users/zw/code/C_and_C++_Projects/parallel-kNN/build /Users/zw/code/C_and_C++_Projects/parallel-kNN/build/CMakeFiles/matrix_multiplication.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/matrix_multiplication.dir/depend

