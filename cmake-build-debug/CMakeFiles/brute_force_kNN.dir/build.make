# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

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
CMAKE_COMMAND = "/Users/zw/Library/Application Support/JetBrains/Toolbox/apps/CLion/ch-0/173.4674.29/CLion.app/Contents/bin/cmake/bin/cmake"

# The command to remove a file.
RM = "/Users/zw/Library/Application Support/JetBrains/Toolbox/apps/CLion/ch-0/173.4674.29/CLion.app/Contents/bin/cmake/bin/cmake" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/zw/code/C_and_C++_Projects/Parallel_kNN_with_SIFT

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/zw/code/C_and_C++_Projects/Parallel_kNN_with_SIFT/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/brute_force_kNN.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/brute_force_kNN.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/brute_force_kNN.dir/flags.make

CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.o: CMakeFiles/brute_force_kNN.dir/flags.make
CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.o: ../brute_force_kNN.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zw/code/C_and_C++_Projects/Parallel_kNN_with_SIFT/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.o"
	/usr/local/opt/llvm/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.o -c /Users/zw/code/C_and_C++_Projects/Parallel_kNN_with_SIFT/brute_force_kNN.cpp

CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.i"
	/usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zw/code/C_and_C++_Projects/Parallel_kNN_with_SIFT/brute_force_kNN.cpp > CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.i

CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.s"
	/usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zw/code/C_and_C++_Projects/Parallel_kNN_with_SIFT/brute_force_kNN.cpp -o CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.s

CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.o.requires:

.PHONY : CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.o.requires

CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.o.provides: CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.o.requires
	$(MAKE) -f CMakeFiles/brute_force_kNN.dir/build.make CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.o.provides.build
.PHONY : CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.o.provides

CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.o.provides.build: CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.o


CMakeFiles/brute_force_kNN.dir/kNN.cpp.o: CMakeFiles/brute_force_kNN.dir/flags.make
CMakeFiles/brute_force_kNN.dir/kNN.cpp.o: ../kNN.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zw/code/C_and_C++_Projects/Parallel_kNN_with_SIFT/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/brute_force_kNN.dir/kNN.cpp.o"
	/usr/local/opt/llvm/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/brute_force_kNN.dir/kNN.cpp.o -c /Users/zw/code/C_and_C++_Projects/Parallel_kNN_with_SIFT/kNN.cpp

CMakeFiles/brute_force_kNN.dir/kNN.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/brute_force_kNN.dir/kNN.cpp.i"
	/usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zw/code/C_and_C++_Projects/Parallel_kNN_with_SIFT/kNN.cpp > CMakeFiles/brute_force_kNN.dir/kNN.cpp.i

CMakeFiles/brute_force_kNN.dir/kNN.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/brute_force_kNN.dir/kNN.cpp.s"
	/usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zw/code/C_and_C++_Projects/Parallel_kNN_with_SIFT/kNN.cpp -o CMakeFiles/brute_force_kNN.dir/kNN.cpp.s

CMakeFiles/brute_force_kNN.dir/kNN.cpp.o.requires:

.PHONY : CMakeFiles/brute_force_kNN.dir/kNN.cpp.o.requires

CMakeFiles/brute_force_kNN.dir/kNN.cpp.o.provides: CMakeFiles/brute_force_kNN.dir/kNN.cpp.o.requires
	$(MAKE) -f CMakeFiles/brute_force_kNN.dir/build.make CMakeFiles/brute_force_kNN.dir/kNN.cpp.o.provides.build
.PHONY : CMakeFiles/brute_force_kNN.dir/kNN.cpp.o.provides

CMakeFiles/brute_force_kNN.dir/kNN.cpp.o.provides.build: CMakeFiles/brute_force_kNN.dir/kNN.cpp.o


# Object files for target brute_force_kNN
brute_force_kNN_OBJECTS = \
"CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.o" \
"CMakeFiles/brute_force_kNN.dir/kNN.cpp.o"

# External object files for target brute_force_kNN
brute_force_kNN_EXTERNAL_OBJECTS =

../bin/brute_force_kNN: CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.o
../bin/brute_force_kNN: CMakeFiles/brute_force_kNN.dir/kNN.cpp.o
../bin/brute_force_kNN: CMakeFiles/brute_force_kNN.dir/build.make
../bin/brute_force_kNN: CMakeFiles/brute_force_kNN.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/zw/code/C_and_C++_Projects/Parallel_kNN_with_SIFT/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../bin/brute_force_kNN"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/brute_force_kNN.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/brute_force_kNN.dir/build: ../bin/brute_force_kNN

.PHONY : CMakeFiles/brute_force_kNN.dir/build

CMakeFiles/brute_force_kNN.dir/requires: CMakeFiles/brute_force_kNN.dir/brute_force_kNN.cpp.o.requires
CMakeFiles/brute_force_kNN.dir/requires: CMakeFiles/brute_force_kNN.dir/kNN.cpp.o.requires

.PHONY : CMakeFiles/brute_force_kNN.dir/requires

CMakeFiles/brute_force_kNN.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/brute_force_kNN.dir/cmake_clean.cmake
.PHONY : CMakeFiles/brute_force_kNN.dir/clean

CMakeFiles/brute_force_kNN.dir/depend:
	cd /Users/zw/code/C_and_C++_Projects/Parallel_kNN_with_SIFT/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/zw/code/C_and_C++_Projects/Parallel_kNN_with_SIFT /Users/zw/code/C_and_C++_Projects/Parallel_kNN_with_SIFT /Users/zw/code/C_and_C++_Projects/Parallel_kNN_with_SIFT/cmake-build-debug /Users/zw/code/C_and_C++_Projects/Parallel_kNN_with_SIFT/cmake-build-debug /Users/zw/code/C_and_C++_Projects/Parallel_kNN_with_SIFT/cmake-build-debug/CMakeFiles/brute_force_kNN.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/brute_force_kNN.dir/depend

