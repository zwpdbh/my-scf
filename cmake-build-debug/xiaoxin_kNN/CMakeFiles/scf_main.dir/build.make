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
CMAKE_COMMAND = "/Users/zw/Library/Application Support/JetBrains/Toolbox/apps/CLion/ch-0/181.4668.70/CLion.app/Contents/bin/cmake/bin/cmake"

# The command to remove a file.
RM = "/Users/zw/Library/Application Support/JetBrains/Toolbox/apps/CLion/ch-0/181.4668.70/CLion.app/Contents/bin/cmake/bin/cmake" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/zw/code/C_and_C++_Projects/parallel-kNN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug

# Include any dependencies generated for this target.
include xiaoxin_kNN/CMakeFiles/scf_main.dir/depend.make

# Include the progress variables for this target.
include xiaoxin_kNN/CMakeFiles/scf_main.dir/progress.make

# Include the compile flags for this target's objects.
include xiaoxin_kNN/CMakeFiles/scf_main.dir/flags.make

xiaoxin_kNN/CMakeFiles/scf_main.dir/main.cpp.o: xiaoxin_kNN/CMakeFiles/scf_main.dir/flags.make
xiaoxin_kNN/CMakeFiles/scf_main.dir/main.cpp.o: ../xiaoxin_kNN/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object xiaoxin_kNN/CMakeFiles/scf_main.dir/main.cpp.o"
	cd /Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug/xiaoxin_kNN && /usr/local/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/scf_main.dir/main.cpp.o -c /Users/zw/code/C_and_C++_Projects/parallel-kNN/xiaoxin_kNN/main.cpp

xiaoxin_kNN/CMakeFiles/scf_main.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/scf_main.dir/main.cpp.i"
	cd /Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug/xiaoxin_kNN && /usr/local/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zw/code/C_and_C++_Projects/parallel-kNN/xiaoxin_kNN/main.cpp > CMakeFiles/scf_main.dir/main.cpp.i

xiaoxin_kNN/CMakeFiles/scf_main.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/scf_main.dir/main.cpp.s"
	cd /Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug/xiaoxin_kNN && /usr/local/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zw/code/C_and_C++_Projects/parallel-kNN/xiaoxin_kNN/main.cpp -o CMakeFiles/scf_main.dir/main.cpp.s

xiaoxin_kNN/CMakeFiles/scf_main.dir/main.cpp.o.requires:

.PHONY : xiaoxin_kNN/CMakeFiles/scf_main.dir/main.cpp.o.requires

xiaoxin_kNN/CMakeFiles/scf_main.dir/main.cpp.o.provides: xiaoxin_kNN/CMakeFiles/scf_main.dir/main.cpp.o.requires
	$(MAKE) -f xiaoxin_kNN/CMakeFiles/scf_main.dir/build.make xiaoxin_kNN/CMakeFiles/scf_main.dir/main.cpp.o.provides.build
.PHONY : xiaoxin_kNN/CMakeFiles/scf_main.dir/main.cpp.o.provides

xiaoxin_kNN/CMakeFiles/scf_main.dir/main.cpp.o.provides.build: xiaoxin_kNN/CMakeFiles/scf_main.dir/main.cpp.o


xiaoxin_kNN/CMakeFiles/scf_main.dir/csv_parser.cpp.o: xiaoxin_kNN/CMakeFiles/scf_main.dir/flags.make
xiaoxin_kNN/CMakeFiles/scf_main.dir/csv_parser.cpp.o: ../xiaoxin_kNN/csv_parser.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object xiaoxin_kNN/CMakeFiles/scf_main.dir/csv_parser.cpp.o"
	cd /Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug/xiaoxin_kNN && /usr/local/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/scf_main.dir/csv_parser.cpp.o -c /Users/zw/code/C_and_C++_Projects/parallel-kNN/xiaoxin_kNN/csv_parser.cpp

xiaoxin_kNN/CMakeFiles/scf_main.dir/csv_parser.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/scf_main.dir/csv_parser.cpp.i"
	cd /Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug/xiaoxin_kNN && /usr/local/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zw/code/C_and_C++_Projects/parallel-kNN/xiaoxin_kNN/csv_parser.cpp > CMakeFiles/scf_main.dir/csv_parser.cpp.i

xiaoxin_kNN/CMakeFiles/scf_main.dir/csv_parser.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/scf_main.dir/csv_parser.cpp.s"
	cd /Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug/xiaoxin_kNN && /usr/local/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zw/code/C_and_C++_Projects/parallel-kNN/xiaoxin_kNN/csv_parser.cpp -o CMakeFiles/scf_main.dir/csv_parser.cpp.s

xiaoxin_kNN/CMakeFiles/scf_main.dir/csv_parser.cpp.o.requires:

.PHONY : xiaoxin_kNN/CMakeFiles/scf_main.dir/csv_parser.cpp.o.requires

xiaoxin_kNN/CMakeFiles/scf_main.dir/csv_parser.cpp.o.provides: xiaoxin_kNN/CMakeFiles/scf_main.dir/csv_parser.cpp.o.requires
	$(MAKE) -f xiaoxin_kNN/CMakeFiles/scf_main.dir/build.make xiaoxin_kNN/CMakeFiles/scf_main.dir/csv_parser.cpp.o.provides.build
.PHONY : xiaoxin_kNN/CMakeFiles/scf_main.dir/csv_parser.cpp.o.provides

xiaoxin_kNN/CMakeFiles/scf_main.dir/csv_parser.cpp.o.provides.build: xiaoxin_kNN/CMakeFiles/scf_main.dir/csv_parser.cpp.o


xiaoxin_kNN/CMakeFiles/scf_main.dir/sdc_index.cpp.o: xiaoxin_kNN/CMakeFiles/scf_main.dir/flags.make
xiaoxin_kNN/CMakeFiles/scf_main.dir/sdc_index.cpp.o: ../xiaoxin_kNN/sdc_index.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object xiaoxin_kNN/CMakeFiles/scf_main.dir/sdc_index.cpp.o"
	cd /Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug/xiaoxin_kNN && /usr/local/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/scf_main.dir/sdc_index.cpp.o -c /Users/zw/code/C_and_C++_Projects/parallel-kNN/xiaoxin_kNN/sdc_index.cpp

xiaoxin_kNN/CMakeFiles/scf_main.dir/sdc_index.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/scf_main.dir/sdc_index.cpp.i"
	cd /Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug/xiaoxin_kNN && /usr/local/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zw/code/C_and_C++_Projects/parallel-kNN/xiaoxin_kNN/sdc_index.cpp > CMakeFiles/scf_main.dir/sdc_index.cpp.i

xiaoxin_kNN/CMakeFiles/scf_main.dir/sdc_index.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/scf_main.dir/sdc_index.cpp.s"
	cd /Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug/xiaoxin_kNN && /usr/local/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zw/code/C_and_C++_Projects/parallel-kNN/xiaoxin_kNN/sdc_index.cpp -o CMakeFiles/scf_main.dir/sdc_index.cpp.s

xiaoxin_kNN/CMakeFiles/scf_main.dir/sdc_index.cpp.o.requires:

.PHONY : xiaoxin_kNN/CMakeFiles/scf_main.dir/sdc_index.cpp.o.requires

xiaoxin_kNN/CMakeFiles/scf_main.dir/sdc_index.cpp.o.provides: xiaoxin_kNN/CMakeFiles/scf_main.dir/sdc_index.cpp.o.requires
	$(MAKE) -f xiaoxin_kNN/CMakeFiles/scf_main.dir/build.make xiaoxin_kNN/CMakeFiles/scf_main.dir/sdc_index.cpp.o.provides.build
.PHONY : xiaoxin_kNN/CMakeFiles/scf_main.dir/sdc_index.cpp.o.provides

xiaoxin_kNN/CMakeFiles/scf_main.dir/sdc_index.cpp.o.provides.build: xiaoxin_kNN/CMakeFiles/scf_main.dir/sdc_index.cpp.o


# Object files for target scf_main
scf_main_OBJECTS = \
"CMakeFiles/scf_main.dir/main.cpp.o" \
"CMakeFiles/scf_main.dir/csv_parser.cpp.o" \
"CMakeFiles/scf_main.dir/sdc_index.cpp.o"

# External object files for target scf_main
scf_main_EXTERNAL_OBJECTS =

../bin/scf_main: xiaoxin_kNN/CMakeFiles/scf_main.dir/main.cpp.o
../bin/scf_main: xiaoxin_kNN/CMakeFiles/scf_main.dir/csv_parser.cpp.o
../bin/scf_main: xiaoxin_kNN/CMakeFiles/scf_main.dir/sdc_index.cpp.o
../bin/scf_main: xiaoxin_kNN/CMakeFiles/scf_main.dir/build.make
../bin/scf_main: xiaoxin_kNN/CMakeFiles/scf_main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable ../../bin/scf_main"
	cd /Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug/xiaoxin_kNN && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/scf_main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
xiaoxin_kNN/CMakeFiles/scf_main.dir/build: ../bin/scf_main

.PHONY : xiaoxin_kNN/CMakeFiles/scf_main.dir/build

xiaoxin_kNN/CMakeFiles/scf_main.dir/requires: xiaoxin_kNN/CMakeFiles/scf_main.dir/main.cpp.o.requires
xiaoxin_kNN/CMakeFiles/scf_main.dir/requires: xiaoxin_kNN/CMakeFiles/scf_main.dir/csv_parser.cpp.o.requires
xiaoxin_kNN/CMakeFiles/scf_main.dir/requires: xiaoxin_kNN/CMakeFiles/scf_main.dir/sdc_index.cpp.o.requires

.PHONY : xiaoxin_kNN/CMakeFiles/scf_main.dir/requires

xiaoxin_kNN/CMakeFiles/scf_main.dir/clean:
	cd /Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug/xiaoxin_kNN && $(CMAKE_COMMAND) -P CMakeFiles/scf_main.dir/cmake_clean.cmake
.PHONY : xiaoxin_kNN/CMakeFiles/scf_main.dir/clean

xiaoxin_kNN/CMakeFiles/scf_main.dir/depend:
	cd /Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/zw/code/C_and_C++_Projects/parallel-kNN /Users/zw/code/C_and_C++_Projects/parallel-kNN/xiaoxin_kNN /Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug /Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug/xiaoxin_kNN /Users/zw/code/C_and_C++_Projects/parallel-kNN/cmake-build-debug/xiaoxin_kNN/CMakeFiles/scf_main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : xiaoxin_kNN/CMakeFiles/scf_main.dir/depend

