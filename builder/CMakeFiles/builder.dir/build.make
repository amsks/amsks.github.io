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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mclovin/git/amsks.github.io/builder

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mclovin/git/amsks.github.io/builder

# Include any dependencies generated for this target.
include CMakeFiles/builder.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/builder.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/builder.dir/flags.make

CMakeFiles/builder.dir/builder.cpp.o: CMakeFiles/builder.dir/flags.make
CMakeFiles/builder.dir/builder.cpp.o: builder.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mclovin/git/amsks.github.io/builder/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/builder.dir/builder.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/builder.dir/builder.cpp.o -c /home/mclovin/git/amsks.github.io/builder/builder.cpp

CMakeFiles/builder.dir/builder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/builder.dir/builder.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mclovin/git/amsks.github.io/builder/builder.cpp > CMakeFiles/builder.dir/builder.cpp.i

CMakeFiles/builder.dir/builder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/builder.dir/builder.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mclovin/git/amsks.github.io/builder/builder.cpp -o CMakeFiles/builder.dir/builder.cpp.s

CMakeFiles/builder.dir/builder.cpp.o.requires:

.PHONY : CMakeFiles/builder.dir/builder.cpp.o.requires

CMakeFiles/builder.dir/builder.cpp.o.provides: CMakeFiles/builder.dir/builder.cpp.o.requires
	$(MAKE) -f CMakeFiles/builder.dir/build.make CMakeFiles/builder.dir/builder.cpp.o.provides.build
.PHONY : CMakeFiles/builder.dir/builder.cpp.o.provides

CMakeFiles/builder.dir/builder.cpp.o.provides.build: CMakeFiles/builder.dir/builder.cpp.o


CMakeFiles/builder.dir/utf8.c.o: CMakeFiles/builder.dir/flags.make
CMakeFiles/builder.dir/utf8.c.o: utf8.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mclovin/git/amsks.github.io/builder/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/builder.dir/utf8.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/builder.dir/utf8.c.o   -c /home/mclovin/git/amsks.github.io/builder/utf8.c

CMakeFiles/builder.dir/utf8.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/builder.dir/utf8.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/mclovin/git/amsks.github.io/builder/utf8.c > CMakeFiles/builder.dir/utf8.c.i

CMakeFiles/builder.dir/utf8.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/builder.dir/utf8.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/mclovin/git/amsks.github.io/builder/utf8.c -o CMakeFiles/builder.dir/utf8.c.s

CMakeFiles/builder.dir/utf8.c.o.requires:

.PHONY : CMakeFiles/builder.dir/utf8.c.o.requires

CMakeFiles/builder.dir/utf8.c.o.provides: CMakeFiles/builder.dir/utf8.c.o.requires
	$(MAKE) -f CMakeFiles/builder.dir/build.make CMakeFiles/builder.dir/utf8.c.o.provides.build
.PHONY : CMakeFiles/builder.dir/utf8.c.o.provides

CMakeFiles/builder.dir/utf8.c.o.provides.build: CMakeFiles/builder.dir/utf8.c.o


CMakeFiles/builder.dir/duktape/duktape.c.o: CMakeFiles/builder.dir/flags.make
CMakeFiles/builder.dir/duktape/duktape.c.o: duktape/duktape.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mclovin/git/amsks.github.io/builder/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/builder.dir/duktape/duktape.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/builder.dir/duktape/duktape.c.o   -c /home/mclovin/git/amsks.github.io/builder/duktape/duktape.c

CMakeFiles/builder.dir/duktape/duktape.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/builder.dir/duktape/duktape.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/mclovin/git/amsks.github.io/builder/duktape/duktape.c > CMakeFiles/builder.dir/duktape/duktape.c.i

CMakeFiles/builder.dir/duktape/duktape.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/builder.dir/duktape/duktape.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/mclovin/git/amsks.github.io/builder/duktape/duktape.c -o CMakeFiles/builder.dir/duktape/duktape.c.s

CMakeFiles/builder.dir/duktape/duktape.c.o.requires:

.PHONY : CMakeFiles/builder.dir/duktape/duktape.c.o.requires

CMakeFiles/builder.dir/duktape/duktape.c.o.provides: CMakeFiles/builder.dir/duktape/duktape.c.o.requires
	$(MAKE) -f CMakeFiles/builder.dir/build.make CMakeFiles/builder.dir/duktape/duktape.c.o.provides.build
.PHONY : CMakeFiles/builder.dir/duktape/duktape.c.o.provides

CMakeFiles/builder.dir/duktape/duktape.c.o.provides.build: CMakeFiles/builder.dir/duktape/duktape.c.o


# Object files for target builder
builder_OBJECTS = \
"CMakeFiles/builder.dir/builder.cpp.o" \
"CMakeFiles/builder.dir/utf8.c.o" \
"CMakeFiles/builder.dir/duktape/duktape.c.o"

# External object files for target builder
builder_EXTERNAL_OBJECTS =

builder: CMakeFiles/builder.dir/builder.cpp.o
builder: CMakeFiles/builder.dir/utf8.c.o
builder: CMakeFiles/builder.dir/duktape/duktape.c.o
builder: CMakeFiles/builder.dir/build.make
builder: CMakeFiles/builder.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mclovin/git/amsks.github.io/builder/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable builder"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/builder.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/builder.dir/build: builder

.PHONY : CMakeFiles/builder.dir/build

CMakeFiles/builder.dir/requires: CMakeFiles/builder.dir/builder.cpp.o.requires
CMakeFiles/builder.dir/requires: CMakeFiles/builder.dir/utf8.c.o.requires
CMakeFiles/builder.dir/requires: CMakeFiles/builder.dir/duktape/duktape.c.o.requires

.PHONY : CMakeFiles/builder.dir/requires

CMakeFiles/builder.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/builder.dir/cmake_clean.cmake
.PHONY : CMakeFiles/builder.dir/clean

CMakeFiles/builder.dir/depend:
	cd /home/mclovin/git/amsks.github.io/builder && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mclovin/git/amsks.github.io/builder /home/mclovin/git/amsks.github.io/builder /home/mclovin/git/amsks.github.io/builder /home/mclovin/git/amsks.github.io/builder /home/mclovin/git/amsks.github.io/builder/CMakeFiles/builder.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/builder.dir/depend

