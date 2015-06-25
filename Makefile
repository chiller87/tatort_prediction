# the compiler to use.
CC=g++
# options to pass to the compiler.
CFLAGS=-std=c++11 -c -Wall -O3 -funroll-loops -funroll-all-loops
# options to pass to the linker
LDFLAGS=
# the soure files
SOURCES=$(wildcard *.cpp) $(wildcard *.h)
# the object files
OBJECTS=$(SOURCES:.cpp=.o)
# name of the executable
EXECUTABLE=test_linux

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *o $(EXECUTABLE)


# Makefile explanations:
# ========================================================
# rule form				->	target : dependencies
#								command to create target
# $(wildcard *.cpp)		->	completes to all .cpp files in current directory
# $(SOURCES:.cpp=.o)	->	describes all names in SOURCES, but replace every appearance of '.cpp' with '.o'
# $@					->	full target name
# $*					->	target name without extension
# %						->	wildcard for any character combination
# $<					->	filename, that fits wildcard
# $(VAR)				->	content of variable VAR

