# Move this file to upper dir (root dir of this project and make)

CXX=g++
CXXFLAGS = -g -std=c++11 -Wall -pedantic -I.
INC_DIR = include
SRC_DIR = RBM
OBJ_DIR = obj
BIN_DIR = bin
UTIL_DIR = utils

$(BIN_DIR)/test: $(OBJ_DIR)/RBM.o $(OBJ_DIR)/utils.o
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/test $(OBJ_DIR)/RBM.o $(OBJ_DIR)/utils.o

$(OBJ_DIR)/RBM.o: $(SRC_DIR)/RBM.cpp $(INC_DIR)/RBM.h
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/RBM.cpp -o $(OBJ_DIR)/RBM.o

$(OBJ_DIR)/utils.o: $(UTIL_DIR)/utils.cpp $(INC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -c $(UTIL_DIR)/utils.cpp -o $(OBJ_DIR)/utils.o

clean:
	rm $(OBJ_DIR)/*
	rm $(BIN_DIR)/*
