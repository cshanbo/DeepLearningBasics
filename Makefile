# Move this file to upper dir (root dir of this project and make)

CXX=g++
CXXFLAGS = -g -std=c++11 -Wall -pedantic -I.
INC_DIR = include
SRC_DIR = CNN
OBJ_DIR = obj
BIN_DIR = bin
UTIL_DIR = utils

$(BIN_DIR)/test: $(OBJ_DIR)/convolutionLayer.o $(OBJ_DIR)/utils.o
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/test $(OBJ_DIR)/convolutionLayer.o $(OBJ_DIR)/utils.o

$(OBJ_DIR)/convolutionLayer.o: $(SRC_DIR)/convolutionLayer.cpp $(INC_DIR)/convolutionLayer.h
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/convolutionLayer.cpp -o $(OBJ_DIR)/convolutionLayer.o

$(OBJ_DIR)/utils.o: $(UTIL_DIR)/utils.cpp $(INC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -c $(UTIL_DIR)/utils.cpp -o $(OBJ_DIR)/utils.o

clean:
	rm $(OBJ_DIR)/*
	rm $(BIN_DIR)/*
