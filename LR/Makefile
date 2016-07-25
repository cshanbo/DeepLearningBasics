# Move this file to upper dir (root dir of this project and make)

CXX=g++
CXXFLAGS = -g -std=c++11 -Wall -pedantic -I.
INC_DIR = include
SRC_DIR = LR 
OBJ_DIR = obj
BIN_DIR = bin
UTIL_DIR = utils

$(BIN_DIR)/test: $(OBJ_DIR)/LR.o $(OBJ_DIR)/utils.o
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/test $(OBJ_DIR)/LR.o $(OBJ_DIR)/utils.o

$(OBJ_DIR)/LR.o: LR/LR.cpp $(INC_DIR)/LR.h
	$(CXX) $(CXXFLAGS) -c LR/LR.cpp -o $(OBJ_DIR)/LR.o

$(OBJ_DIR)/utils.o: $(UTIL_DIR)/utils.cpp $(INC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -c $(UTIL_DIR)/utils.cpp -o $(OBJ_DIR)/utils.o

clean:
	rm -rf $(OBJ_DIR)/*
	rm $(BIN_DIR)/*
