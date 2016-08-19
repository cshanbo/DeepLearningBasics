# Move this file to upper dir (root dir of this project and make)

CXX=g++
CXXFLAGS = -g -std=c++11 -Wall -pedantic -I.
INC_DIR = include
SRC_DIR = RNN
OBJ_DIR = obj
BIN_DIR = bin
UTIL_DIR = utils

$(BIN_DIR)/test: $(OBJ_DIR)/RNN.o $(OBJ_DIR)/utils.o
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/test $(OBJ_DIR)/RNN.o $(OBJ_DIR)/utils.o

$(OBJ_DIR)/RNN.o: $(SRC_DIR)/RNN.cpp $(INC_DIR)/RNN.h $(INC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/RNN.cpp -o $(OBJ_DIR)/RNN.o

$(OBJ_DIR)/utils.o: $(UTIL_DIR)/utils.cpp $(INC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -c $(UTIL_DIR)/utils.cpp -o $(OBJ_DIR)/utils.o

clean:
	rm $(OBJ_DIR)/*
	rm $(BIN_DIR)/*
