CXX=g++
CXXFLAGS = -g -std=c++11 -Wall -pedantic -I.
INC_DIR = include
SRC_DIR = MLP
OBJ_DIR = obj
BIN_DIR = bin
UTIL_DIR = utils

$(BIN_DIR)/test: $(OBJ_DIR)/HiddenLayer.o $(OBJ_DIR)/utils.o
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/test $(OBJ_DIR)/HiddenLayer.o $(OBJ_DIR)/utils.o
$(OBJ_DIR)/HiddenLayer.o: $(SRC_DIR)/HiddenLayer.cpp $(INC_DIR)/HiddenLayer.h $(INC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/HiddenLayer.cpp -o $(OBJ_DIR)/HiddenLayer.o
$(OBJ_DIR)/utils.o: $(UTIL_DIR)/utils.cpp $(INC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -c $(UTIL_DIR)/utils.cpp -o $(OBJ_DIR)/utils.o
clean:
	rm -rf $(OBJ_DIR)/*
	rm $(BIN_DIR)/*
