#
# Move this file to the root directory of this project, then make
#

CXX=g++
CXXFLAGS = -g -std=c++11 -Wall -pedantic -I.
INC_DIR = include
SRC_DIR = MLP
OBJ_DIR = obj
BIN_DIR = bin
UTIL_DIR = utils

$(BIN_DIR)/test: |$(OBJ_DIR) $(BIN_DIR) $(OBJ_DIR)/MLP.o $(OBJ_DIR)/HiddenLayer.o $(OBJ_DIR)/utils.o $(OBJ_DIR)/LR.o
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/test $(OBJ_DIR)/MLP.o $(OBJ_DIR)/HiddenLayer.o $(OBJ_DIR)/utils.o $(OBJ_DIR)/LR.o

$(OBJ_DIR): 
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR)/utils.o: $(UTIL_DIR)/utils.cpp $(INC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -c $(UTIL_DIR)/utils.cpp -o $(OBJ_DIR)/utils.o

$(OBJ_DIR)/MLP.o: $(SRC_DIR)/MLP.cpp $(INC_DIR)/MLP.h $(INC_DIR)/HiddenLayer.h $(INC_DIR)/LR.h\
				$(OBJ_DIR)/HiddenLayer.o $(OBJ_DIR)/LR.o
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/MLP.cpp -o $(OBJ_DIR)/MLP.o

$(OBJ_DIR)/HiddenLayer.o: HiddenLayer/HiddenLayer.cpp $(INC_DIR)/HiddenLayer.h $(INC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -c HiddenLayer/HiddenLayer.cpp -o $(OBJ_DIR)/HiddenLayer.o

$(OBJ_DIR)/LR.o: LR/LR.cpp $(INC_DIR)/LR.h
	$(CXX) $(CXXFLAGS) -c LR/LR.cpp -o $(OBJ_DIR)/LR.o

clean:
	rm $(OBJ_DIR)/*
	rm $(BIN_DIR)/*
