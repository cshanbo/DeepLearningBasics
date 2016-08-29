# Move this file to upper dir (root dir of this project and make)

CXX=g++
CXXFLAGS = -g -std=c++11 -Wall -pedantic -I.
INC_DIR = include
SRC_DIR = CNN
OBJ_DIR = obj
BIN_DIR = bin
UTIL_DIR = utils

$(BIN_DIR)/test: | $(OBJ_DIR) $(BIN_DIR) $(OBJ_DIR)/CNN.o $(OBJ_DIR)/convolutionLayer.o $(OBJ_DIR)/HiddenLayer.o $(OBJ_DIR)/LR.o $(OBJ_DIR)/utils.o
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/test $(OBJ_DIR)/CNN.o $(OBJ_DIR)/convolutionLayer.o $(OBJ_DIR)/HiddenLayer.o $(OBJ_DIR)/LR.o $(OBJ_DIR)/utils.o

$(OBJ_DIR): 
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR)/CNN.o: $(SRC_DIR)/CNN.cpp $(INC_DIR)/CNN.h $(INC_DIR)/convolutionLayer.h $(INC_DIR)/HiddenLayer.h $(INC_DIR)/LR.h $(INC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/CNN.cpp -o $(OBJ_DIR)/CNN.o

$(OBJ_DIR)/convolutionLayer.o: $(SRC_DIR)/convolutionLayer.cpp $(INC_DIR)/convolutionLayer.h
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/convolutionLayer.cpp -o $(OBJ_DIR)/convolutionLayer.o

$(OBJ_DIR)/HiddenLayer.o: HiddenLayer/HiddenLayer.cpp $(INC_DIR)/HiddenLayer.h
	$(CXX) $(CXXFLAGS) -c HiddenLayer/HiddenLayer.cpp -o $(OBJ_DIR)/HiddenLayer.o

$(OBJ_DIR)/LR.o: LR/LR.cpp $(INC_DIR)/LR.h
	$(CXX) $(CXXFLAGS) -c LR/LR.cpp -o $(OBJ_DIR)/LR.o

$(OBJ_DIR)/utils.o: $(UTIL_DIR)/utils.cpp $(INC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -c $(UTIL_DIR)/utils.cpp -o $(OBJ_DIR)/utils.o

.PHONY: clean
clean:
	rm $(OBJ_DIR)/*
	rm $(BIN_DIR)/*
