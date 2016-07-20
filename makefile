CXX=g++
CXXFLAGS = -g -std=c++11 -Wall -pedantic -I.
INC_DIR = include
SRC_DIR = MLP
OBJ_DIR = obj

test: $(OBJ_DIR)/HiddenLayer.o
	$(CXX) $(CXXFLAGS) -o test $(OBJ_DIR)/HiddenLayer.o
$(OBJ_DIR)/HiddenLayer.o: $(SRC_DIR)/HiddenLayer.cpp $(INC_DIR)/HiddenLayer.h $(INC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/HiddenLayer.cpp -o $(OBJ_DIR)/HiddenLayer.o
clean:
	rm -rf *.o
	rm test
