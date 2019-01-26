#include <iostream>
#include <fstream>

typedef float (*boundary)(float, float);

__global__ 
void jacobiIteration(int dimX, int dimY, float * in, float * out);

__host__
void copyToDevice(float * values, const int dimensions[2], float ** in, float ** out);

__host__
void setBoundaryValues(float * values, const int dimensions[2], const float lowerLeft[2], const float upperRight[2], boundary f);

__host__
float * makeInitialValues(const int dimensions[2], const float lowerLeft[2], const float upperRight[2], boundary f);

__host__
float * makeTrueValues(const int dimensions[2], const float lowerLeft[2], const float upperRight[2], boundary f);

__host__
float * getErrors(const float * values, const float * trueValues, const int dimensions[2]);

__host__
float * getRelativeErrors(const float * errors, const float * trueValues, const int dimensions[2], float cutOff = 0.0001);

__host__
float getAverageError(const float * values, const float * trueValues, const int dimX, const int dimY );

__host__ 
void printValues(const int dimensions[2], const float * values);

__host__
void saveToFile(const float * value, const int dimensions[2], const float lowerLeft[2], const float upperRight[2],
                const char * filename);
