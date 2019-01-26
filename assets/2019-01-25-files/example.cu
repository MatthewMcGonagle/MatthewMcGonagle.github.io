#include <iostream>
#include "jacobi.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <string>

typedef float (*boundary)(float, float);

__host__
float boundaryValues(float x, float y) 
{
    return pow(x - 0.5, 3) - 2 * (x-0.5) * pow(y - 0.5, 2); 
}

int main(int argc, char * argv[]) 
{
    // First get the dimensions from command line arguments.

    int N;
    if(argc < 2) // default vale for no parameters.
        N = 20;
    else { 
        N = std::stoi(argv[1]);
        if( N < 2) // Use default for not good values of N.
            N = 20;
    }

    int nIterations = 3 * N * N,
        dimensions[2] = {N, N},
        nThreads = N / 10 + 1,
        memSize = dimensions[0] * dimensions[1] * sizeof(float);
    const float lowerLeft[2] = {0, 0},
                upperRight[2] = {1, 1};
    float * values, * trueValues, * in, * out, * errors, * relErrors;
    const dim3 blockSize( nThreads , nThreads),
               gridSize( (dimensions[0] + nThreads - 1) / nThreads, (dimensions[1] + nThreads - 1) / nThreads);

    std::cout << "Making initial values and true values" << std::endl;
    values = makeInitialValues( dimensions, lowerLeft, upperRight, & boundaryValues ); 
    trueValues = makeTrueValues( dimensions, lowerLeft, upperRight, & boundaryValues );

    std::cout << "Before Average Error = " << getAverageError(values, trueValues, dimensions[0], dimensions[1]) << std::endl;

    std::cout << "Copying to Device" << std::endl;
    try 
    {
        copyToDevice(values, dimensions, &in, &out);
    }
    catch( ... )
    {
        std::cout << "Exception happened while copying to device" << std::endl;
    }

    // At end of loop, output is inside in pointer.

    std::cout << "Doing Jacobi Iterations" << std::endl;
    for( int i = 0; i < nIterations; i++)
    {
        jacobiIteration<<< gridSize, blockSize >>>(dimensions[0], dimensions[1], in, out);
        cudaDeviceSynchronize();
        if(cudaGetLastError() != cudaSuccess)
        {
            std::cout << "Error Launching Kernel" << std::endl;
            return 1;
        }
        std::swap(in, out);
    }

    std::cout << "Copying result to values" << std::endl;
    if(cudaMemcpy( values, in, memSize, cudaMemcpyDeviceToHost ) != cudaSuccess) 
    {
        std::cout << "There was a problem retrieving the result from the device" << std::endl;
        return 1;    
    }

    std::cout << "Copying to file 'values.dat'" << std::endl;
    saveToFile( values, dimensions, lowerLeft, upperRight, "data/values.dat");

    std::cout << "Now getting errors" << std::endl;
    errors = getErrors(values, trueValues, dimensions);
    saveToFile( errors, dimensions, lowerLeft, upperRight, "data/errors.dat");
    std::cout << "After Average Error = " << getAverageError(values, trueValues, dimensions[0], dimensions[1]) << std::endl;


    std::cout << "Now getting relative errors" << std::endl;
    relErrors = getRelativeErrors(errors, trueValues, dimensions);
    saveToFile( relErrors, dimensions, lowerLeft, upperRight, "data/log10RelErrors.dat");
    // Clean up memory.

    cudaFree(in);
    cudaFree(out);
    delete values;
    delete errors;
    delete relErrors;
    delete trueValues;

    return 0;
}
