#ifndef _TYPEDEF_H_
#define _TYPEDEF_H_

#define BASE_CLASSIFIER_SIZE 20
#define STARTING_CLASSIFIER_SIZE 20
#define SCALE_INCREASE 1.2f
#define STAGES_COUNT 22
#define WINDOW_STEP_SHIFT 3

#define NUM_PHASES_PER_CUDA_RUN 200
#define TILE_SIZE 8


typedef unsigned int UInt;
#include "cuda_runtime.h"
#include "cuda.h"

//#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

#endif