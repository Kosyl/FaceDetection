#ifndef _TYPEDEF_H_
#define _TYPEDEF_H_

#define BASE_CLASSIFIER_SIZE 20
#define SCALE_INCREASE 1.2f
#define STAGES_COUNT 22
#define WINDOW_STEP_SHIFT 3
#define WINDOW_STEP_IN_PIX (BASE_CLASSIFIER_SIZE>>WINDOW_STEP_SHIFT)

#define NUM_PHASES_PER_CUDA_RUN 220
#define TILE_SIZE 16
#define INV_AREA 0.0025f // 1/BASE_CLASSIFIER_SIZE/BASE_CLASSIFIER_SIZE

#define TILE_DISTANCE_IN_PIX (TILE_SIZE*(WINDOW_STEP_IN_PIX))
#define TILE_SIZE_IN_PIX (TILE_DISTANCE_IN_PIX+BASE_CLASSIFIER_SIZE+1)

typedef unsigned int UInt;
#include "cuda_runtime.h"
#include "cuda.h"

//#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

#endif