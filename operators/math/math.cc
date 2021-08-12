#include "ocos.h"
#include "negpos.hpp"
#include "inverse.hpp"
#include "multithreshold.hpp"

template const OrtCustomOp** LoadCustomOpClasses<CustomOpNegPos, CustomOpInverse, MultithresholdOp>();

FxLoadCustomOpFactory LoadCustomOpClasses_Math = &LoadCustomOpClasses<CustomOpNegPos, CustomOpInverse, MultithresholdOp>;
