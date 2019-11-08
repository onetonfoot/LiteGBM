# Automatically generated using Clang.jl


const C_API_DTYPE_FLOAT32 = 0
const C_API_DTYPE_FLOAT64 = 1
const C_API_DTYPE_INT32 = 2
const C_API_DTYPE_INT64 = 3
const C_API_DTYPE_INT8 = 4
const C_API_PREDICT_NORMAL = 0
const C_API_PREDICT_RAW_SCORE = 1
const C_API_PREDICT_LEAF_INDEX = 2
const C_API_PREDICT_CONTRIB = 3
# const THREAD_LOCAL = thread_local # What is this for?
const DatasetHandle = Ptr{Cvoid}
const BoosterHandle = Ptr{Cvoid}
