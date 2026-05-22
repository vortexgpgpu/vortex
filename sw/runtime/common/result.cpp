// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <vortex2.h>

extern "C" const char* vx_result_string(vx_result_t r) {
    switch (r) {
    case VX_SUCCESS:                  return "VX_SUCCESS";
    case VX_ERR_INVALID_HANDLE:       return "VX_ERR_INVALID_HANDLE";
    case VX_ERR_INVALID_INFO:         return "VX_ERR_INVALID_INFO";
    case VX_ERR_INVALID_VALUE:        return "VX_ERR_INVALID_VALUE";
    case VX_ERR_OUT_OF_HOST_MEMORY:   return "VX_ERR_OUT_OF_HOST_MEMORY";
    case VX_ERR_OUT_OF_DEVICE_MEMORY: return "VX_ERR_OUT_OF_DEVICE_MEMORY";
    case VX_ERR_DEVICE_LOST:          return "VX_ERR_DEVICE_LOST";
    case VX_ERR_TIMEOUT:              return "VX_ERR_TIMEOUT";
    case VX_ERR_EVENT_FAILED:         return "VX_ERR_EVENT_FAILED";
    case VX_ERR_NOT_SUPPORTED:        return "VX_ERR_NOT_SUPPORTED";
    case VX_ERR_INTERNAL:             return "VX_ERR_INTERNAL";
    default:                          return "VX_ERR_UNKNOWN";
    }
}
