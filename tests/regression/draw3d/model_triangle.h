#pragma once

#include "model.h"

const model_t model_triangle = {
    {
        {-0.5f,-0.5f, 0.0f, 1.0, 0xff0000ff, 0.000000, 0.000000},
        { 0.5f,-0.5f, 0.0f, 1.0, 0xff00ff00, 0.000000, 0.000000},
        { 0.0f, 0.5f, 0.0f, 1.0, 0xffff0000, 0.000000, 0.000000}
    }, {
        {0, 1, 2},
    },
    ""
};