#pragma once

#include <cstdint>
#include <algorithm>
#include <assert.h>
#include <bitmanip.h>

template <typename... Args>
void unused(Args&&...) {}

#define __unused(...) unused(__VA_ARGS__)

// return file extension
const char* fileExtension(const char* filepath);