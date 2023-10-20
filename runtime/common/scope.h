// Copyright © 2019-2023
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <vortex.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*pfn_registerWrite)(vx_device_h hdevice, uint64_t value);
typedef int (*pfn_registerRead)(vx_device_h hdevice, uint64_t *value);

struct scope_callback_t {
	pfn_registerWrite registerWrite;
	pfn_registerRead  registerRead;
};

int vx_scope_start(scope_callback_t* callback, vx_device_h hdevice, uint64_t start_time, uint64_t stop_time);
int vx_scope_stop(vx_device_h hdevice);

#ifdef __cplusplus
}
#endif
