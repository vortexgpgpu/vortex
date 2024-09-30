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

#include "driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <linux/limits.h>
#include <dlfcn.h>
#include <string>
#include <vector>
#include <sstream>

#define DEFAULT_OPAE_DRV_PATHS "libopae-c.so"

#define SET_API(func) \
	opae_drv_funcs->func = (pfn_##func)dlsym(dl_handle, #func); \
	if (opae_drv_funcs->func == nullptr) { \
    printf("dlsym failed: %s\n", dlerror()); \
		dlclose(dl_handle); \
    return -1; \
	}

void* dl_handle = nullptr;

int drv_init(opae_drv_api_t* opae_drv_funcs) {
  if (opae_drv_funcs == nullptr)
    return -1;

  const char* api_path_s = getenv("OPAE_DRV_PATHS");
  if (api_path_s == nullptr || api_path_s[0] == '\0') {
    api_path_s = DEFAULT_OPAE_DRV_PATHS;
  }

  std::vector<std::string> api_paths;
  {
    std::stringstream ss(api_path_s);
    while (ss.good()) {
      std::string path;
      getline(ss, path, ',');
      api_paths.push_back(path);
    }
  }

  for (auto& api_path : api_paths) {
    dl_handle = dlopen(api_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (dl_handle)
      break;
  }

  if (dl_handle == nullptr) {
      printf("dlopen failed: %s\n", dlerror());
      return -1;
  }

	SET_API (fpgaGetProperties);
	SET_API (fpgaPropertiesSetObjectType);
	SET_API (fpgaPropertiesSetGUID);
	SET_API (fpgaDestroyProperties);
  SET_API (fpgaDestroyToken);
  SET_API (fpgaPropertiesGetLocalMemorySize);
	SET_API (fpgaEnumerate);
	SET_API (fpgaOpen);
	SET_API (fpgaClose);
	SET_API (fpgaPrepareBuffer);
	SET_API (fpgaReleaseBuffer);
	SET_API (fpgaGetIOAddress);
	SET_API (fpgaWriteMMIO64);
	SET_API (fpgaReadMMIO64);
	SET_API (fpgaErrStr);

  return 0;
}

void drv_close() {
    dlclose(dl_handle);
}
