#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <linux/limits.h>
#include <dlfcn.h>
#include <string>
#include <vector>
#include<sstream>

#define DEFAULT_OPAE_API_PATHS "/usr/lib64/opae/libope-c.so,/usr/lib/opae/libope-c.so,libope-c.so"

#define SET_API(func) \
	opae_api_funcs->func = (pfn_##func)dlsym(dl_handle, #func); \
	if (opae_api_funcs->func == nullptr) { \
        printf("dlsym failed: %s\n", dlerror()); \
		dlclose(dl_handle); \
        return -1; \
	}

int api_init(opae_api_funcs_t* opae_api_funcs) {
    if (opae_api_funcs == nullptr)
        return -1;

    const char* api_path_s = getenv("OPAE_API_PATHS");
    if (api_path_s == nullptr || api_path_s[0] == '\0') {
        api_path_s = DEFAULT_OPAE_API_PATHS;
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

    void* dl_handle = nullptr;
    for (auto& api_path : api_paths) {
		dl_handle = dlopen(api_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
		if (dl_handle)
			break;
	}
    if (dl_handle == nullptr) {
        printf("dlopen failed: %s\n", dlerror());
        return -1;
    }

	SET_API(fpgaGetProperties);
	SET_API(fpgaPropertiesSetObjectType);
	SET_API(fpgaPropertiesSetGUID);
	SET_API(fpgaDestroyProperties);
	SET_API(fpgaEnumerate);
	SET_API(fpgaDestroyToken);
	SET_API(fpgaOpen);
	SET_API(fpgaClose);
	SET_API(fpgaPrepareBuffer);
	SET_API(fpgaReleaseBuffer);
	SET_API(fpgaGetIOAddress);
	SET_API(fpgaWriteMMIO64);
	SET_API(fpgaReadMMIO64);
	SET_API(fpgaErrStr);

    dlclose(dl_handle);

    return 0;
}