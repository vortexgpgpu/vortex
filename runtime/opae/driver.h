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

#include <fpga.h>

typedef fpga_result (*pfn_fpgaGetProperties)(fpga_token token, fpga_properties *prop);
typedef fpga_result (*pfn_fpgaPropertiesSetObjectType)(fpga_properties prop, fpga_objtype objtype);
typedef fpga_result (*pfn_fpgaPropertiesSetGUID)(fpga_properties prop, fpga_guid guid);
typedef fpga_result (*pfn_fpgaDestroyProperties)(fpga_properties *prop);
typedef fpga_result (*pfn_fpgaEnumerate)(const fpga_properties *filters, uint32_t num_filters, fpga_token *tokens, uint32_t max_tokens, uint32_t *num_matches);
typedef fpga_result (*pfn_fpgaDestroyToken)(fpga_token *token);
typedef fpga_result (*pfn_fpgaPropertiesGetLocalMemorySize)(fpga_properties prop, uint64_t *lms);

typedef fpga_result (*pfn_fpgaOpen)(fpga_token token, fpga_handle *handle, int flags);
typedef fpga_result (*pfn_fpgaClose)(fpga_handle handle);
typedef fpga_result (*pfn_fpgaPrepareBuffer)(fpga_handle handle, uint64_t len, void **buf_addr, uint64_t *wsid, int flags);
typedef fpga_result (*pfn_fpgaReleaseBuffer)(fpga_handle handle, uint64_t wsid);
typedef fpga_result (*pfn_fpgaGetIOAddress)(fpga_handle handle, uint64_t wsid, uint64_t *ioaddr);
typedef fpga_result (*pfn_fpgaWriteMMIO64)(fpga_handle handle, uint32_t mmio_num, uint64_t offset, uint64_t value);
typedef fpga_result (*pfn_fpgaReadMMIO64)(fpga_handle handle, uint32_t mmio_num, uint64_t offset, uint64_t *value);
typedef const char *(*pfn_fpgaErrStr)(fpga_result e);

struct opae_drv_api_t {
	pfn_fpgaGetProperties fpgaGetProperties;
	pfn_fpgaPropertiesSetObjectType fpgaPropertiesSetObjectType;
	pfn_fpgaPropertiesSetGUID fpgaPropertiesSetGUID;
	pfn_fpgaDestroyProperties fpgaDestroyProperties;
	pfn_fpgaEnumerate 		fpgaEnumerate;
	pfn_fpgaDestroyToken 	fpgaDestroyToken;
	pfn_fpgaPropertiesGetLocalMemorySize fpgaPropertiesGetLocalMemorySize;

	pfn_fpgaOpen 					fpgaOpen;
	pfn_fpgaClose 				fpgaClose;
	pfn_fpgaPrepareBuffer fpgaPrepareBuffer;
	pfn_fpgaReleaseBuffer fpgaReleaseBuffer;
	pfn_fpgaGetIOAddress 	fpgaGetIOAddress;
	pfn_fpgaWriteMMIO64  	fpgaWriteMMIO64;
	pfn_fpgaReadMMIO64    fpgaReadMMIO64;
	pfn_fpgaErrStr     		fpgaErrStr;
};

int drv_init(opae_drv_api_t* opae_drv_funcs);

void drv_close();
