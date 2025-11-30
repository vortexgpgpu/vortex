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

`ifndef VORTEX_AFU_VH
`define VORTEX_AFU_VH

`define AFU_ACCEL_NAME "vortex_afu"
`define AFU_ACCEL_UUID 128'h35F9452B_25C2_434C_93D5_6F8C60DB361C

`define AFU_IMAGE_CMD_MEM_READ  1
`define AFU_IMAGE_CMD_MEM_WRITE 2
`define AFU_IMAGE_CMD_RUN       3
`define AFU_IMAGE_CMD_DCR_WRITE 4
`define AFU_IMAGE_CMD_MAX_VALUE 4

`define AFU_IMAGE_MMIO_CMD_TYPE 10
`define AFU_IMAGE_MMIO_CMD_ARG0 12
`define AFU_IMAGE_MMIO_CMD_ARG1 14
`define AFU_IMAGE_MMIO_CMD_ARG2 16
`define AFU_IMAGE_MMIO_STATUS 18
`define AFU_IMAGE_MMIO_SCOPE_READ 20
`define AFU_IMAGE_MMIO_SCOPE_WRITE 22
`define AFU_IMAGE_MMIO_DEV_CAPS 24
`define AFU_IMAGE_MMIO_ISA_CAPS 26

`define AFU_IMAGE_POWER 0
`define AFU_TOP_IFC "ccip_std_afu_avalon_mm"

`endif // VORTEX_AFU_VH
