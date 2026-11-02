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

// The folded memory ports' records (st32_1031/lk2/img4dbg/PORTFOLD_IFACE.md):
// one REQ record (the target's 14 output fields of a port's target cycle) and
// one RSP record (the endpoint's 11 output fields), packed low field first in
// the order below. Used inside a module with ID_W, ADDR_W, DATA_W, NC and NW
// parameters.
`ifndef FOLDSIM_MEM_REC_VH
`define FOLDSIM_MEM_REC_VH

// REQ: awvalid, awaddr, awid, awlen, wvalid, wdata, wstrb, wlast, arvalid,
// araddr, arid, arlen, rready, bready.
`define FMR_REQ_FIELDS 14
`define FMR_REQ_W(i) ( \
    (i) == 0  ? 1 : (i) == 1  ? ADDR_W : (i) == 2  ? ID_W   : (i) == 3  ? 8 : \
    (i) == 4  ? 1 : (i) == 5  ? DATA_W : (i) == 6  ? DATA_W/8 : (i) == 7 ? 1 : \
    (i) == 8  ? 1 : (i) == 9  ? ADDR_W : (i) == 10 ? ID_W   : (i) == 11 ? 8 : \
    (i) == 12 ? 1 : 1)
// RSP: awready, wready, arready, rvalid, rdata, rlast, rid, rresp, bvalid,
// bresp, bid.
`define FMR_RSP_FIELDS 11
`define FMR_RSP_W(i) ( \
    (i) == 0 ? NW : (i) == 1 ? NW : (i) == 2 ? NC : (i) == 3 ? NC : \
    (i) == 4 ? NC*DATA_W : (i) == 5 ? NC : (i) == 6 ? NC*ID_W : (i) == 7 ? NC*2 : \
    (i) == 8 ? 1 : (i) == 9 ? 2 : ID_W)

`endif
