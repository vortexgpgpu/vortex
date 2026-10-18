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

// The kernel's memory ports carry their index in their names, so they are
// listed by FOLDSIM_REPEAT, as VX_platform.vh's MP_REPEAT lists the native
// shells' ports.

`ifndef FOLDSIM_PORTS_VH
`define FOLDSIM_PORTS_VH

`define FOLDSIM_REPEAT(n,f,s)  `_FOLDSIM_REPEAT_``n(f,s)
`define _FOLDSIM_REPEAT_1(f,s) `f(0)
`define _FOLDSIM_REPEAT_2(f,s) `_FOLDSIM_REPEAT_1(f,s) `s `f(1)
`define _FOLDSIM_REPEAT_3(f,s) `_FOLDSIM_REPEAT_2(f,s) `s `f(2)
`define _FOLDSIM_REPEAT_4(f,s) `_FOLDSIM_REPEAT_3(f,s) `s `f(3)
`define _FOLDSIM_REPEAT_5(f,s) `_FOLDSIM_REPEAT_4(f,s) `s `f(4)
`define _FOLDSIM_REPEAT_6(f,s) `_FOLDSIM_REPEAT_5(f,s) `s `f(5)
`define _FOLDSIM_REPEAT_7(f,s) `_FOLDSIM_REPEAT_6(f,s) `s `f(6)
`define _FOLDSIM_REPEAT_8(f,s) `_FOLDSIM_REPEAT_7(f,s) `s `f(7)

`define FOLDSIM_COMMA ,
`define FOLDSIM_SEMI  ;

`endif // FOLDSIM_PORTS_VH
