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



#include "Vrf2_32x128_wm1_rtl.h"
#include "verilated.h"

int main()
{
	Vrf2_32x128_wm1_rtl module;

	for (int i = 0; i < 10; i++)
	{
		// module.clk = 0;
    	module.eval();
        // module.clk = 1;
        module.eval();
	}

	return 0;
}