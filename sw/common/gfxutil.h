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

#include <cocogfx/include/cgltrace.hpp>
#include <cocogfx/include/format.hpp>

namespace graphics {

uint32_t toVXFormat(cocogfx::ePixelFormat format);

uint32_t toVXCompare(cocogfx::CGLTrace::ecompare compare);

uint32_t toVXStencilOp(cocogfx::CGLTrace::eStencilOp op);

uint32_t toVXBlendFunc(cocogfx::CGLTrace::eBlendOp op);

uint32_t Binning(std::vector<uint8_t>& tilebuf,
                 std::vector<uint8_t>& primbuf,
                 const std::unordered_map<uint32_t, cocogfx::CGLTrace::vertex_t>& vertices,
                 const std::vector<cocogfx::CGLTrace::primitive_t>& primitives,
                 uint32_t width,
                 uint32_t height,
                 float near,
                 float far,
                 uint32_t tileLogSize);

std::string ResolveFilePath(const std::string& filename, const std::string& searchPaths);

} // namespace graphics