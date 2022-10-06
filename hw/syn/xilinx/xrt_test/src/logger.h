/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/
#ifndef LOGGER_H_
#define LOGGER_H_

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define ENABLE_LOG_TOFILE 1
#define ENABLE_LOG_TIME 1

// global logging
#define LogInfo(desc, ...) sda::LogWrapper(0, __FILE__, __LINE__, desc, ##__VA_ARGS__)
#define LogWarn(desc, ...) sda::LogWrapper(1, __FILE__, __LINE__, desc, ##__VA_ARGS__)
#define LogError(desc, ...) sda::LogWrapper(2, __FILE__, __LINE__, desc, ##__VA_ARGS__)

using namespace std;

namespace sda {

enum LOGTYPE { etInfo, etWarning, etError };

// string
string& ltrim(string& s);
string& rtrim(string& s);
string& trim(string& s);
string GetFileExt(const string& s);
string GetFileTitleOnly(const string& s);

string ToLower(const string& s);
string ToUpper(const string& s);

// time
string GetTimeStamp();

// paths
string GetApplicationPath();

// debug
template <typename T>
void PrintPOD(const vector<T>& pod, size_t display_count = 0, const int precision = 4) {
    size_t count = pod.size();
    if (display_count > 0) count = std::min<size_t>(pod.size(), display_count);

    for (size_t i = 0; i < count; i++) {
        cout << std::setprecision(precision) << pod[i] << ", ";
    }
    cout << endl;
}

// logging
void LogWrapper(int etype, const char* file, int line, const char* desc, ...);
}

#endif /* LOGGER_H_ */
