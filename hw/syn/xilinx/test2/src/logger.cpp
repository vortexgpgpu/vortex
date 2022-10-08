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
#include "logger.h"
#include <algorithm>
#include <fstream>
#include <functional>
#include <stdarg.h>
#include <time.h>
#ifdef WINDOWS
#include <direct.h>
#else
#include <unistd.h>
#endif

using namespace std;

namespace sda {

///////////////////////////////////////////////////////////////////////
string GetApplicationPath() {
#ifdef WINDOWS
#define GetCurrentDir _getcwd
#else
#define GetCurrentDir getcwd
#endif

    char strCurrentPath[FILENAME_MAX];

    if (!GetCurrentDir(strCurrentPath, sizeof(strCurrentPath))) {
        return string("");
    }

    /* not really required */
    strCurrentPath[sizeof(strCurrentPath) - 1] = '\0';
    return string(strCurrentPath);
}

string ToLower(const string& s) {
    string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

string ToUpper(const string& s) {
    string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

string GetTimeStamp() {
    return "";
}

// trim from start
string& ltrim(std::string& s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

// trim from end
string& rtrim(std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

// trim from both ends
string& trim(std::string& s) {
    return ltrim(rtrim(s));
}

string GetFileExt(const string& s) {
    string strext = s.substr(s.find_last_of(".") + 1);
    return strext;
}

string GetFileTitleOnly(const string& s) {
    string temp = s;
    string::size_type d = temp.find_last_of("//");
    if (d == string::npos) d = temp.find_last_of("\\");
    if (d != string::npos) temp = temp.substr(d + 1);

    d = temp.find_last_of(".");
    if (d != string::npos) temp = temp.substr(0, d);

    return temp;
}

void LogWrapper(int etype, const char* file, int line, const char* desc, ...) {
    // crop file name from full path
    string strFileLoc(file);
    strFileLoc = strFileLoc.substr(strFileLoc.find_last_of("\\/") + 1);

    string strHeader = "";
    {
        char header[512];
        // source
        switch (etype) {
            case (sda::etError): {
                snprintf(header, sizeof(header), "ERROR: [%s:%d]", strFileLoc.c_str(), line);
                break;
            }
            case (sda::etInfo): {
                snprintf(header, sizeof(header), "INFO: [%s:%d]", strFileLoc.c_str(), line);
                break;
            }
            case (sda::etWarning): {
                snprintf(header, sizeof(header), "WARN: [%s:%d]", strFileLoc.c_str(), line);
                break;
            }
        }
        strHeader = string(header);
    }

    // time
    string strTime = "";
#ifdef ENABLE_LOG_TIME
    {
        time_t rawtime;
        time(&rawtime);
#ifdef ENABLE_SECURE_API
        char buffer[64];
        struct tm timeinfo;
        localtime_s(&timeinfo, &rawtime);
        asctime_s(timeinfo, buffer, sizeof(buffer)) snprintf(buffer, sizeof(buffer), "TIME: [%s]", asctime(timeinfo));
        strTime = string(buffer);
#else
        char buffer[64];
        struct tm* timeinfo = localtime(&rawtime);
        string temp = string(asctime(timeinfo));
        temp = trim(temp);

        //        strftime(buffer, sizeof(buffer), "TIME: []")
        snprintf(buffer, sizeof(buffer), "TIME: [%s]", temp.c_str());
        strTime = string(buffer);
#endif
    }
#endif

    // format the message itself
    string strMsg = "";
    {
        char msg[512];
        va_list args;
        va_start(args, desc);
        vsnprintf(msg, sizeof(msg), desc, args);
        va_end(args);
        strMsg = string(msg);
    }

    // combine
    string strOut = strHeader + string(" ") + strTime + string(" ") + strMsg + string("\n");

    // display
    cout << strOut;

// store
#ifdef ENABLE_LOG_TOFILE
    std::ofstream outfile;
    outfile.open("benchapp.log", std::ios_base::app);
    outfile << strOut;
#endif

    return;
}

} // namespace sda
