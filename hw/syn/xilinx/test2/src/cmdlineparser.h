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
#ifndef CMDLINEPARSER_H_
#define CMDLINEPARSER_H_

#include <map>
#include <string>
#include <vector>

namespace sda {
namespace utils {

bool is_file(const std::string& name);

/*!
 * Synopsis:
 * 1.Parses the command line passed in from the user and stores all enabled
 *      system options.
 * 2.Prints help for the user if an option is not valid.
 * 3.Stores options and provides a mechanism to read those options
 */
class CmdLineParser {
   public:
    class CmdSwitch {
       public:
        std::string key;
        std::string shortcut;
        std::string default_value;
        std::string value;
        std::string desc;
        bool istoggle;
        bool isvalid;
    };

   public:
    CmdLineParser();
    // CmdLineParser(int argc, char* argv[]);
    virtual ~CmdLineParser();

    bool addSwitch(const CmdSwitch& s);
    bool addSwitch(const std::string& name,
                   const std::string& shortcut,
                   const std::string& desc,
                   const std::string& default_value = "",
                   bool istoggle = false);

    /*!
     * sets default key to be able to read a 2 argumented call
     */
    bool setDefaultKey(const char* key);

    /*!
     * parse and store command line
     */
    int parse(int argc, char* argv[]);

    /*!
     * retrieve value using a key
     */
    std::string value(const char* key);

    bool value_to_bool(const char* key);

    int value_to_int(const char* key);

    double value_to_double(const char* key);

    /*!
     * Returns true if a valid value is supplied by user
     */
    bool isValid(const char* key);

    /*!
     * prints the help menu in case the options are not correct.
     */
    virtual void printHelp();

   protected:
    /*!
     * Retrieve command switch
     */
    CmdSwitch* getCmdSwitch(const char* key);

    bool token_to_fullkeyname(const std::string& token, std::string& fullkey);

   private:
    std::map<std::string, CmdSwitch*> m_mapKeySwitch;
    std::map<std::string, std::string> m_mapShortcutKeys;
    std::vector<CmdSwitch*> m_vSwitches;
    std::string m_strDefaultKey;
    std::string m_appname;
};

// bool starts_with(const string& src, const string& sub);
}
}
#endif /* CMDLINEPARSER_H_ */
