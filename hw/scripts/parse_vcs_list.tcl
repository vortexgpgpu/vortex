# Copyright © 2019-2023
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

proc parse_vcs_list {flist_path} {
    set f [split [string trim [read [open $flist_path r]]] "\n"]
    set flist [list ]
    set dir_list [list ]
    set def_list [list ]
    foreach x $f {
        if {![string match "" $x]} {
            # If the item starts with +incdir+, directory files need to be added
            if {[string match "#*" $x]} {
                # get rid of comment line
            } elseif {[string match "+incdir+*" $x]} {
                set trimchars "+incdir+"
                set temp [string trimleft $x $trimchars]
                set expanded [subst $temp]
                lappend dir_list $expanded
            } elseif {[string match "+define+*" $x]} {
                set trimchars "+define+"
                set temp [string trimleft $x $trimchars]
                set expanded [subst $temp]
                lappend def_list $expanded
            } else {
                set expanded [subst $x]
                lappend flist $expanded
            }
        }
    }

    #puts $flist
    #puts $dir_list
    #puts $def_list

    return [list $flist $dir_list $def_list]
}
