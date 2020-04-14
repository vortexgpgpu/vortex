/****************************************************************************\ 
 * Copyright (c) 2011, Advanced Micro Devices, Inc.                           *
 * All rights reserved.                                                       *
 *                                                                            *
 * Redistribution and use in source and binary forms, with or without         *
 * modification, are permitted provided that the following conditions         *
 * are met:                                                                   *
 *                                                                            *
 * Redistributions of source code must retain the above copyright notice,     *
 * this list of conditions and the following disclaimer.                      *
 *                                                                            *
 * Redistributions in binary form must reproduce the above copyright notice,  *
 * this list of conditions and the following disclaimer in the documentation  *
 * and/or other materials provided with the distribution.                     *
 *                                                                            *
 * Neither the name of the copyright holder nor the names of its contributors *
 * may be used to endorse or promote products derived from this software      *
 * without specific prior written permission.                                 *
 *                                                                            *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS        *
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED  *
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR *
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR          *
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,      *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,        *
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR         *
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF     *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING       *
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         *
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.               *
 *                                                                            *
 * If you use the software (in whole or in part), you shall adhere to all     *
 * applicable U.S., European, and other export laws, including but not        *
 * limited to the U.S. Export Administration Regulations (EAR), (15 C.F.R.  *
 * Sections 730 through 774), and E.U. Council Regulation (EC) No 1334/2000   *
 * of 22 June 2000.  Further, pursuant to Section 740.6 of the EAR, you       *
 * hereby certify that, except pursuant to a license granted by the United    *
 * States Department of Commerce Bureau of Industry and Security or as        *
 * otherwise permitted pursuant to a License Exception under the U.S. Export  *
 * Administration Regulations ("EAR"), you will not (1) export, re-export or  *
 * release to a national of a country in Country Groups D:1, E:1 or E:2 any   *
 * restricted technology, software, or source code you receive hereunder,     *
 * or (2) export to Country Groups D:1, E:1 or E:2 the direct product of such *
 * technology or software, if such foreign produced direct product is subject *
 * to national security controls as identified on the Commerce Control List   *
 *(currently found in Supplement 1 to Part 774 of EAR).  For the most current *
 * Country Group listings, or for additional information about the EAR or     *
 * your obligations under those regulations, please refer to the U.S. Bureau  *
 * of Industry and Securitys website at http://www.bis.doc.gov/.             *
 \****************************************************************************/

#include <stdio.h>
#include <sys/stat.h>
#include <string.h>
#include <stdlib.h>

#include "utils.h"

static bool usingImages = true;

//! A wrapper for malloc that checks the return value
void* alloc(size_t size) {

    void* ptr = NULL;
    ptr = malloc(size);
    if(ptr == NULL) {
        perror("malloc");
        exit(-1);
    }

    return ptr;
}

// This function checks to make sure a file exists before we open it
void checkFile(char* filename) 
{
    
    struct stat fileStatus;
    if(stat(filename, &fileStatus) != 0) {
        printf("Error opening file: %s\n", filename);
        exit(-1);
    }
    else {
        if(!(S_IFREG & fileStatus.st_mode)) {
            printf("File %s is not a regular file\n", filename);
            exit(-1);
        }
    }
}


// This function checks to make sure a directory exists 
void checkDir(char* dirpath) 
{
    
    struct stat fileStatus;
    if(stat(dirpath, &fileStatus) != 0) {
        printf("Directory does not exist: %s\n", dirpath);
        exit(-1);
    }
    else {
        if(!(S_IFDIR & fileStatus.st_mode)) {
            printf("Directory was not provided: %s\n", dirpath);
            exit(-1);
        }
    }
}

// Parse the command line arguments
void parseArguments(int argc, char** argv, char** input, char** events, 
    char** ipts, char* devicePref, bool* verifyResults) 
{
    
    for(int i = 2; i < argc; i++) {
        if(strcmp(argv[i], "-d") == 0) {   // Event dump found
            if(i == argc-1) {
                printf("Usage: -e Needs directory path\n");
                exit(-1);
            }
            devicePref[0] = argv[i+1][0];
            i++;
            continue;
        }
        if(strcmp(argv[i], "-e") == 0) {   // Event dump found
            if(i == argc-1) {
                printf("Usage: -e Needs directory path\n");
                exit(-1);
            }
            *events = argv[i+1];
            i++;
            continue;
        }
        if(strcmp(argv[i], "-i") == 0) {   // Input found
            if(i == argc-1) {
                printf("Usage: -i Needs directory path\n");
                exit(-1);
            }
            *input = argv[i+1];
            i++;
            continue;
        }
        if(strcmp(argv[i], "-l") == 0) {   // Ipts dump found
            if(i == argc-1) {
                printf("Usage: -l Needs directory path\n");
                exit(-1);
            }
            *ipts = argv[i+1];
            i++;
            continue;
        }
        if(strcmp(argv[i], "-n") == 0) {   // Don't use OpenCL images
            setUsingImages(false);
            continue;
        }
        if(strcmp(argv[i], "-v") == 0) {   // Verify results
            *verifyResults = true;
            continue;
        }
    }
}


// This function that takes a positive integer 'value' and returns
// the nearest multiple of 'multiple' (used for padding columns)
unsigned int roundUp(unsigned int value, unsigned int multiple) {

   unsigned int remainder = value % multiple;
   
   // Make the value a multiple of multiple
   if(remainder != 0) {
      value += (multiple-remainder);
   }

   return value;
}


// Concatenate two strings and return a pointer to the new string
char* smartStrcat(char* str1, char* str2) 
{
    char* newStr = NULL;

    newStr = (char*)alloc((strlen(str1)+strlen(str2)+1)*sizeof(char));

    strcpy(newStr, str1);
    strcat(newStr, str2);

    return newStr;
}


// Set the value of using images to true if they are being
// used, or false if they are not
void setUsingImages(bool val) 
{
    usingImages = val;
}


// Return whether or not images are being used
bool isUsingImages() 
{
    return usingImages;
}
