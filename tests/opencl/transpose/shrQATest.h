/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#ifndef SHR_QATEST_H
#define SHR_QATEST_H

// *********************************************************************
// Generic utilities for NVIDIA GPU Computing SDK 
// *********************************************************************

// OS dependent includes
#ifdef _WIN32
    #pragma message ("Note: including windows.h")
    #pragma message ("Note: including math.h")
    #pragma message ("Note: including assert.h")
    #pragma message ("Note: including time.h")

// Headers needed for Windows
    #include <windows.h>
	#include <time.h>
#else
    // Headers needed for Linux
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <sys/time.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <stdarg.h>
    #include <unistd.h>
    #include <time.h>
#endif

#ifndef STRCASECMP
#ifdef _WIN32
#define STRCASECMP _stricmp
#else
#define STRCASECMP strcasecmp
#endif
#endif

#ifndef STRNCASECMP
#ifdef _WIN32
#define STRNCASECMP _strnicmp
#else
#define STRNCASECMP strncasecmp
#endif
#endif


// Standardized QA Start/Finish for CUDA SDK tests
#define shrQAStart(a, b)      __shrQAStart(a, b)
#define shrQAFinish(a, b, c)  __shrQAFinish(a, b, c)
#define shrQAFinish2(a, b, c, d) __shrQAFinish2(a, b, c, d)

inline int findExeNameStart(const char *exec_name)
{
    int exename_start = (int)strlen(exec_name);

    while( (exename_start > 0) && 
            (exec_name[exename_start] != '\\') && 
            (exec_name[exename_start] != '/') )
    {
        exename_start--;
    }
    if (exec_name[exename_start] == '\\' || 
        exec_name[exename_start] == '/')
    {
        return exename_start+1;
    } else {
        return exename_start;
    }
}

inline int __shrQAStart(int argc, char **argv)
{
    bool bQATest = false;
    // First clear the output buffer
    fflush(stdout);
    fflush(stdout);

    for (int i=1; i < argc; i++) {
        int string_start = 0;
        while (argv[i][string_start] == '-')
           string_start++;
        char *string_argv = &argv[i][string_start];

        if (!STRCASECMP(string_argv, "qatest")) {
           bQATest = true;
        }
    }
    
    // We don't want to print the entire path, so we search for the first 
    int exename_start = findExeNameStart(argv[0]);
    if (bQATest) {
        fprintf(stdout, "&&&& RUNNING %s", &(argv[0][exename_start]));
        for (int i=1; i < argc; i++) fprintf(stdout, " %s", argv[i]);
        fprintf(stdout, "\n");
    } else {
        fprintf(stdout, "[%s] starting...\n", &(argv[0][exename_start]));
    }
    fflush(stdout);
    printf("\n"); fflush(stdout);
    return exename_start;
}

enum eQAstatus {
    QA_FAILED = 0,
    QA_PASSED = 1,
    QA_WAIVED = 2
};

inline void __ExitInTime(int seconds)
{
    fprintf(stdout, "> exiting in %d seconds: ", seconds);
    fflush(stdout);
    time_t t;
    int count;
    for (t=time(0)+seconds, count=seconds; time(0) < t; count--) {
        fprintf(stdout, "%d...", count);
#ifdef WIN32
        Sleep(1000);
#else
        sleep(1);
#endif
    }
    fprintf(stdout,"done!\n\n"); 
	fflush(stdout);
}


inline void __shrQAFinish(int argc, const char **argv, int iStatus)
{
    // By default QATest is disabled and NoPrompt is Enabled (times out at seconds passed into __ExitInTime() )
    bool bQATest = false, bNoPrompt = true, bQuitInTime = true;
    const char *sStatus[] = { "FAILED", "PASSED", "WAIVED", NULL };
	
    for (int i=1; i < argc; i++) {
        int string_start = 0;
        while (argv[i][string_start] == '-')
           string_start++;

        const char *string_argv = &argv[i][string_start];
        if (!STRCASECMP(string_argv, "qatest")) {
           bQATest = true;
        }	
        // For SDK individual samples that don't specify -noprompt or -prompt, 
        // a 3 second delay will happen before exiting, giving a user time to view results
        if (!STRCASECMP(string_argv, "noprompt") || !STRCASECMP(string_argv, "help")) {
            bNoPrompt = true;
            bQuitInTime = false;
        }
        if (!STRCASECMP(string_argv, "prompt")) {
            bNoPrompt = false;
            bQuitInTime = false;
        }
    }

    int exename_start = findExeNameStart(argv[0]);
    if (bQATest) {
        fprintf(stdout, "&&&& %s %s", sStatus[iStatus], &(argv[0][exename_start]));
        for (int i=1; i < argc; i++) fprintf(stdout, " %s", argv[i]);
        fprintf(stdout, "\n");
    } else {
        fprintf(stdout, "[%s] test results...\n%s\n", &(argv[0][exename_start]), sStatus[iStatus]);
    }
    fflush(stdout);
    printf("\n"); fflush(stdout);
    if (bQuitInTime) {
        __ExitInTime(3);
    } else {
        if (!bNoPrompt) {
            fprintf(stdout, "\nPress <Enter> to exit...\n");
            fflush(stdout);
            getchar();
        }
    }
}

inline void __shrQAFinish2(bool bQATest, int argc, const char **argv, int iStatus)
{
    bool bQuitInTime = true;
    const char *sStatus[] = { "FAILED", "PASSED", "WAIVED", NULL };
	
    for (int i=1; i < argc; i++) {
        int string_start = 0;
        while (argv[i][string_start] == '-')
           string_start++;

        const char *string_argv = &argv[i][string_start];
        // For SDK individual samples that don't specify -noprompt or -prompt, 
        // a 3 second delay will happen before exiting, giving a user time to view results
        if (!STRCASECMP(string_argv, "noprompt") || !STRCASECMP(string_argv, "help")) {
            bQuitInTime = false;
        }
        if (!STRCASECMP(string_argv, "prompt")) {
            bQuitInTime = false;
        }
    }

    int exename_start = findExeNameStart(argv[0]);
    if (bQATest) {
        fprintf(stdout, "&&&& %s %s", sStatus[iStatus], &(argv[0][exename_start]));
        for (int i=1; i < argc; i++) fprintf(stdout, " %s", argv[i]);
        fprintf(stdout, "\n");
    } else {
        fprintf(stdout, "[%s] test results...\n%s\n", &(argv[0][exename_start]), sStatus[iStatus]);
    }
    fflush(stdout);
    
    if (bQuitInTime) {
        __ExitInTime(3);
    }
}

inline void shrQAFinishExit(int argc, const char **argv, int iStatus)
{
    __shrQAFinish(argc, argv, iStatus);

    exit(iStatus ? EXIT_SUCCESS : EXIT_FAILURE); 
}

inline void shrQAFinishExit2(bool bQAtest, int argc, const char **argv, int iStatus)
{
    __shrQAFinish2(bQAtest, argc, argv, iStatus);

    exit(iStatus ? EXIT_SUCCESS : EXIT_FAILURE);
}

#endif