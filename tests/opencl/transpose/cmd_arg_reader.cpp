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

/* CUda UTility Library */

// includes, file
#include "cmd_arg_reader.h"

// includes, system
#include <vector>

// internal unnamed namespace

namespace 
{
    // types, internal (class, enum, struct, union, typedef)

    // variables, internal

} // namespace {

// variables, exported

/*static*/ CmdArgReader* CmdArgReader::self;
/*static*/ char** CmdArgReader::rargv;
/*static*/ int CmdArgReader::rargc;

// functions, exported

////////////////////////////////////////////////////////////////////////////////
//! Public construction interface
//! @return a handle to the class instance
//! @param argc number of command line arguments (as given to main())
//! @param argv command line argument string (as given to main())
////////////////////////////////////////////////////////////////////////////////
/*static*/ void
CmdArgReader::init( const int argc, const char** argv) 
{  
    if ( NULL != self) 
    {
        return;
    }

    // command line arguments 
    if (( 0 == argc) || ( 0 == argv)) 
    {
        LOGIC_EXCEPTION( "No command line arguments given.");
    }

    self = new CmdArgReader();

    self->createArgsMaps( argc, argv);

    rargc = argc;
    rargv = const_cast<char**>( argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Constructor, default
////////////////////////////////////////////////////////////////////////////////
CmdArgReader::CmdArgReader() :
    args(),
    unprocessed(),
    iter(),
    iter_unprocessed()
{  }

////////////////////////////////////////////////////////////////////////////////
//! Destructor
////////////////////////////////////////////////////////////////////////////////
CmdArgReader::~CmdArgReader() 
{
    for( iter = args.begin(); iter != args.end(); ++iter) 
    {
        if( *(iter->second.first) == typeid( int)) 
        {
            delete static_cast<int*>( iter->second.second);
            break;
        }
        else if( *(iter->second.first) == typeid( bool)) 
        {
            delete static_cast<bool*>( iter->second.second);
            break;
        }
        else if( *(iter->second.first) == typeid( std::string)) 
        {
            delete static_cast<std::string*>( iter->second.second);
            break;
        }
        else if( *(iter->second.first) == typeid( std::vector< std::string>) ) 
        {
            delete static_cast< std::vector< std::string>* >( iter->second.second);
            break;
        }
        else if( *(iter->second.first) == typeid( std::vector<int>) ) 
        {
            delete static_cast< std::vector<int>* >( iter->second.second);
            break;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Read args as token value pair into map for better processing (Even the 
//! values remain strings until the parameter values is requested by the
//! program.)
//! @param argc the argument count (as given to 'main')
//! @param argv the char* array containing the command line arguments
////////////////////////////////////////////////////////////////////////////////
void
CmdArgReader::createArgsMaps( const int argc, const char** argv) {

    std::string token;
    std::string val_str;

    std::map< std::string, std::string> args;

    std::string::size_type pos;
    std::string arg;
    for( int i=1; i<argc; ++i) 
    {
        arg = argv[i];

        // check if valid command line argument: all arguments begin with - or --
        if (arg[0] != '-') 
        {
            RUNTIME_EXCEPTION("Invalid command line argument.");
        }

        int numDashes = (arg[1] == '-' ? 2 : 1);

        // check if only flag or if a value is given
        if ( (pos = arg.find( '=')) == std::string::npos) 
        {  
            unprocessed[ std::string( arg, numDashes, arg.length()-numDashes)] = "FLAG";                                  
        }
        else 
        {
            unprocessed[ std::string( arg, numDashes, pos-numDashes)] = 
                                      std::string( arg, pos+1, arg.length()-1);
        }
    }
}

