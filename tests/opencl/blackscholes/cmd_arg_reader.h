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

#ifndef _CMDARGREADER_H_
#define _CMDARGREADER_H_

// includes, system
#include <map>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <typeinfo>

// includes, project
#include "exception.h"

//! Preprocessed command line arguments
//! @note Lazy evaluation: The arguments are converted from strings to
//!       the correct data type upon request. Converted values are stored 
//!       in an additonal map so that no additional conversion is
//!       necessary. Arrays of command line arguments are stored in 
//!       std::vectors 
//! @note Usage: 
//!          const std::string* file = 
//!                       CmdArgReader::getArg< std::string>( "model")
//!          const std::vector< std::string>* files = 
//!          CmdArgReader::getArg< std::vector< std::string> >( "model")
//! @note All command line arguments begin with '--' followed by the token; 
//!   token and value are seperated by '='; example --samples=50
//! @note Arrays have the form --model=[one.obj,two.obj,three.obj] 
//!       (without whitespaces)

//! Command line argument parser
class CmdArgReader 
{
    template<class> friend class TestCmdArgReader;

protected:

    //! @param self handle to the only instance of this class
    static  CmdArgReader*  self;

public:

    //! Public construction interface
    //! @return a handle to the class instance
    //! @param argc number of command line arguments (as given to main())
    //! @param argv command line argument string (as given to main())
    static void init( const int argc, const char** argv);

public:

    //! Get the value of the command line argument with given name
    //! @return A const handle to the requested argument.
    //! If the argument does not exist or if it 
    //!  is not from type T NULL is returned
    //! @param name the name of the requested argument
    //! @note T the type of the argument requested
    template<class T>
        static inline const T* getArg( const std::string& name);

    //! Check if a command line argument with the given name exists
    //! @return  true if a command line argument with name \a name exists,
    //!          otherwise false
    //! @param name  name of the command line argument in question
    static inline bool existArg( const std::string& name);

    //! Get the original / raw argc program argument
    static inline int& getRArgc();

    //! Get the original / raw argv program argument
    static inline char**& getRArgv();

public:

    //! Destructor
    ~CmdArgReader();

protected:

    //! Constructor, default
    CmdArgReader();

private:

    // private helper functions

    //! Get the value of the command line argument with given name
    //! @note Private helper function for 'getArg' to work on the members
    //! @return A const handle to the requested argument. If the argument
    //!         does not exist or if it  is not from type T a NULL pointer
    //!         is returned.
    //! @param name the name of the requested argument
    //! @note T the type of the argument requested
    template<class T>
        inline const T* getArgHelper( const std::string& name);

    //! Check if a command line argument with name \a name exists
    //! @return true if a command line argument of name \a name exists, 
    //!         otherwise false
    //! @param name the name of the requested argument
    inline bool existArgHelper( const std::string& name) const;

    //! Read args as token value pair into map for better processing
    //!  (Even the values remain strings until the parameter values is 
    //!   requested by the program.)
    //! @param argc the argument count (as given to 'main')
    //! @param argv the char* array containing the command line arguments
    void  createArgsMaps( const int argc, const char** argv);

    //! Helper for "casting" the strings from the map with the unprocessed 
    //! values to the correct 
    //!  data type.
    //! @return true if conversion succeeded, otherwise false
    //! @param element the value as string
    //! @param val the value as type T
    template<class T>
        static inline bool convertToT( const std::string& element, T& val);

public:

    // typedefs internal

    //! container for a processed command line argument
    //! typeid is used to easily be able to decide if a re-requested token-value
    //! pair match the type of the first conversion
    typedef std::pair< const std::type_info*, void*>  ValType;
    //! map of already converted values
    typedef std::map< std::string, ValType >          ArgsMap;
    //! iterator for the map of already converted values
    typedef ArgsMap::iterator                         ArgsMapIter;
    typedef ArgsMap::const_iterator                   ConstArgsMapIter;

    //! map of unprocessed (means unconverted) token-value pairs
    typedef std::map< std::string, std::string>            UnpMap;
    //! iterator for the map of unprocessed (means unconverted) token-value pairs
    typedef std::map< std::string, std::string>::iterator  UnpMapIter;

private:

#ifdef _WIN32
#  pragma warning( disable: 4251)
#endif

    //! rargc original value of argc
    static  int  rargc;

    //! rargv contains command line arguments in raw format
    static char**  rargv;

    //! args Map containing the already converted token-value pairs
    ArgsMap     args;

    //! args Map containing the unprocessed / unconverted token-value pairs
    UnpMap     unprocessed;

    //! iter Iterator for the map with the already converted token-value 
    //!  pairs (to avoid frequent reallocation)
    ArgsMapIter iter;

    //! iter Iterator for the map with the unconverted token-value 
    //!  pairs (to avoid frequent reallocation)
    UnpMapIter iter_unprocessed;

#ifdef _WIN32
#  pragma warning( default: 4251)
#endif

private:

    //! Constructor, copy (not implemented)
    CmdArgReader( const CmdArgReader&);

    //! Assignment operator (not implemented)
    CmdArgReader& operator=( const CmdArgReader&);
};

// variables, exported (extern)

// functions, inlined (inline)

////////////////////////////////////////////////////////////////////////////////
//! Conversion function for command line argument arrays
//! @note This function is used each type for which no template specialization
//!  exist (which will cause errors if the type does not fulfill the std::vector
//!  interface).
////////////////////////////////////////////////////////////////////////////////
template<class T>
/*static*/ inline bool
CmdArgReader::convertToT( const std::string& element, T& val)
{
    // preallocate storage
    val.resize( std::count( element.begin(), element.end(), ',') + 1);

    unsigned int i = 0;
    std::string::size_type pos_start = 1;  // leave array prefix '['
    std::string::size_type pos_end = 0;

    // do for all elements of the comma seperated list
    while( std::string::npos != ( pos_end = element.find(',', pos_end+1)) ) 
    {
        // convert each element by the appropriate function
        if ( ! convertToT< typename T::value_type >( 
            std::string( element, pos_start, pos_end - pos_start), val[i])) 
        {
            return false;
        }

        pos_start = pos_end + 1;
        ++i;
    }

    std::string tmp1(  element, pos_start, element.length() - pos_start - 1);

    // process last element (leave array postfix ']')
    if ( ! convertToT< typename T::value_type >( std::string( element,
        pos_start,
        element.length() - pos_start - 1),
        val[i])) 
    {
        return false;
    }

    // possible to process all elements?
    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Conversion function for command line arguments of type int
////////////////////////////////////////////////////////////////////////////////
template<>
inline bool
CmdArgReader::convertToT<int>( const std::string& element, int& val) 
{
    std::istringstream ios( element);
    ios >> val;

    bool ret_val = false;
    if ( ios.eof()) 
    {
        ret_val = true;
    }

    return ret_val;
}

////////////////////////////////////////////////////////////////////////////////
//! Conversion function for command line arguments of type float
////////////////////////////////////////////////////////////////////////////////
template<>
inline bool
CmdArgReader::convertToT<float>( const std::string& element, float& val) 
{
    std::istringstream ios( element);
    ios >> val;

    bool ret_val = false;
    if ( ios.eof()) 
    {
        ret_val = true;
    }

    return ret_val;
}

////////////////////////////////////////////////////////////////////////////////
//! Conversion function for command line arguments of type double
////////////////////////////////////////////////////////////////////////////////
template<>
inline bool
CmdArgReader::convertToT<double>( const std::string& element, double& val) 
{
    std::istringstream ios( element);
    ios >> val;

    bool ret_val = false;
    if ( ios.eof()) 
    {
        ret_val = true;
    }

    return ret_val;
}

////////////////////////////////////////////////////////////////////////////////
//! Conversion function for command line arguments of type string
////////////////////////////////////////////////////////////////////////////////
template<>
inline bool
CmdArgReader::convertToT<std::string>( const std::string& element, 
                                      std::string& val)
{
    val = element;
    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Conversion function for command line arguments of type bool
////////////////////////////////////////////////////////////////////////////////
template<>
inline bool
CmdArgReader::convertToT<bool>( const std::string& element, bool& val) 
{
    // check if value is given as string-type { true | false }
    if ( "true" == element) 
    {
        val = true;
        return true;
    }
    else if ( "false" == element) 
    {
        val = false;
        return true;
    }
    // check if argument is given as integer { 0 | 1 }
    else 
    {
        int tmp;
        if ( convertToT<int>( element, tmp)) 
        {
            if ( 1 == tmp) 
            {
                val = true;
                return true;
            }
            else if ( 0 == tmp) 
            {
                val = false;
                return true;
            }
        }
    }

    return false;
}

////////////////////////////////////////////////////////////////////////////////
//! Get the value of the command line argument with given name
//! @return A const handle to the requested argument. If the argument does
//!  not exist or if it is not from type T NULL is returned
//! @param T the type of the argument requested
//! @param name the name of the requested argument
////////////////////////////////////////////////////////////////////////////////
template<class T>
/*static*/ const T*
CmdArgReader::getArg( const std::string& name) 
{
    if( ! self) 
    {
        RUNTIME_EXCEPTION("CmdArgReader::getArg(): CmdArgReader not initialized.");
        return NULL;
    }

    return self->getArgHelper<T>( name);
}

////////////////////////////////////////////////////////////////////////////////
//! Check if a command line argument with the given name exists
//! @return  true if a command line argument with name \a name exists,
//!          otherwise false
//! @param name  name of the command line argument in question
////////////////////////////////////////////////////////////////////////////////
/*static*/ inline bool 
CmdArgReader::existArg( const std::string& name) 
{
    if( ! self) 
    {
        RUNTIME_EXCEPTION("CmdArgReader::getArg(): CmdArgReader not initialized.");
        return false;
    }

    return self->existArgHelper( name);
}

////////////////////////////////////////////////////////////////////////////////
//! @brief Get the value of the command line argument with given name
//! @return A const handle to the requested argument. If the argument does
//!  not exist or if it is not from type T NULL is returned
//! @param T the type of the argument requested
//! @param name the name of the requested argument
////////////////////////////////////////////////////////////////////////////////
template<class T>
const T*
CmdArgReader::getArgHelper( const std::string& name) 
{
    // check if argument already processed and stored in correct type
    if ( args.end() != (iter = args.find( name))) 
    {
        if ( (*(iter->second.first)) == typeid( T) ) 
        {
            return (T*) iter->second.second;
        }
    }
    else 
    {
        T* tmp = new T;

        // check the array with unprocessed values
        if ( unprocessed.end() != (iter_unprocessed = unprocessed.find( name))) 
        {
            // try to "cast" the string to the type requested
            if ( convertToT< T >( iter_unprocessed->second, *tmp)) 
            {
                // add the token element pair to map of already converted values
                args[name] = std::make_pair( &(typeid( T)), (void*) tmp);

                return tmp;
            }
        }

        // not used while not inserted into the map -> cleanup
        delete tmp;
    }

    // failed, argument not available
    return NULL;
}

////////////////////////////////////////////////////////////////////////////////
//! Check if a command line argument with name \a name exists
//! @return true if a command line argument of name \a name exists, 
//!         otherwise false
//! @param name the name of the requested argument
////////////////////////////////////////////////////////////////////////////////
inline bool
CmdArgReader::existArgHelper( const std::string& name) const 
{
    bool ret_val = false;

    // check if argument already processed and stored in correct type
    if( args.end() != args.find( name)) 
    {
        ret_val = true;
    }
    else 
    {

        // check the array with unprocessed values
        if ( unprocessed.end() != unprocessed.find( name)) 
        {
            ret_val = true; 
        }
    }

    return ret_val;
}

////////////////////////////////////////////////////////////////////////////////
//! Get the original / raw argc program argument
////////////////////////////////////////////////////////////////////////////////
/*static*/ inline int&
CmdArgReader::getRArgc() 
{
    if( ! self) 
    {
        RUNTIME_EXCEPTION("CmdArgReader::getRArgc(): CmdArgReader not initialized.");
    }

    return rargc;
}

////////////////////////////////////////////////////////////////////////////////
//! Get the original / raw argv program argument
////////////////////////////////////////////////////////////////////////////////
/*static*/ inline char**&
CmdArgReader::getRArgv() 
{
    if( ! self) 
    {
        RUNTIME_EXCEPTION("CmdArgReader::getRArgc(): CmdArgReader not initialized.");
    }

    return rargv;
}

// functions, exported (extern)

#endif // #ifndef _CMDARGREADER_H_
