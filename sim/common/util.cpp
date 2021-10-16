#include "util.h"
#include <string.h>

// return file extension
const char* fileExtension(const char* filepath) {
    const char *ext = strrchr(filepath, '.');
    if (ext == NULL || ext == filepath) 
      return "";
    return ext + 1;
}