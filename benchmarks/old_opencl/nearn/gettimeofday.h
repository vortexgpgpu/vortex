
#ifdef _WIN32
#include <WinSock.h>
/**
Based on code seen at.

http://www.winehq.org/pipermail/wine-devel/2003-June/018082.html

http://msdn.microsoft.com/en-us/library/ms740560

*/
int gettimeofday(struct timeval *tv, struct timezone *tz);
#else
#include <sys/time.h>
#endif


