#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>

#include "timer.h"


using namespace std;

double timer::CPU_speed_in_MHz = timer::get_CPU_speed_in_MHz();


double timer::get_CPU_speed_in_MHz()
{
#if defined __linux__
    ifstream infile("/proc/cpuinfo");
    char     buffer[256], *colon;

    while (infile.good()) {
	infile.getline(buffer, 256);

	if (strncmp("cpu MHz", buffer, 7) == 0 && (colon = strchr(buffer, ':')) != 0)
	    return atof(colon + 2);
    }
#endif

    return 0.0;
}


void timer::print_time(ostream &str, const char *which, double time) const
{
    static const char *units[] = { " ns", " us", " ms", "  s", " ks", 0 };
    const char	      **unit   = units;

    time = 1000.0 * time / CPU_speed_in_MHz;

    while (time >= 999.5 && unit[1] != 0) {
	time /= 1000.0;
	++ unit;
    }

    str << which << " = " << setprecision(3) << setw(4) << time << *unit;
}


ostream &timer::print(ostream &str)
{
    str << left << setw(25) << (name != 0 ? name : "timer") << ": " << right;

    if (CPU_speed_in_MHz == 0)
	str << "could not determine CPU speed\n";
    else if (count > 0) {
	double total = static_cast<double>(total_time);

	print_time(str, "avg", total / static_cast<double>(count));
	print_time(str, ", total", total);
	str << ", count = " << setw(9) << count << '\n';
    }
    else
	str << "not used\n";

    return str;
}


ostream &operator << (ostream &str, class timer &timer)
{
    return timer.print(str);
}

double timer::getTimeInSeconds()
{
    double total = static_cast<double>(total_time);
    double res = (total / 1000000.0) / CPU_speed_in_MHz;
    return res;
}
