#ifndef timer_h
#define timer_h

#include <iostream>


class timer {
    public:
			   timer(const char *name = 0);
			   timer(const char *name, std::ostream &write_on_exit);

			   ~timer();

	void		   start(), stop();
	void		   reset();
	std::ostream   	   &print(std::ostream &);

	double             getTimeInSeconds();

    private:
	void		   print_time(std::ostream &, const char *which, double time) const;

	union {
	    long long	   total_time;
	    struct {
#if defined __PPC__
		int	   high, low;
#else
		int	   low, high;
#endif
	    };
	};

	unsigned long long count;
	const char	   *const name;
	std::ostream	   *const write_on_exit;

	static double	   CPU_speed_in_MHz, get_CPU_speed_in_MHz();
};


std::ostream &operator << (std::ostream &, class timer &);


inline void timer::reset()
{
    total_time = 0;
    count      = 0;
}


inline timer::timer(const char *name)
:
    name(name),
    write_on_exit(0)
{
    reset();
}


inline timer::timer(const char *name, std::ostream &write_on_exit)
:
    name(name),
    write_on_exit(&write_on_exit)
{
    reset();
}


inline timer::~timer()
{
    if (write_on_exit != 0)
	print(*write_on_exit);
}


inline void timer::start()
{
#if (defined __PATHSCALE__) && (defined __i386 || defined __x86_64)
    unsigned eax, edx;

    asm volatile ("rdtsc" : "=a" (eax), "=d" (edx));

    total_time -= ((unsigned long long) edx << 32) + eax;
#elif (defined __GNUC__ || defined __INTEL_COMPILER) && (defined __i386 || defined __x86_64)
    asm volatile
    (
	"rdtsc\n\t"
	"subl %%eax, %0\n\t"
	"sbbl %%edx, %1"
    :
	"+m" (low), "+m" (high)
    :
    :
	"eax", "edx"
    );
#else
#error Compiler/Architecture not recognized
#endif
}


inline void timer::stop()
{
#if (defined __PATHSCALE__) && (defined __i386 || defined __x86_64)
    unsigned eax, edx;

    asm volatile ("rdtsc" : "=a" (eax), "=d" (edx));

    total_time += ((unsigned long long) edx << 32) + eax;
#elif (defined __GNUC__ || defined __INTEL_COMPILER) && (defined __i386 || defined __x86_64)
    asm volatile
    (
	"rdtsc\n\t"
	"addl %%eax, %0\n\t"
	"adcl %%edx, %1"
    :
	"+m" (low), "+m" (high)
    :
    :
	"eax", "edx"
    );
#endif

    ++ count;
}

#endif
