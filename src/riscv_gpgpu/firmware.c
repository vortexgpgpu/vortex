/*
 *  PicoSoC - A simple example SoC using PicoRV32
 *
 *  Copyright (C) 2017  Clifford Wolf <clifford@clifford.at>
 *
 *  Permission to use, copy, modify, and/or distribute this software for any
 *  purpose with or without fee is hereby granted, provided that the above
 *  copyright notice and this permission notice appear in all copies.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 *  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 *  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 *  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 *  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 *  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 */

#include <stdint.h>
#include <stdbool.h>

// #if !defined(ICEBREAKER) && !defined(HX8KDEMO)
// #  error "Set -DICEBREAKER or -DHX8KDEMO when compiling firmware.c"
// #endif

// a pointer to this is a null pointer, but the compiler does not
// know that because "sram" is a linker symbol from sections.lds.


#define reg_uart_data (*(volatile uint32_t*)0xFF000000)
// --------------------------------------------------------


// --------------------------------------------------------
int main()
{
	// FUNC DEFINITIONS
	void putchar(char);
	void print(const char *);
	char getchar_prompt(char *);
	void getsent_prompt(char *, char *);

	char arr[2];
	arr[1] = 0;
	char c;

	print("$  Booting..\n");


	while ((c = getchar_prompt("$  Press ENTER to continue.. ")) != '\n')
	{
		// arr[0] = c;
		// print("char printed[ ");
		// print(arr);
		// print(" ]\n");
	}
	// getchar_prompt("Press ENTER to continue..\n");


	print("$  \n");
	print("  ______           __       __                          _______  ______  ______   ______       __     __ \n");
	print(" /      \\         |  \\     |  \\                        |       \\|      \\/      \\ /      \\     |  \\   |  \\\n");
	print("|  $$$$$$\\ ______ | $$____  \\$$_______  __    __       | $$$$$$$\\\\$$$$$|  $$$$$$|  $$$$$$\\    | $$   | $$\n");
	print("| $$___\\$$/      \\| $$    \\|  |       \\|  \\  /  \\      | $$__| $$ | $$ | $$___\\$| $$   \\$_____| $$   | $$\n");
	print(" \\$$    \\|  $$$$$$| $$$$$$$| $| $$$$$$$\\\\$$\\/  $$      | $$    $$ | $$  \\$$    \\| $$    |      \\$$\\ /  $$\n");
	print(" _\\$$$$$$| $$  | $| $$  | $| $| $$  | $$ >$$  $$       | $$$$$$$\\ | $$  _\\$$$$$$| $$   __\\$$$$$$\\$$\\  $$ \n");
	print("|  \\__| $| $$__/ $| $$  | $| $| $$  | $$/  $$$$\\       | $$  | $$_| $$_|  \\__| $| $$__/  \\       \\$$ $$  \n");
	print(" \\$$    $| $$    $| $$  | $| $| $$  | $|  $$ \\$$\\      | $$  | $|   $$ \\\\$$    $$\\$$    $$        \\$$$   \n");
	print("  \\$$$$$$| $$$$$$$ \\$$   \\$$\\$$\\$$   \\$$\\$$   \\$$       \\$$   \\$$\\$$$$$$ \\$$$$$$  \\$$$$$$          \\$    \n");
	print("         | $$                                                                                            \n");
	print("         | $$                                                                                            \n");
	print("          \\$$                                                                                            \n");

	print("\n");


	char buf[50];

	while (1)
	{
		getsent_prompt("$  Do you want to play a guessing game? [y/n] ", buf);
		c = buf[0];
		if (c == 'y')
		{
			break;
		} else if (c == 'n')
		{
			return 0;
		} else
		{
			print("\n$  Wrong input, please try again\n");
		}
	}

	char n = '5';

	while (1)
	{
		getsent_prompt(">  Guess a number between 0 and 9: ", buf);
		c = buf[0];
		if (c < n)
		{
			print(">  The number is larger\n");
		} else if (c > n)
		{
			print(">  The number is smaller\n");
		} else
		{



			print("                                  _         _       _   _                 _ _ \n");
			print("                                 | |       | |     | | (_)               | | |\n");
			print("   ___ ___  _ __   __ _ _ __ __ _| |_ _   _| | __ _| |_ _  ___  _ __  ___| | |\n");
			print("  / __/ _ \\| '_ \\ / _` | '__/ _` | __| | | | |/ _` | __| |/ _ \\| '_ \\/ __| | |\n");
			print(" | (_| (_) | | | | (_| | | | (_| | |_| |_| | | (_| | |_| | (_) | | | \\__ \\_|_|\n");
			print("  \\___\\___/|_| |_|\\__, |_|  \\__,_|\\__|\\__,_|_|\\__,_|\\__|_|\\___/|_| |_|___(_|_)\n");
			print("                   __/ |                                                      \n");
			print("                  |___/                                                       \n");


			print("\n");


			break;
		}
	}

	print("$  Press CNTR-C to exit ");

	return 0;
}

void putchar(char c)
{
	if (c == '\n')
		putchar('\r');
	reg_uart_data = c;
}

void print(const char *p)
{
	while (*p)
		putchar(*(p++));
}

char getchar_prompt(char *prompt)
{
	int32_t c = -1;

	uint32_t cycles_begin, cycles_now, cycles;
	__asm__ volatile ("rdcycle %0" : "=r"(cycles_begin));


	cycles = 0;

	if (prompt)
		print(prompt);

	while (c == -1) {
		__asm__ volatile ("rdcycle %0" : "=r"(cycles_now));
		cycles = cycles_now - cycles_begin;
		// if (cycles > 1200000) {
		// 	if (prompt)
		// 		print(prompt);
		// 	cycles_begin = cycles_now;
		// }
		c = reg_uart_data;
	}

	print("\n");

	return c;
}


void getsent_prompt(char *prompt, char * buf)
{
	int32_t c = -1;

	uint32_t cycles_begin, cycles_now, cycles;
	__asm__ volatile ("rdcycle %0" : "=r"(cycles_begin));



	cycles = 0;

	if (prompt)
		print(prompt);

	int inx = 0;
	while (c != (int) '\n') {
		__asm__ volatile ("rdcycle %0" : "=r"(cycles_now));
		cycles = cycles_now - cycles_begin;
		// if (cycles > 1200000) {
		// 	if (prompt)
		// 		print(prompt);
		// 	cycles_begin = cycles_now;
		// }
		c = reg_uart_data;
		if (c != -1)
		{
			buf[inx] = c;
			inx++;
		}
	}

	// print("\n");

	return;
}