// bin2mif -- Convert binary file to Memory Initialization File used by some
//            FPGA toolchains.

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <stack>

int main(int argc, char** argv) {
  using namespace std;

  if (argc != 5) {
    cerr << "Usage:\n  " << argv[0] << ' ' << "<word size(bytes)>"
         << " <mem size(bytes)> <in file> <out file>\n";;
    return 1;
  }

  ifstream in(argv[3]);
  ofstream out(argv[4]);

  if (!in) {
    cerr << "Failed to open input file \"" << argv[3] << "\"\n";
    return 1;
  }

  if (!out) {
    cerr << "Failed to open output file \"" << argv[4] << "\"\n";
    return 1;
  }

  unsigned word(atol(argv[1])), mem_sz(atol(argv[2])/word);

  out << "DEPTH = " << mem_sz << ";\n"
      << "WIDTH = " << word*8 << ";\n"
      << "ADDRESS_RADIX = HEX;\n"
      << "DATA_RADIX = HEX;\n"
      << "CONTENT\n"
      << "BEGIN\n";

  // HARP is little endian, so no matter what the endianness of the machine on
  // which this utility runs, this swapping of the byte order when constructing
  // hex values is necessary.
  for (unsigned j = 0; j < mem_sz; ++j) {
    stack<unsigned char> bytes;

    out << setw(4) << setfill('0') << hex << j << " : ";
    for (unsigned i = 0; i < word; ++i) {
      if (!in.eof()) bytes.push(in.get());
      else           bytes.push(0);
    }
    for (unsigned i = 0; i < word; ++i) {
      out << hex << setw(2) << setfill('0') << unsigned(bytes.top());
      bytes.pop();
    }
    out << ";\n";
    if (in.eof()) break;
  }

  out << "END;";

  return 0;
}
