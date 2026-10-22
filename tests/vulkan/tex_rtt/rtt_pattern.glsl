/* Round k's texel at (x, y), as four bytes; main.c computes the same. */
uint pattern(uint x, uint y, uint k)
{
   uint v = (x * 2654435761u) ^ (y * 40503u) ^ ((k + 1u) * 0x9e3779b9u);
   v ^= v >> 13;
   return v * 0x85ebca6bu;
}
