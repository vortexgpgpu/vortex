#include <iostream>
#include <bitset>

#include <iostream>
#include <bitset>

// get float "in-memory" to exploit iee754 binary representation of floating point values
// use a u to trick compiler into letting you access float's bits directly
// bitwise operations cannot be done directly on iee754 representations per compiler settings
// ordering of the fields is important here
class MyFloat
{
private:
    void printBinary(int n, int i)
    {
        // Prints the binary representation
        // of a number n up to i-bits.
        int k;
        for (k = i - 1; k >= 0; k--)
        {

            if ((n >> k) & 1)
                std::cout << "1";
            else
                std::cout << "0";
        }
    }

public:
    union BFloat_t
    {
        float f;
        int i;
        struct
        {
            uint32_t dead : 16;    // don't use these, just place-holders
            uint32_t mantissa : 7; // Mantissa (fractional part) of the number
            uint32_t exponent : 8; // Exponent (power of 2) of the number
            uint32_t sign : 1;
        } parts;
    };

    void printBFloat(BFloat_t b)
    {
        std::cout << b.parts.sign << " | ";
        printBinary(b.parts.exponent, 8);
        std::cout << " | ";
        printBinary(b.parts.mantissa, 7);
        std::cout << std::endl;
    }

    BFloat_t in_mem;

    MyFloat(float x)
    {
        in_mem.f = x;
        printBFloat(in_mem);
    }

    MyFloat(uint8_t mantissa, uint8_t exponent, bool sign)
    {
        in_mem.parts.mantissa = mantissa & 0x7F;
        in_mem.parts.exponent = exponent;
        in_mem.parts.sign = (int)sign;

        std::cout << "inside constructor" << std::endl;
        std::cout << "bfloat:" << in_mem.f << std::endl;
        printBFloat(in_mem);
    }

    friend MyFloat operator+(const MyFloat &a, const MyFloat &b)
    {
        // get fields
        bool a_sign = (bool)a.in_mem.parts.sign;
        uint8_t a_exp = a.in_mem.parts.exponent - 127;
        uint8_t a_mantissa = a.in_mem.parts.mantissa | 0x80; // add in the implicit bit

        bool b_sign = (bool)b.in_mem.parts.sign;
        uint8_t b_exp = b.in_mem.parts.exponent - 127;
        uint8_t b_mantissa = b.in_mem.parts.mantissa | 0x80; // add in the implicit bit

        // align mantissas by shifting the smaller exponent to the larger exponent
        if (a_exp < b_exp)
        {
            a_mantissa >>= (b_exp - a_exp);
            a_exp = b_exp;
        }
        else
        {
            b_mantissa >>= (a_exp - b_exp);
            b_exp = a_exp;
        }

        // add mantissas and adjust exponent if necessary
        int sum_mantissa = a_mantissa + b_mantissa;
        if (sum_mantissa & 0x100)
        { // this val check might be wrong
            sum_mantissa >>= 1;
            a_exp++;
        }

        // build binary representation of result
        return MyFloat(sum_mantissa, a_exp, a_sign);
    }

    friend MyFloat operator*(const MyFloat &a, const MyFloat &b)
    {
        uint16_t a_exp = a.in_mem.parts.exponent;
        uint16_t b_exp = b.in_mem.parts.exponent;
        uint16_t a_mantissa = a.in_mem.parts.mantissa | 0x0080; // Add implicit bit
        uint16_t b_mantissa = b.in_mem.parts.mantissa | 0x0080; // Add implicit bi

        std::bitset<8> bits(a_exp);
        std::cout << "Binary a exp: " << bits << std::endl;

        bool product_sign = a.in_mem.parts.sign ^ b.in_mem.parts.sign;

        if (a_exp == 0xFF || b_exp == 0xff)
        {
            return MyFloat(0, 0xFF, product_sign);
        }
        // Multiply mantissas
        uint32_t product_mantissa = static_cast<uint32_t>(a_mantissa) * static_cast<uint32_t>(b_mantissa);

        // Add exponents
        int product_exp = a_exp + b_exp - 127;

        product_mantissa = (product_mantissa + 0x40) >> 7;

        // Round to nearest even (round half to even)
        if ((product_mantissa & 0x7F) == 0x40 && (product_mantissa & 0x1) != 0)
        {
            product_mantissa++;
        }
        if (product_mantissa & 0x0100)
        { // Check if the implicit bit shifted to the left
            product_mantissa >>= 1;
            product_exp++;
        }
        else
        {
            product_mantissa &= 0x7F; // Remove the implicit bit
        }
        return MyFloat(product_mantissa, product_exp, product_sign);
    }

    friend MyFloat operator/(const MyFloat &a, const MyFloat &b)
    {
        uint16_t a_exp = a.in_mem.parts.exponent;
        uint16_t b_exp = b.in_mem.parts.exponent;
        std::bitset<8> bits(b_exp);
        std::cout << "Binary b exp: " << bits << std::endl;
        uint16_t a_mantissa = a.in_mem.parts.mantissa | 0x0080; // Add implicit bit
        uint16_t b_mantissa = b.in_mem.parts.mantissa | 0x0080; // Add implicit bit

        bool quotient_sign = a.in_mem.parts.sign ^ b.in_mem.parts.sign;

        // Check if divisor is zero
        if (b_exp == 0 && b_mantissa == 0)
        {
            std::cout << "HERE" << std::endl;
            return MyFloat(0, 0xFF, quotient_sign); // Return infinity with the appropriate sign
        }

        // Check for infinity or zero in dividend
        if (a_exp == 0xFF || a_exp == 0)
        {
            return MyFloat(0, a_exp, quotient_sign);
        }

        // Subtract exponents
        int quotient_exp = a_exp - b_exp + 127;

        // Divide mantissas
        uint32_t quotient_mantissa = (static_cast<uint32_t>(a_mantissa) << 8) / static_cast<uint32_t>(b_mantissa);

        quotient_mantissa = (quotient_mantissa + 0x40) >> 8;

        // Round to nearest even (round half to even)
        if ((quotient_mantissa & 0x1) != 0 && (quotient_mantissa & 0x7F) == 0x40)
        {
            quotient_mantissa--;
        }
        else if ((quotient_mantissa & 0x7F) == 0x40)
        {
            quotient_mantissa++;
        }

        if (quotient_mantissa & 0x0100)
        { // Check if the implicit bit shifted to the left
            quotient_mantissa >>= 1;
            quotient_exp++;
        }
        else
        {
            quotient_mantissa &= 0x7F; // Remove the implicit bit
        }
        return MyFloat(quotient_mantissa, quotient_exp, quotient_sign);
    }
};

int main()
{
    float a = 8;
    float b = 0;
    std::cout << a << std::endl;

    std::bitset<sizeof(float) * 8> bits(*reinterpret_cast<unsigned long *>(&a));
    std::cout << "Binary representation of " << a << " is \n"
              << bits << std::endl;
    std::cout << "Binary representation of " << b << " is \n"
              << bits << std::endl;

    MyFloat bfloat_version_of_a(a);
    MyFloat bfloat_version_of_b(b);
    MyFloat c = bfloat_version_of_a / bfloat_version_of_b;

    // You can now print the result stored in c or perform other operations with it.

    return 0;
}
