# C++ Coding Guidelines

Keep your code warning-free, consistent and easy to read.

## 1. Indentation
- Use **2 spaces** per indent level.
- **Do not** use tab characters.

```cpp
if (condition) {
  doSomething();
}
```

## 2. Brace Placement
- Opening braces go on the **same line** (K&R style).
- Closing braces go on their **own line**, aligned with the start of the declaration.

```cpp
class MyClass {
public:
  void foo() {
    // ...
  }
};
```

## 3. Spaces
- **One space** after keywords (`if`, `for`, `while`, `switch`).
- **No space** before function call parentheses.
- **Spaces** around binary operators.

```cpp
int x = a + b;
if (x == 0) {
  x = 1;
}
foo(x);
```

## 4. Function Declarations and Definitions
- In headers, parameter names may be omitted if unused.
- In definitions, **align** parameters vertically when **multi-line**.

```cpp
void foo(int a, int b, int c, int d, int e, int f, int g, int h
         int i);
```

## 5. Constructor Initializer Lists
- Place each initializer on its **own line** if the list is multi-line.
- Align subsequent lines under the colon.

```cpp
MyClass::MyClass(int a, int b)
    : a_(a)
    , b_(b) {
}
```

## 6. Comments
- Use `//` for **single-line** comments.
- Reserve `/* ... */` for **block** comments sparingly.
- Use Doxygen-style for **public API**:

```cpp
/// Computes the foo of bar.
/// @param x The input value.
/// @return The computed result.
int foo(int x);
```

## 7. Using #ifdef
- Preserve indent of nested code and shift pre-processor left
```cpp
regno_t to_regno(const reg_t& reg) {
#if defined(EXT_V_ENABLE)
    return {reg.rtype, reg.id};
#elif defined(EXT_F_ENABLE)
    return {reg.rtype, reg.id};
#else
    return {reg.id, 0};
#endif
endfunction
```