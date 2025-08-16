# maths for add, substract, multiply, divide, modulus, power, square root, factorial, gcd, lcm, prime check, fibonacci, great than, less than, equal to, not equal to, greater than or equal to, less than or equal to
# pip install langchain-community langchain-core langchain-openai

def add(a: int, b: int) -> int:
    """Adds two integers and returns the result."""
    return a + b
def subtract(a: int, b: int) -> int:
    """Subtracts second integer from first and returns the result."""
    return a - b
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the result."""
    return a * b
def divide(a: int, b: int) -> float:
    """Divides first integer by second and returns the result. Raises ZeroDivisionError if second integer is zero."""
    if b == 0:
        raise ZeroDivisionError("Division by zero is not allowed.")
    return a / b
def modulus(a: int, b: int) -> int:
    """Returns the modulus of first integer by second."""
    return a % b
def power(a: int, b: int) -> int:
    """Raises first integer to the power of second and returns the result."""
    return a ** b
def square_root(a: int) -> float:
    """Returns the square root of the integer."""
    if a < 0:
        raise ValueError("Cannot compute square root of negative number.")
    return a ** 0.5
def factorial(a: int) -> int:
    """Returns the factorial of the integer. Raises ValueError if integer is negative."""
    if a < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if a == 0 or a == 1:
        return 1
    result = 1
    for i in range(2, a + 1):
        result *= i
    return result
def gcd(a: int, b: int) -> int:
    """Returns the greatest common divisor of two integers."""
    while b:
        a, b = b, a % b
    return abs(a)
def lcm(a: int, b: int) -> int:
    """Returns the least common multiple of two integers."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)
def is_prime(n: int) -> bool:
    """Checks if the integer is a prime number."""
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
def fibonacci(n: int) -> int:
    """Returns the nth Fibonacci number. Raises ValueError if n is negative."""
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative indices.")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
def greater_than(a: int, b: int) -> bool:
    """Checks if first integer is greater than second."""
    return a > b
def less_than(a: int, b: int) -> bool:
    """Checks if first integer is less than second."""
    return a < b
def equal_to(a: int, b: int) -> bool:   
    """Checks if two integers are equal."""
    return a == b
def not_equal_to(a: int, b: int) -> bool:
    """Checks if two integers are not equal."""
    return a != b
def greater_than_or_equal_to(a: int, b: int) -> bool:
    """Checks if first integer is greater than or equal to second."""
    return a >= b
def less_than_or_equal_to(a: int, b: int) -> bool:
    """Checks if first integer is less than or equal to second."""
    return a <= b   