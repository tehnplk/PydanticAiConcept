#!/usr/bin/env python3
"""
FastMCP calculator server for basic arithmetic operations.
Supports multiple numbers in single operations using lists.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from fastmcp import FastMCP
import re
from functools import reduce

# Initialize FastMCP server
mcp = FastMCP("calculator-server")

# Memory to store the last result
memory = 0.0

@mcp.tool()
def add(numbers: List[float]) -> Dict[str, Any]:
    """
    Add multiple numbers together.
    
    Args:
        numbers: List of numbers to add
        
    Returns:
        Dictionary with operation details and result
    """
    if not numbers:
        raise ValueError("At least one number is required")
    
    result = sum(numbers)
    numbers_str = " + ".join(map(str, numbers))
    
    return {
        "operation": numbers_str,
        "result": result,
        "type": "addition",
        "operands": numbers,
        "count": len(numbers)
    }

@mcp.tool()
def subtract(numbers: List[float]) -> Dict[str, Any]:
    """
    Subtract multiple numbers from the first number.
    Example: subtract([10, 3, 2]) = 10 - 3 - 2 = 5
    
    Args:
        numbers: List of numbers (first - second - third - ...)
        
    Returns:
        Dictionary with operation details and result
    """
    if not numbers:
        raise ValueError("At least one number is required")
    
    if len(numbers) == 1:
        result = numbers[0]
    else:
        result = numbers[0]
        for num in numbers[1:]:
            result -= num
    
    numbers_str = " - ".join(map(str, numbers))
    
    return {
        "operation": numbers_str,
        "result": result,
        "type": "subtraction",
        "operands": numbers,
        "count": len(numbers)
    }

@mcp.tool()
def multiply(numbers: List[float]) -> Dict[str, Any]:
    """
    Multiply multiple numbers together.
    
    Args:
        numbers: List of numbers to multiply
        
    Returns:
        Dictionary with operation details and result
    """
    if not numbers:
        raise ValueError("At least one number is required")
    
    result = 1
    for num in numbers:
        result *= num
    
    numbers_str = " × ".join(map(str, numbers))
    
    return {
        "operation": numbers_str,
        "result": result,
        "type": "multiplication",
        "operands": numbers,
        "count": len(numbers)
    }

@mcp.tool()
def divide(numbers: List[float]) -> Dict[str, Any]:
    """
    Divide multiple numbers sequentially.
    Example: divide([20, 4, 2]) = 20 ÷ 4 ÷ 2 = 2.5
    
    Args:
        numbers: List of numbers (first ÷ second ÷ third ÷ ...)
        
    Returns:
        Dictionary with operation details and result
        
    Raises:
        ValueError: If any divisor is zero
    """
    if not numbers:
        raise ValueError("At least one number is required")
    
    # Check for zero divisors (except the first number)
    if len(numbers) > 1 and any(num == 0 for num in numbers[1:]):
        raise ValueError("Cannot divide by zero")
    
    if len(numbers) == 1:
        result = numbers[0]
    else:
        result = numbers[0]
        for num in numbers[1:]:
            result /= num
    
    numbers_str = " ÷ ".join(map(str, numbers))
    
    return {
        "operation": numbers_str,
        "result": result,
        "type": "division",
        "operands": numbers,
        "count": len(numbers)
    }

@mcp.tool()
def add_two(a: float, b: float) -> Dict[str, Any]:
    """
    Add two numbers together (convenience function).
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Dictionary with operation details and result
    """
    return add([a, b])

@mcp.tool()
def subtract_two(a: float, b: float) -> Dict[str, Any]:
    """
    Subtract second number from first number (convenience function).
    
    Args:
        a: First number (minuend)
        b: Second number (subtrahend)
        
    Returns:
        Dictionary with operation details and result
    """
    return subtract([a, b])

@mcp.tool()
def multiply_two(a: float, b: float) -> Dict[str, Any]:
    """
    Multiply two numbers together (convenience function).
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Dictionary with operation details and result
    """
    return multiply([a, b])

@mcp.tool()
def divide_two(a: float, b: float) -> Dict[str, Any]:
    """
    Divide first number by second number (convenience function).
    
    Args:
        a: Dividend (number to be divided)
        b: Divisor (number to divide by)
        
    Returns:
        Dictionary with operation details and result
    """
    return divide([a, b])

@mcp.tool()
def calculate_expression(expression: str) -> Dict[str, Any]:
    """
    Calculate a mathematical expression with +, -, *, / operations.
    Supports parentheses and follows order of operations.
    
    Args:
        expression: Mathematical expression as string (e.g., "2 + 3 * 4 + 5")
        
    Returns:
        Dictionary with expression and result
        
    Raises:
        ValueError: If expression is invalid
    """
    # Replace × and ÷ with standard operators
    clean_expr = expression.replace('×', '*').replace('÷', '/')
    
    # Validate expression (only allow numbers, operators, parentheses, spaces, and decimal points)
    if not re.match(r'^[0-9+\-*/().\s]+$', clean_expr):
        raise ValueError("Invalid characters in expression")
    
    try:
        # Use eval safely with restricted scope
        result = eval(clean_expr, {"__builtins__": {}}, {})
        return {
            "expression": expression,
            "result": float(result),
            "type": "expression"
        }
    except ZeroDivisionError:
        raise ValueError("Division by zero in expression")
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")

@mcp.tool()
def average(numbers: List[float]) -> Dict[str, Any]:
    """
    Calculate the average (mean) of multiple numbers.
    
    Args:
        numbers: List of numbers
        
    Returns:
        Dictionary with operation details and result
    """
    if not numbers:
        raise ValueError("At least one number is required")
    
    result = sum(numbers) / len(numbers)
    numbers_str = f"avg({', '.join(map(str, numbers))})"
    
    return {
        "operation": numbers_str,
        "result": result,
        "type": "average",
        "operands": numbers,
        "count": len(numbers),
        "sum": sum(numbers)
    }

@mcp.tool()
def sum_range(start: int, end: int) -> Dict[str, Any]:
    """
    Calculate the sum of integers in a range.
    
    Args:
        start: Start of range (inclusive)
        end: End of range (inclusive)
        
    Returns:
        Dictionary with operation details and result
    """
    if start > end:
        start, end = end, start  # Swap if needed
    
    numbers = list(range(start, end + 1))
    result = sum(numbers)
    
    return {
        "operation": f"sum({start} to {end})",
        "result": result,
        "type": "sum_range",
        "range": [start, end],
        "count": len(numbers),
        "numbers": numbers if len(numbers) <= 20 else f"[{start}, {start+1}, ..., {end-1}, {end}]"
    }

@mcp.tool()
def factorial(n: int) -> Dict[str, Any]:
    """
    Calculate the factorial of a number.
    
    Args:
        n: Non-negative integer
        
    Returns:
        Dictionary with operation details and result
        
    Raises:
        ValueError: If n is negative or too large
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n > 170:  # Prevent overflow
        raise ValueError("Number too large for factorial calculation")
    
    result = 1
    for i in range(1, n + 1):
        result *= i
    
    return {
        "operation": f"{n}!",
        "result": result,
        "type": "factorial",
        "operand": n
    }

@mcp.tool()
def power(base: float, exponent: float) -> Dict[str, Any]:
    """
    Calculate base raised to the power of exponent.
    
    Args:
        base: The base number
        exponent: The exponent
        
    Returns:
        Dictionary with operation details and result
    """
    result = base ** exponent
    
    return {
        "operation": f"{base}^{exponent}",
        "result": result,
        "type": "power",
        "base": base,
        "exponent": exponent
    }

@mcp.tool()
def power_chain(numbers: List[float]) -> Dict[str, Any]:
    """
    Calculate power chain: first^second^third^...
    Example: power_chain([2, 3, 2]) = 2^(3^2) = 2^9 = 512
    
    Args:
        numbers: List of numbers for power chain
        
    Returns:
        Dictionary with operation details and result
    """
    if not numbers:
        raise ValueError("At least one number is required")
    
    if len(numbers) == 1:
        result = numbers[0]
    else:
        # Calculate from right to left (right associative)
        result = numbers[-1]
        for i in range(len(numbers) - 2, -1, -1):
            result = numbers[i] ** result
    
    numbers_str = "^".join(map(str, numbers))
    
    return {
        "operation": numbers_str,
        "result": result,
        "type": "power_chain",
        "operands": numbers,
        "count": len(numbers)
    }

@mcp.tool()
def percentage(number: float, percent: float) -> Dict[str, Any]:
    """
    Calculate percentage of a number.
    
    Args:
        number: The base number
        percent: The percentage (e.g., 25 for 25%)
        
    Returns:
        Dictionary with percentage calculation
    """
    result = (number * percent) / 100
    return {
        "operation": f"{percent}% of {number}",
        "result": result,
        "type": "percentage",
        "base_number": number,
        "percentage": percent
    }

@mcp.tool()
def square_root(number: float) -> Dict[str, Any]:
    """
    Calculate the square root of a number.
    
    Args:
        number: The number to find square root of
        
    Returns:
        Dictionary with operation details and result
        
    Raises:
        ValueError: If number is negative
    """
    if number < 0:
        raise ValueError("Cannot calculate square root of negative number")
    
    result = number ** 0.5
    
    return {
        "operation": f"√{number}",
        "result": result,
        "type": "square_root",
        "operand": number
    }

@mcp.tool()
def absolute_value(number: float) -> Dict[str, Any]:
    """
    Calculate the absolute value of a number.
    
    Args:
        number: The number
        
    Returns:
        Dictionary with operation details and result
    """
    result = abs(number)
    
    return {
        "operation": f"|{number}|",
        "result": result,
        "type": "absolute_value",
        "operand": number
    }

@mcp.tool()
def median(numbers: List[float]) -> Dict[str, Any]:
    """
    Calculate the median (middle value) of a list of numbers.
    
    Args:
        numbers: List of numbers
        
    Returns:
        Dictionary with operation details and result
    """
    if not numbers:
        raise ValueError("At least one number is required")
    
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    
    if n % 2 == 0:
        # Even number of elements - average of two middle values
        result = (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        # Odd number of elements - middle value
        result = sorted_numbers[n//2]
    
    return {
        "operation": f"median({', '.join(map(str, numbers))})",
        "result": result,
        "type": "median",
        "operands": numbers,
        "sorted_data": sorted_numbers,
        "count": n
    }

@mcp.tool()
def mode(numbers: List[float]) -> Dict[str, Any]:
    """
    Calculate the mode (most frequent value) of a list of numbers.
    
    Args:
        numbers: List of numbers
        
    Returns:
        Dictionary with operation details and result
    """
    if not numbers:
        raise ValueError("At least one number is required")
    
    # Count frequency of each number
    frequency = {}
    for num in numbers:
        frequency[num] = frequency.get(num, 0) + 1
    
    # Find maximum frequency
    max_freq = max(frequency.values())
    
    # Get all numbers with maximum frequency
    modes = [num for num, freq in frequency.items() if freq == max_freq]
    
    return {
        "operation": f"mode({', '.join(map(str, numbers))})",
        "result": modes[0] if len(modes) == 1 else modes,
        "type": "mode",
        "operands": numbers,
        "all_modes": modes,
        "frequency_table": frequency,
        "max_frequency": max_freq,
        "is_multimodal": len(modes) > 1
    }

@mcp.tool()
def range_stat(numbers: List[float]) -> Dict[str, Any]:
    """
    Calculate the range (difference between max and min) of a list of numbers.
    
    Args:
        numbers: List of numbers
        
    Returns:
        Dictionary with operation details and result
    """
    if not numbers:
        raise ValueError("At least one number is required")
    
    min_val = min(numbers)
    max_val = max(numbers)
    result = max_val - min_val
    
    return {
        "operation": f"range({', '.join(map(str, numbers))})",
        "result": result,
        "type": "range",
        "operands": numbers,
        "minimum": min_val,
        "maximum": max_val,
        "count": len(numbers)
    }

@mcp.tool()
def variance(numbers: List[float], sample: bool = False) -> Dict[str, Any]:
    """
    Calculate the variance of a list of numbers.
    
    Args:
        numbers: List of numbers
        sample: If True, calculate sample variance (n-1), if False, population variance (n)
        
    Returns:
        Dictionary with operation details and result
    """
    if not numbers:
        raise ValueError("At least one number is required")
    
    if sample and len(numbers) < 2:
        raise ValueError("Sample variance requires at least 2 numbers")
    
    mean_val = sum(numbers) / len(numbers)
    squared_diffs = [(x - mean_val) ** 2 for x in numbers]
    
    divisor = len(numbers) - 1 if sample else len(numbers)
    result = sum(squared_diffs) / divisor
    
    variance_type = "sample" if sample else "population"
    
    return {
        "operation": f"{variance_type}_variance({', '.join(map(str, numbers))})",
        "result": result,
        "type": f"{variance_type}_variance",
        "operands": numbers,
        "mean": mean_val,
        "count": len(numbers),
        "divisor": divisor
    }

@mcp.tool()
def standard_deviation(numbers: List[float], sample: bool = False) -> Dict[str, Any]:
    """
    Calculate the standard deviation of a list of numbers.
    
    Args:
        numbers: List of numbers
        sample: If True, calculate sample std dev (n-1), if False, population std dev (n)
        
    Returns:
        Dictionary with operation details and result
    """
    if not numbers:
        raise ValueError("At least one number is required")
    
    if sample and len(numbers) < 2:
        raise ValueError("Sample standard deviation requires at least 2 numbers")
    
    # Calculate variance directly
    mean_val = sum(numbers) / len(numbers)
    squared_diffs = [(x - mean_val) ** 2 for x in numbers]
    
    divisor = len(numbers) - 1 if sample else len(numbers)
    variance_val = sum(squared_diffs) / divisor
    
    # Standard deviation is square root of variance
    result = variance_val ** 0.5
    
    variance_type = "sample" if sample else "population"
    
    return {
        "operation": f"{variance_type}_std_dev({', '.join(map(str, numbers))})",
        "result": result,
        "type": f"{variance_type}_standard_deviation",
        "operands": numbers,
        "variance": variance_val,
        "mean": mean_val,
        "count": len(numbers),
        "divisor": divisor
    }

@mcp.tool()
def quartiles(numbers: List[float]) -> Dict[str, Any]:
    """
    Calculate the quartiles (Q1, Q2, Q3) of a list of numbers.
    
    Args:
        numbers: List of numbers
        
    Returns:
        Dictionary with quartile information
    """
    if not numbers:
        raise ValueError("At least one number is required")
    
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    
    # Q2 (median)
    if n % 2 == 0:
        q2 = (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        q2 = sorted_numbers[n//2]
    
    # Q1 (median of lower half)
    lower_half = sorted_numbers[:n//2]
    if len(lower_half) % 2 == 0 and len(lower_half) > 0:
        q1 = (lower_half[len(lower_half)//2 - 1] + lower_half[len(lower_half)//2]) / 2
    elif len(lower_half) > 0:
        q1 = lower_half[len(lower_half)//2]
    else:
        q1 = sorted_numbers[0]
    
    # Q3 (median of upper half)
    upper_half = sorted_numbers[(n+1)//2:] if n % 2 == 1 else sorted_numbers[n//2:]
    if len(upper_half) % 2 == 0 and len(upper_half) > 0:
        q3 = (upper_half[len(upper_half)//2 - 1] + upper_half[len(upper_half)//2]) / 2
    elif len(upper_half) > 0:
        q3 = upper_half[len(upper_half)//2]
    else:
        q3 = sorted_numbers[-1]
    
    iqr = q3 - q1
    
    return {
        "operation": f"quartiles({', '.join(map(str, numbers))})",
        "result": {"Q1": q1, "Q2": q2, "Q3": q3, "IQR": iqr},
        "type": "quartiles",
        "operands": numbers,
        "sorted_data": sorted_numbers,
        "q1": q1,
        "q2_median": q2,
        "q3": q3,
        "iqr": iqr,
        "count": n
    }

@mcp.tool()
def five_number_summary(numbers: List[float]) -> Dict[str, Any]:
    """
    Calculate the five-number summary (min, Q1, median, Q3, max) of a list of numbers.
    
    Args:
        numbers: List of numbers
        
    Returns:
        Dictionary with five-number summary
    """
    if not numbers:
        raise ValueError("At least one number is required")
    
    quartile_result = quartiles(numbers)
    min_val = min(numbers)
    max_val = max(numbers)
    
    return {
        "operation": f"five_number_summary({', '.join(map(str, numbers))})",
        "result": {
            "minimum": min_val,
            "q1": quartile_result["q1"],
            "median": quartile_result["q2_median"],
            "q3": quartile_result["q3"],
            "maximum": max_val
        },
        "type": "five_number_summary",
        "operands": numbers,
        "minimum": min_val,
        "q1": quartile_result["q1"],
        "median": quartile_result["q2_median"],
        "q3": quartile_result["q3"],
        "maximum": max_val,
        "iqr": quartile_result["iqr"],
        "range": max_val - min_val,
        "count": len(numbers)
    }

@mcp.tool()
def descriptive_statistics(numbers: List[float], sample: bool = False) -> Dict[str, Any]:
    """
    Calculate comprehensive descriptive statistics for a list of numbers.
    
    Args:
        numbers: List of numbers
        sample: If True, use sample calculations for variance/std dev
        
    Returns:
        Dictionary with comprehensive statistics
    """
    if not numbers:
        raise ValueError("At least one number is required")
    
    # Basic calculations
    mean_val = sum(numbers) / len(numbers)
    min_val = min(numbers)
    max_val = max(numbers)
    range_val = max_val - min_val
    
    # Median calculation
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 0:
        median_val = (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        median_val = sorted_numbers[n//2]
    
    # Mode calculation
    frequency = {}
    for num in numbers:
        frequency[num] = frequency.get(num, 0) + 1
    max_freq = max(frequency.values())
    modes = [num for num, freq in frequency.items() if freq == max_freq]
    mode_val = modes[0] if len(modes) == 1 else modes
    
    # Variance and standard deviation
    squared_diffs = [(x - mean_val) ** 2 for x in numbers]
    divisor = len(numbers) - 1 if sample else len(numbers)
    variance_val = sum(squared_diffs) / divisor if divisor > 0 else 0
    std_dev_val = variance_val ** 0.5
    
    # Quartiles
    if n % 2 == 0:
        q2 = (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        q2 = sorted_numbers[n//2]
    
    lower_half = sorted_numbers[:n//2]
    if len(lower_half) % 2 == 0 and len(lower_half) > 0:
        q1 = (lower_half[len(lower_half)//2 - 1] + lower_half[len(lower_half)//2]) / 2
    elif len(lower_half) > 0:
        q1 = lower_half[len(lower_half)//2]
    else:
        q1 = sorted_numbers[0]
    
    upper_half = sorted_numbers[(n+1)//2:] if n % 2 == 1 else sorted_numbers[n//2:]
    if len(upper_half) % 2 == 0 and len(upper_half) > 0:
        q3 = (upper_half[len(upper_half)//2 - 1] + upper_half[len(upper_half)//2]) / 2
    elif len(upper_half) > 0:
        q3 = upper_half[len(upper_half)//2]
    else:
        q3 = sorted_numbers[-1]
    
    iqr = q3 - q1
    
    return {
        "operation": f"descriptive_stats({', '.join(map(str, numbers))})",
        "result": {
            "count": len(numbers),
            "mean": mean_val,
            "median": median_val,
            "mode": mode_val,
            "range": range_val,
            "variance": variance_val,
            "standard_deviation": std_dev_val,
            "minimum": min_val,
            "maximum": max_val,
            "q1": q1,
            "q3": q3,
            "iqr": iqr
        },
        "type": "descriptive_statistics",
        "operands": numbers,
        "sample_statistics": sample,
        "frequency_table": frequency
    }

@mcp.tool()
def percentile(numbers: List[float], p: float) -> Dict[str, Any]:
    """
    Calculate the p-th percentile of a list of numbers.
    
    Args:
        numbers: List of numbers
        p: Percentile value (0-100)
        
    Returns:
        Dictionary with percentile information
        
    Raises:
        ValueError: If percentile is not between 0 and 100
    """
    if not numbers:
        raise ValueError("At least one number is required")
    
    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    
    # Calculate index
    index = (p / 100) * (n - 1)
    
    if index == int(index):
        # Exact index
        result = sorted_numbers[int(index)]
    else:
        # Interpolate between two values
        lower_index = int(index)
        upper_index = lower_index + 1
        weight = index - lower_index
        
        if upper_index < n:
            result = sorted_numbers[lower_index] * (1 - weight) + sorted_numbers[upper_index] * weight
        else:
            result = sorted_numbers[lower_index]
    
    return {
        "operation": f"{p}th_percentile({', '.join(map(str, numbers))})",
        "result": result,
        "type": "percentile",
        "operands": numbers,
        "percentile": p,
        "sorted_data": sorted_numbers,
        "count": n
    }

@mcp.tool()
def store_in_memory(value: float) -> Dict[str, Any]:
    """
    Store a value in calculator memory.
    
    Args:
        value: Value to store in memory
        
    Returns:
        Dictionary confirming storage
    """
    global memory
    memory = value
    return {
        "operation": "memory store",
        "stored_value": memory,
        "type": "memory"
    }

@mcp.tool()
def recall_memory() -> Dict[str, Any]:
    """
    Recall the value stored in calculator memory.
    
    Returns:
        Dictionary with memory value
    """
    return {
        "operation": "memory recall",
        "memory_value": memory,
        "type": "memory"
    }

@mcp.tool()
def clear_memory() -> Dict[str, Any]:
    """
    Clear the calculator memory (set to 0).
    
    Returns:
        Dictionary confirming memory clear
    """
    global memory
    old_value = memory
    memory = 0.0
    return {
        "operation": "memory clear",
        "old_value": old_value,
        "new_value": memory,
        "type": "memory"
    }

@mcp.tool()
def batch_calculate(operations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform multiple calculations in sequence.
    
    Args:
        operations: List of operation dictionaries
                   Examples:
                   - {"type": "add", "numbers": [5, 3, 2]}
                   - {"type": "multiply", "numbers": [4, 2, 3]}
                   - {"type": "expression", "expression": "2 + 3 * 4"}
                   - {"type": "add_two", "a": 5, "b": 3}
                   - {"type": "average", "numbers": [1, 2, 3, 4, 5]}
                   - {"type": "median", "numbers": [1, 2, 3, 4, 5]}
                   - {"type": "std_dev", "numbers": [1, 2, 3], "sample": true}
        
    Returns:
        Dictionary with batch calculation results
    """
    results = []
    
    for i, op in enumerate(operations):
        try:
            op_type = op.get("type")
            
            if op_type == "add":
                result = add(op["numbers"])
            elif op_type == "subtract":
                result = subtract(op["numbers"])
            elif op_type == "multiply":
                result = multiply(op["numbers"])
            elif op_type == "divide":
                result = divide(op["numbers"])
            elif op_type == "add_two":
                result = add_two(op["a"], op["b"])
            elif op_type == "subtract_two":
                result = subtract_two(op["a"], op["b"])
            elif op_type == "multiply_two":
                result = multiply_two(op["a"], op["b"])
            elif op_type == "divide_two":
                result = divide_two(op["a"], op["b"])
            elif op_type == "expression":
                result = calculate_expression(op["expression"])
            elif op_type == "percentage":
                result = percentage(op["number"], op["percent"])
            elif op_type == "power":
                result = power(op["base"], op["exponent"])
            elif op_type == "average":
                result = average(op["numbers"])
            elif op_type == "factorial":
                result = factorial(op["n"])
            elif op_type == "median":
                result = median(op["numbers"])
            elif op_type == "mode":
                result = mode(op["numbers"])
            elif op_type == "range":
                result = range_stat(op["numbers"])
            elif op_type == "variance":
                result = variance(op["numbers"], op.get("sample", False))
            elif op_type == "std_dev":
                result = standard_deviation(op["numbers"], op.get("sample", False))
            elif op_type == "quartiles":
                result = quartiles(op["numbers"])
            elif op_type == "descriptive_stats":
                result = descriptive_statistics(op["numbers"], op.get("sample", False))
            elif op_type == "percentile":
                result = percentile(op["numbers"], op["p"])
            else:
                raise ValueError(f"Unknown operation type: {op_type}")
            
            results.append({
                "step": i + 1,
                "success": True,
                **result
            })
            
        except Exception as e:
            results.append({
                "step": i + 1,
                "success": False,
                "error": str(e),
                "operation_input": op
            })
    
    successful_ops = len([r for r in results if r["success"]])
    
    return {
        "batch_operation": "multiple calculations",
        "total_operations": len(operations),
        "successful_operations": successful_ops,
        "results": results
    }

@mcp.resource("file://calculator-help")
def calculator_help() -> str:
    """
    Get help information for the calculator.
    
    Returns:
        Help text with available operations
    """
    return """
Multi-Number Calculator Server with Statistics
==============================================

Multi-Number Operations (use lists):
- add([1, 2, 3, 4, 5]): Add multiple numbers
- subtract([100, 20, 5, 10]): Subtract from first number
- multiply([2, 3, 4, 5]): Multiply multiple numbers  
- divide([1000, 10, 5, 2]): Divide sequentially

Two-Number Operations (convenience):
- add_two(5, 3): Add two numbers
- subtract_two(10, 4): Subtract two numbers
- multiply_two(7, 6): Multiply two numbers
- divide_two(15, 3): Divide two numbers

Basic Statistics:
- average([1, 2, 3, 4, 5]): Calculate mean (3.0)
- median([1, 2, 3, 4, 5]): Calculate median (3.0)
- mode([1, 2, 2, 3, 4]): Calculate mode (2)
- range_stat([1, 2, 3, 4, 5]): Calculate range (4)

Advanced Statistics:
- variance([1, 2, 3, 4, 5], sample=False): Population variance
- standard_deviation([1, 2, 3, 4, 5], sample=True): Sample std dev
- quartiles([1, 2, 3, 4, 5, 6, 7]): Q1, Q2, Q3, IQR
- five_number_summary([data]): Min, Q1, Median, Q3, Max
- descriptive_statistics([data], sample=False): Complete summary
- percentile([1, 2, 3, 4, 5], 75): Calculate 75th percentile

Mathematical Operations:
- calculate_expression("2 + 3 * 4 + 5"): Complex expressions
- sum_range(1, 100): Sum integers in range
- factorial(5): Calculate n!
- power(2, 8): Calculate 2^8
- power_chain([2, 3, 2]): Calculate 2^(3^2)
- square_root(25): Calculate √25
- absolute_value(-5): Calculate |−5|
- percentage(200, 25): Calculate 25% of 200

Memory Functions:
- store_in_memory(value): Store in memory
- recall_memory(): Get memory value  
- clear_memory(): Clear memory

Batch Operations:
- batch_calculate([{...}, {...}]): Multiple operations

Statistics Examples:
- median([1, 3, 5, 7, 9]) = 5
- mode([1, 2, 2, 3, 3, 3]) = 3
- standard_deviation([2, 4, 6, 8, 10], sample=True) = 3.16
- quartiles([1, 2, 3, 4, 5, 6, 7]) = {Q1: 2, Q2: 4, Q3: 6, IQR: 4}
- percentile([10, 20, 30, 40, 50], 25) = 20

Current Memory: {memory}
""".format(memory=memory)

@mcp.resource("file://memory-state")
def memory_state() -> str:
    """
    Get current memory state.
    
    Returns:
        Current memory value
    """
    return f"Calculator Memory: {memory}"

if __name__ == "__main__":
    # Run the server
    mcp.run()