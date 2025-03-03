"""
Symbolic math engine for processing and solving equations.
"""

import re
from typing import Dict, List, Optional, Union, Tuple
import math
import numpy as np
from dataclasses import dataclass
import logging

@dataclass
class Token:
    type: str
    value: str
    position: int

class MathEngine:
    def __init__(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Define token patterns
        self.patterns = {
            'number': r'\d+(\.\d+)?',
            'variable': r'[a-zA-Z_][a-zA-Z0-9_]*',
            'operator': r'[\+\-\*/^]',
            'function': r'sin|cos|tan|log|exp|sqrt',
            'parenthesis': r'[\(\)]',
            'whitespace': r'\s+'
        }
        
        # Define operator precedence
        self.precedence = {
            '+': 1,
            '-': 1,
            '*': 2,
            '/': 2,
            '^': 3
        }
    
    def tokenize(self, expression: str) -> List[Token]:
        """Convert mathematical expression into tokens."""
        tokens = []
        position = 0
        
        while position < len(expression):
            # Skip whitespace
            if expression[position].isspace():
                position += 1
                continue
            
            # Try to match each pattern
            matched = False
            for token_type, pattern in self.patterns.items():
                match = re.match(pattern, expression[position:])
                if match:
                    value = match.group(0)
                    tokens.append(Token(token_type, value, position))
                    position += len(value)
                    matched = True
                    break
            
            if not matched:
                raise ValueError(f"Invalid character at position {position}: {expression[position]}")
        
        return tokens
    
    def infix_to_postfix(self, tokens: List[Token]) -> List[Token]:
        """Convert infix notation to postfix (RPN) notation."""
        output = []
        operators = []
        
        for token in tokens:
            if token.type == 'number' or token.type == 'variable':
                output.append(token)
            elif token.type == 'function':
                operators.append(token)
            elif token.type == 'operator':
                while (operators and operators[-1].type == 'operator' and
                       self.precedence[operators[-1].value] >= self.precedence[token.value]):
                    output.append(operators.pop())
                operators.append(token)
            elif token.value == '(':
                operators.append(token)
            elif token.value == ')':
                while operators and operators[-1].value != '(':
                    output.append(operators.pop())
                if operators and operators[-1].value == '(':
                    operators.pop()
        
        while operators:
            output.append(operators.pop())
        
        return output
    
    def evaluate_postfix(self, tokens: List[Token], variables: Optional[Dict[str, float]] = None) -> float:
        """Evaluate postfix expression."""
        if variables is None:
            variables = {}
        
        stack = []
        
        for token in tokens:
            if token.type == 'number':
                stack.append(float(token.value))
            elif token.type == 'variable':
                if token.value in variables:
                    stack.append(variables[token.value])
                else:
                    raise ValueError(f"Undefined variable: {token.value}")
            elif token.type == 'operator':
                if len(stack) < 2:
                    raise ValueError("Invalid expression: insufficient operands")
                b = stack.pop()
                a = stack.pop()
                
                if token.value == '+':
                    stack.append(a + b)
                elif token.value == '-':
                    stack.append(a - b)
                elif token.value == '*':
                    stack.append(a * b)
                elif token.value == '/':
                    if b == 0:
                        raise ValueError("Division by zero")
                    stack.append(a / b)
                elif token.value == '^':
                    stack.append(a ** b)
            elif token.type == 'function':
                if not stack:
                    raise ValueError("Invalid expression: insufficient operands")
                x = stack.pop()
                
                if token.value == 'sin':
                    stack.append(math.sin(x))
                elif token.value == 'cos':
                    stack.append(math.cos(x))
                elif token.value == 'tan':
                    stack.append(math.tan(x))
                elif token.value == 'log':
                    if x <= 0:
                        raise ValueError("Invalid argument for logarithm")
                    stack.append(math.log(x))
                elif token.value == 'exp':
                    stack.append(math.exp(x))
                elif token.value == 'sqrt':
                    if x < 0:
                        raise ValueError("Invalid argument for square root")
                    stack.append(math.sqrt(x))
        
        if len(stack) != 1:
            raise ValueError("Invalid expression: too many operands")
        
        return stack[0]
    
    def solve_equation(self, equation: str, variable: str) -> List[float]:
        """
        Solve an equation for a given variable.
        
        Args:
            equation: Mathematical equation (e.g., "2x + 3 = 7")
            variable: Variable to solve for
            
        Returns:
            List of solutions
        """
        # Split equation into left and right sides
        left, right = equation.split('=')
        
        # Convert to standard form (move everything to left side)
        left_tokens = self.tokenize(left.strip())
        right_tokens = self.tokenize(right.strip())
        
        # Evaluate right side
        right_value = self.evaluate_postfix(self.infix_to_postfix(right_tokens))
        
        # Move right value to left side
        left_tokens.extend([
            Token('operator', '-', 0),
            Token('number', str(right_value), 0)
        ])
        
        # Convert to postfix
        postfix = self.infix_to_postfix(left_tokens)
        
        # Collect coefficients
        coefficients = self._collect_coefficients(postfix, variable)
        
        # Solve linear equation
        if len(coefficients) == 2:  # Linear equation
            a, b = coefficients
            if a == 0:
                if b == 0:
                    return [float('inf')]  # Infinite solutions
                return []  # No solutions
            return [-b / a]
        
        # For higher-degree equations, use numerical methods
        return self._solve_numerical(postfix, variable)
    
    def _collect_coefficients(self, tokens: List[Token], variable: str) -> List[float]:
        """Collect coefficients of terms in the equation."""
        coefficients = [0, 0]  # [coefficient of x, constant term]
        
        for i, token in enumerate(tokens):
            if token.type == 'variable' and token.value == variable:
                if i > 0 and tokens[i-1].type == 'number':
                    coefficients[0] += float(tokens[i-1].value)
                else:
                    coefficients[0] += 1
            elif token.type == 'number':
                if i == len(tokens) - 1 or tokens[i+1].type != 'variable':
                    coefficients[1] += float(token.value)
        
        return coefficients
    
    def _solve_numerical(self, tokens: List[Token], variable: str) -> List[float]:
        """Solve equation using numerical methods."""
        def f(x: float) -> float:
            variables = {variable: x}
            return self.evaluate_postfix(tokens, variables)
        
        # Use Newton's method
        solutions = []
        x = 0.0  # Initial guess
        
        for _ in range(100):  # Maximum iterations
            try:
                # Compute function value and derivative
                h = 1e-6
                fx = f(x)
                fpx = (f(x + h) - f(x)) / h
                
                if abs(fx) < 1e-10:
                    solutions.append(x)
                    break
                
                if abs(fpx) < 1e-10:
                    break
                
                x = x - fx / fpx
                
            except Exception:
                break
        
        return solutions
    
    def simplify_expression(self, expression: str) -> str:
        """Simplify a mathematical expression."""
        tokens = self.tokenize(expression)
        postfix = self.infix_to_postfix(tokens)
        
        # Convert back to infix with simplified form
        stack = []
        for token in postfix:
            if token.type in ['number', 'variable']:
                stack.append(token.value)
            elif token.type == 'operator':
                b = stack.pop()
                a = stack.pop()
                stack.append(f"({a}{token.value}{b})")
            elif token.type == 'function':
                x = stack.pop()
                stack.append(f"{token.value}({x})")
        
        return stack[0]
    
    def differentiate(self, expression: str, variable: str) -> str:
        """Compute the derivative of an expression with respect to a variable."""
        tokens = self.tokenize(expression)
        postfix = self.infix_to_postfix(tokens)
        
        # Basic differentiation rules
        def diff_term(term: str) -> str:
            if term == variable:
                return "1"
            elif term.isdigit():
                return "0"
            elif term.startswith(variable + "^"):
                power = int(term.split("^")[1])
                if power == 1:
                    return "1"
                return f"{power}{variable}^{power-1}"
            return "0"
        
        # Convert postfix to infix and apply differentiation rules
        stack = []
        for token in postfix:
            if token.type in ['number', 'variable']:
                stack.append(diff_term(token.value))
            elif token.type == 'operator':
                b = stack.pop()
                a = stack.pop()
                if token.value == '+':
                    stack.append(f"{a}+{b}")
                elif token.value == '-':
                    stack.append(f"{a}-{b}")
                elif token.value == '*':
                    stack.append(f"{a}*{b}+{b}*{a}")
                elif token.value == '/':
                    stack.append(f"({a}*{b}-{b}*{a})/({b}^2)")
                elif token.value == '^':
                    stack.append(f"{b}*{a}^{b-1}*{a}")
        
        return stack[0]
    
    def integrate(self, expression: str, variable: str) -> str:
        """Compute the indefinite integral of an expression with respect to a variable."""
        tokens = self.tokenize(expression)
        postfix = self.infix_to_postfix(tokens)
        
        # Basic integration rules
        def integrate_term(term: str) -> str:
            if term == variable:
                return f"{variable}^2/2"
            elif term.isdigit():
                return f"{term}{variable}"
            elif term.startswith(variable + "^"):
                power = int(term.split("^")[1])
                return f"{variable}^{power+1}/{power+1}"
            return f"{term}{variable}"
        
        # Convert postfix to infix and apply integration rules
        stack = []
        for token in postfix:
            if token.type in ['number', 'variable']:
                stack.append(integrate_term(token.value))
            elif token.type == 'operator':
                b = stack.pop()
                a = stack.pop()
                if token.value == '+':
                    stack.append(f"{a}+{b}")
                elif token.value == '-':
                    stack.append(f"{a}-{b}")
                elif token.value == '*':
                    stack.append(f"{a}*{b}")
                elif token.value == '/':
                    stack.append(f"{a}/{b}")
                elif token.value == '^':
                    stack.append(f"{a}^{b}")
        
        return stack[0] + "+C"  # Add constant of integration 