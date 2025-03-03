"""
Safe code execution tool with sandboxing and security measures.
"""

import subprocess
import tempfile
import os
import sys
import ast
import logging
from typing import Dict, List, Optional, Tuple
import signal
import resource
import threading
import queue
import time

class CodeExecutor:
    def __init__(self, timeout: int = 5, max_memory: int = 100 * 1024 * 1024):  # 100MB
        self.timeout = timeout
        self.max_memory = max_memory
        self.allowed_modules = {
            'math', 'random', 'datetime', 'json', 're', 'string',
            'collections', 'itertools', 'functools', 'operator'
        }
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def execute(self, code: str, input_data: Optional[str] = None) -> Dict:
        """
        Execute code in a safe environment.
        
        Args:
            code: Python code to execute
            input_data: Optional input data for the program
            
        Returns:
            Dictionary containing execution results
        """
        # Validate code
        if not self._validate_code(code):
            return {
                'success': False,
                'error': 'Invalid or potentially dangerous code detected'
            }
        
        # Create temporary directory for execution
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write code to file
            code_path = os.path.join(temp_dir, 'code.py')
            with open(code_path, 'w') as f:
                f.write(code)
            
            # Create input file if needed
            input_path = None
            if input_data:
                input_path = os.path.join(temp_dir, 'input.txt')
                with open(input_path, 'w') as f:
                    f.write(input_data)
            
            # Execute code
            return self._run_code(code_path, input_path)
    
    def _validate_code(self, code: str) -> bool:
        """Validate code for potential security risks."""
        try:
            # Parse code into AST
            tree = ast.parse(code)
            
            # Check for dangerous operations
            for node in ast.walk(tree):
                # Check for imports
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for name in node.names:
                        if name.name.split('.')[0] not in self.allowed_modules:
                            self.logger.warning(f"Attempted to import restricted module: {name.name}")
                            return False
                
                # Check for file operations
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in {'open', 'file', 'exec', 'eval', 'compile'}:
                            self.logger.warning(f"Attempted to use restricted function: {node.func.id}")
                            return False
                
                # Check for system commands
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in {'system', 'popen', 'spawn'}:
                            self.logger.warning(f"Attempted to use restricted system call: {node.func.attr}")
                            return False
            
            return True
            
        except SyntaxError:
            self.logger.warning("Invalid Python syntax")
            return False
    
    def _run_code(self, code_path: str, input_path: Optional[str] = None) -> Dict:
        """Run code in a subprocess with resource limits."""
        output_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def run_process():
            try:
                # Set up process with resource limits
                process = subprocess.Popen(
                    [sys.executable, code_path],
                    stdin=subprocess.PIPE if input_path else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Set memory limit
                resource.setrlimit(resource.RLIMIT_AS, (self.max_memory, self.max_memory))
                
                # Handle input if provided
                if input_path:
                    with open(input_path, 'r') as f:
                        input_data = f.read()
                    stdout, stderr = process.communicate(input=input_data)
                else:
                    stdout, stderr = process.communicate()
                
                output_queue.put((stdout, stderr))
                
            except Exception as e:
                error_queue.put(str(e))
        
        # Start execution thread
        thread = threading.Thread(target=run_process)
        thread.start()
        
        try:
            # Wait for completion or timeout
            thread.join(timeout=self.timeout)
            
            if thread.is_alive():
                # Kill process if timeout
                thread._stop()
                return {
                    'success': False,
                    'error': f'Execution timed out after {self.timeout} seconds'
                }
            
            # Check for errors
            if not error_queue.empty():
                return {
                    'success': False,
                    'error': error_queue.get()
                }
            
            # Get output
            stdout, stderr = output_queue.get()
            
            return {
                'success': True,
                'output': stdout,
                'error': stderr
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def execute_with_tests(self, code: str, test_cases: List[Dict]) -> Dict:
        """
        Execute code with multiple test cases.
        
        Args:
            code: Python code to execute
            test_cases: List of test cases with input and expected output
            
        Returns:
            Dictionary containing test results
        """
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            input_data = test_case.get('input', '')
            expected_output = test_case.get('expected_output', '')
            
            # Execute code with test input
            result = self.execute(code, input_data)
            
            # Compare output with expected
            if result['success']:
                actual_output = result['output'].strip()
                passed = actual_output == expected_output.strip()
            else:
                passed = False
            
            results.append({
                'test_case': i,
                'passed': passed,
                'input': input_data,
                'expected_output': expected_output,
                'actual_output': result.get('output', ''),
                'error': result.get('error', '')
            })
        
        # Calculate summary
        total_tests = len(test_cases)
        passed_tests = sum(1 for r in results if r['passed'])
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests
            },
            'results': results
        }
    
    def analyze_code(self, code: str) -> Dict:
        """
        Analyze code for potential issues and provide suggestions.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            tree = ast.parse(code)
            
            analysis = {
                'complexity': self._calculate_complexity(tree),
                'suggestions': [],
                'warnings': [],
                'statistics': {
                    'lines': len(code.splitlines()),
                    'functions': 0,
                    'classes': 0,
                    'imports': 0
                }
            }
            
            # Analyze AST
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis['statistics']['functions'] += 1
                elif isinstance(node, ast.ClassDef):
                    analysis['statistics']['classes'] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    analysis['statistics']['imports'] += 1
                
                # Check for common issues
                if isinstance(node, ast.While):
                    analysis['warnings'].append('While loops can lead to infinite execution')
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id == 'print':
                            analysis['suggestions'].append('Consider using logging instead of print statements')
            
            return analysis
            
        except SyntaxError:
            return {
                'error': 'Invalid Python syntax',
                'suggestions': ['Fix syntax errors before analysis']
            }
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate code complexity using cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
        
        return complexity 