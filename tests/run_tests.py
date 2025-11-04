#!/usr/bin/env python
"""
Run all tests for the recommender library.

Usage:
    python tests/run_tests.py                # Run all tests
    python tests/run_tests.py -v             # Verbose mode
    python tests/run_tests.py TestDataset    # Run specific test class
"""
import sys
import unittest
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all_tests(verbosity=2, pattern='test_*.py'):
    """
    Run all tests in the tests directory.
    
    Args:
        verbosity: Output verbosity level (0-2)
        pattern: Pattern to match test files
    
    Returns:
        Test result object
    """
    # Discover and run tests
    loader = unittest.TestLoader()
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(tests_dir, pattern=pattern)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def run_specific_test(test_name, verbosity=2):
    """
    Run a specific test class or method.
    
    Args:
        test_name: Name of test class or method
        verbosity: Output verbosity level
    
    Returns:
        Test result object
    """
    loader = unittest.TestLoader()
    
    # Try to load as module.class or module.class.method
    try:
        suite = loader.loadTestsFromName(test_name)
    except AttributeError:
        # Try to find it in test files
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        suite = loader.discover(tests_dir, pattern=f'*{test_name}*.py')
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run recommender library tests')
    parser.add_argument('test', nargs='?', default=None,
                      help='Specific test to run (optional)')
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Verbose output')
    parser.add_argument('-q', '--quiet', action='store_true',
                      help='Minimal output')
    parser.add_argument('--pattern', default='test_*.py',
                      help='Pattern for test files (default: test_*.py)')
    
    args = parser.parse_args()
    
    # Determine verbosity
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1
    
    # Run tests
    if args.test:
        print(f"Running specific test: {args.test}")
        result = run_specific_test(args.test, verbosity=verbosity)
    else:
        print("Running all tests...")
        result = run_all_tests(verbosity=verbosity, pattern=args.pattern)
    
    # Print summary
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print(f"✅ ALL TESTS PASSED ({result.testsRun} tests)")
        sys.exit(0)
    else:
        print(f"❌ SOME TESTS FAILED")
        print(f"   Tests run: {result.testsRun}")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        sys.exit(1)


if __name__ == '__main__':
    main()

