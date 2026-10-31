"""Years between two founding years. Usage: python gap.py 1941 1935"""
import sys

a, b = (int(x) for x in sys.argv[1:3])
print(abs(a - b))
