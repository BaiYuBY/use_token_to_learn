"""Age of a company in 2026. Usage: python company_age.py 1941"""
import sys

year = int(sys.argv[1])
if not 1800 <= year <= 2026:
    sys.exit(f"implausible founding year: {year}")
print(2026 - year)
