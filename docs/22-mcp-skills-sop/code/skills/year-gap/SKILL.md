---
name: year-gap
description: Compute how many years apart two companies were founded. Use when the user asks how much older one company is than another, or for the gap, difference or years between two founding dates.
---

# Year gap between two companies

## Steps

1. `read <person>` for each person mentioned, and note the company they work at.
2. `read <company>` for each distinct company and note its founding year.
3. `run scripts/gap.py <year 1> <year 2>`: prints the absolute difference in years.
4. Answer with the number only.

If both people work at the same company, the answer is 0; you still need to read the company once.
