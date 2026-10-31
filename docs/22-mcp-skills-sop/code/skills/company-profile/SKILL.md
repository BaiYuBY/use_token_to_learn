---
name: company-profile
description: Write a short profile of the company a person works for, with its headquarters city and age. Use when the user asks about someone's employer, workplace or company, or wants a company summary or profile.
license: MIT
metadata:
  author: use_token_to_learn
  version: "1.0"
---

# Company profile

Write a three-sentence profile of the company a person works for, from the documents only.

## Steps

1. `read <person>`: find the line "<person> works at <company>."
2. `read <company>`: find the founding year and the headquarters city.
3. `run scripts/company_age.py <founding year>`: the company's age in 2026. Use the script, do not
   work it out in your head.
4. Fill in [the template](references/template.md) and return it as the final answer.

## Rules

- Use only facts from the documents you read. If a fact is missing, write "unknown"; never guess.
- Cite each document you used as [Title].
- Keep the person's and the company's names exactly as written in the documents.
