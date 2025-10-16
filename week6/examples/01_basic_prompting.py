"""
Example 1: Basic Prompting Patterns for Text Annotation
Shows simple f-string and template-based prompting approaches
"""

from openai import OpenAI
import os

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Sample political texts
texts = [
    "We must invest in renewable energy now!",
    "Cut taxes and reduce business regulations",
    "Healthcare is a human right for all citizens",
    "Maintain current spending levels and balanced budget"
]

print("=" * 60)
print("EXAMPLE 1A: Simple f-string prompting")
print("=" * 60)

# Simple f-string approach
text = texts[0]
prompt = f"""Classify the political stance of this text as:
- Progressive
- Conservative
- Centrist

Text: {text}
Stance:"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

print(f"Text: {text}")
print(f"Response: {response.choices[0].message.content}")
print()

print("=" * 60)
print("EXAMPLE 1B: Reusable template for batch processing")
print("=" * 60)

# Template approach for consistency
STANCE_TEMPLATE = """Classify the political stance of this text as:
- Progressive
- Conservative
- Centrist

Text: {text}
Stance:"""

results = []
for text in texts:
    prompt = STANCE_TEMPLATE.format(text=text)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    result = response.choices[0].message.content.strip()
    results.append({"text": text, "stance": result})
    print(f"Text: {text}")
    print(f"Stance: {result}\n")

print("=" * 60)
print("EXAMPLE 1C: Few-shot prompting")
print("=" * 60)

# Few-shot template with examples
FEW_SHOT_TEMPLATE = """Classify political stance as Progressive, Conservative, or Centrist.

Examples:
Text: "Cut taxes and reduce regulations" -> Conservative
Text: "Expand healthcare access for all" -> Progressive
Text: "Maintain current spending levels" -> Centrist

Text: {text} ->"""

text = "Protect traditional family values and limit government overreach"
prompt = FEW_SHOT_TEMPLATE.format(text=text)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

print(f"Text: {text}")
print(f"Response: {response.choices[0].message.content}")
print()

print("=" * 60)
print("EXAMPLE 1D: Chain-of-thought prompting")
print("=" * 60)

COT_TEMPLATE = """Classify the stance and explain your reasoning.

Text: {text}

Think step-by-step:
1. What policy domain is this?
2. What values does it express?
3. What stance does this suggest?

Reasoning:
Stance:"""

text = "Invest heavily in public education and teacher salaries"
prompt = COT_TEMPLATE.format(text=text)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)

print(f"Text: {text}")
print(f"Response:\n{response.choices[0].message.content}")
