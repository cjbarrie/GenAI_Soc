"""
Example 2: Four Approaches to Structured Outputs
Demonstrates the progression from basic to most reliable structured output methods
"""

import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

text = "We must expand Medicare to cover everyone"

print("=" * 60)
print("APPROACH 1: Prompt-Only Formatting (basic)")
print("=" * 60)

def llm(prompt: str) -> str:
    """Generic LLM call wrapper"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

prompt = f"""
Extract fields as JSON and respond ONLY with valid JSON:
{{
  "stance": "Progressive/Conservative/Centrist",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}

Input: {text}
"""

try:
    data = json.loads(llm(prompt))
    print(f"✓ Successfully parsed JSON")
    print(json.dumps(data, indent=2))
except json.JSONDecodeError as e:
    print(f"✗ Failed to parse: {e}")
    print(f"Raw output: {llm(prompt)}")

print()

print("=" * 60)
print("APPROACH 2: Few-Shot with Schema + Examples (better)")
print("=" * 60)

text = "Cut taxes and reduce regulations"
prompt = """
You output ONLY valid JSON with keys: stance, confidence, reasoning.

Example:
Input: Expand healthcare access for all
Output: {{"stance":"Progressive","confidence":0.9,"reasoning":"Universal healthcare is progressive policy"}}

Example:
Input: Maintain current spending levels
Output: {{"stance":"Centrist","confidence":0.8,"reasoning":"Status quo signals moderate position"}}

Now do the same:
Input: {input}
Output:
""".format(input=text)

data = json.loads(llm(prompt))
print(f"✓ Successfully parsed JSON")
print(json.dumps(data, indent=2))
print()

print("=" * 60)
print("APPROACH 3: Provider JSON Mode / Schema (more reliable)")
print("=" * 60)

# Define schema for structured output
schema = {
  "type": "object",
  "properties": {
    "stance": {
      "type": "string",
      "enum": ["Progressive", "Conservative", "Centrist"]
    },
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "reasoning": {"type": "string"}
  },
  "required": ["stance", "confidence", "reasoning"],
  "additionalProperties": False
}

text = "Protect traditional family values"
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system",
         "content": "You are a political analyst. Return valid JSON only."},
        {"role": "user",
         "content": f"Analyze political stance: {text}"}
    ],
    response_format={"type": "json_object"},  # Force JSON mode
    temperature=0
)

data = json.loads(response.choices[0].message.content)
print(f"✓ JSON mode guarantees valid JSON")
print(json.dumps(data, indent=2))
print()

print("=" * 60)
print("APPROACH 4: Function/Tool Calling (most structured)")
print("=" * 60)

# Define function schema with typed arguments
tools = [{
  "type": "function",
  "function": {
    "name": "analyze_stance",
    "description": "Return structured political stance analysis",
    "parameters": {
      "type": "object",
      "properties": {
        "stance": {
          "type": "string",
          "enum": ["Progressive", "Conservative", "Centrist"]
        },
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"}
      },
      "required": ["stance", "confidence", "reasoning"]
    }
  }
}]

text = "Expand social safety nets"
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": f"Analyze: {text}"}],
    tools=tools,
    tool_choice="auto",
    temperature=0
)

# Extract structured arguments
call = response.choices[0].message.tool_calls[0]
args = json.loads(call.function.arguments)
print(f"✓ Function calling provides strongest guarantees")
print(f"Function called: {call.function.name}")
print(json.dumps(args, indent=2))
print()

print("=" * 60)
print("COMPARISON: Reliability & Flexibility")
print("=" * 60)

comparison = {
    "Prompt-only": {"reliability": "Low", "flexibility": "High", "support": "Universal"},
    "Few-shot": {"reliability": "Medium", "flexibility": "High", "support": "Universal"},
    "JSON mode": {"reliability": "High", "flexibility": "Medium", "support": "Most APIs"},
    "Function calling": {"reliability": "Highest", "flexibility": "Low", "support": "OpenAI, Anthropic, Google"}
}

for approach, metrics in comparison.items():
    print(f"{approach:20} | Reliability: {metrics['reliability']:8} | Flexibility: {metrics['flexibility']:8} | Support: {metrics['support']}")

print("\n✓ Recommendation: Start with JSON mode (Approach 3)")
