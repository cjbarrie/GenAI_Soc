"""
Example 3: Robust JSON Extraction (without native JSON mode)
Shows how to reliably extract JSON from models that don't support JSON mode
"""

import json
import re
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("=" * 60)
print("ROBUST JSON EXTRACTION: Best practices")
print("=" * 60)

# Clear instructions for JSON-only output
INSTRUCTIONS = (
    'Return only a JSON object like this:\n'
    '{"stance":"Progressive|Conservative|Centrist|null",'
    '"confidence":0-1,"reasoning":"brief"}\n'
    'Do not add any extra text.'
)

def get_labels(client, text, model="gpt-4", max_retries=1):
    """
    Robust JSON extraction with error handling and retry logic

    Args:
        client: OpenAI client
        text: Text to analyze
        model: Model name
        max_retries: Number of retry attempts on parse failure

    Returns:
        dict: Parsed JSON response
    """
    # 1) Ask for JSON only with low temperature
    prompt = f'{INSTRUCTIONS}\n\nText: "{text}"'

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1  # Low temp for consistency
    )

    output = response.choices[0].message.content.strip()

    # 2) Try to parse as JSON
    try:
        return json.loads(output)
    except json.JSONDecodeError as e:
        print(f"⚠ Parse failed on first attempt: {e}")
        print(f"Raw output: {output}\n")

        if max_retries > 0:
            # 3) One retry asking for just JSON again
            fix_prompt = (
                "That was not valid JSON. Please send ONLY the JSON object, "
                "nothing else. No explanations, no markdown fences."
            )

            retry_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": output},
                    {"role": "user", "content": fix_prompt}
                ],
                temperature=0.0  # Zero temp for retry
            )

            retry_output = retry_response.choices[0].message.content.strip()
            print(f"Retry output: {retry_output}\n")

            try:
                return json.loads(retry_output)
            except json.JSONDecodeError as e:
                print(f"✗ Parse failed after retry: {e}")
                raise
        else:
            raise


def extract_json_with_fallbacks(text):
    """
    Multiple fallback strategies for JSON extraction
    """
    # Try to find JSON in markdown code fences
    fence_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(fence_pattern, text, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Last resort: raise error
    raise ValueError("Could not extract valid JSON from response")


# Test cases
test_texts = [
    "We must expand Medicare to cover everyone",
    "Cut taxes and reduce government spending",
    "Maintain balanced approach to fiscal policy"
]

print("\nTesting robust extraction with retry logic:\n")

for i, text in enumerate(test_texts, 1):
    print(f"Test {i}: {text}")
    try:
        result = get_labels(client, text)
        print(f"✓ Success: {json.dumps(result, indent=2)}\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")

print("=" * 60)
print("BEST PRACTICES FOR JSON WITHOUT JSON MODE")
print("=" * 60)

best_practices = [
    "1. Use clear instructions: 'Return ONLY valid JSON'",
    "2. Show exact schema in prompt",
    "3. Set low temperature (≤0.2)",
    "4. Use sentinels or fences for parsing",
    "5. Allow null values instead of forced guesses",
    "6. Implement one retry with error feedback",
    "7. Test with edge cases and validate schema"
]

for practice in best_practices:
    print(f"  {practice}")

print("\n✓ But prefer native JSON mode when available!")
