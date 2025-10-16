"""
Example 4: Batch Annotation with JSON Mode
Shows how to efficiently annotate multiple texts with structured outputs
"""

import json
import os
import pandas as pd
from openai import OpenAI
from datetime import datetime

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Sample corpus
texts = [
    "We need stronger borders and immigration control",
    "Healthcare is a human right for all",
    "Balance the budget through moderate tax reform",
    "Invest in renewable energy infrastructure",
    "Cut regulations on small businesses",
    "Expand access to affordable childcare",
    "Maintain current defense spending levels",
    "Protect voting rights and access",
    "Reduce corporate tax rates",
    "Fund public education and teacher salaries"
]

print("=" * 60)
print("BATCH ANNOTATION WITH JSON MODE")
print("=" * 60)

# Template for consistent prompting
JSON_TEMPLATE = """Analyze this political text: {text}

Return JSON with keys:
- stance (Progressive/Conservative/Centrist)
- confidence (0-1)
- reasoning (brief explanation)
- policy_domain (e.g., healthcare, economy, education)"""

def annotate_text(text, model="gpt-4", temperature=0):
    """
    Annotate a single text with structured output
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are a political analyst. Return valid JSON only."},
            {"role": "user",
             "content": JSON_TEMPLATE.format(text=text)}
        ],
        response_format={"type": "json_object"},
        temperature=temperature
    )

    return json.loads(response.choices[0].message.content)


def batch_annotate(texts, model="gpt-4", temperature=0, log_file=None):
    """
    Annotate multiple texts and optionally log results

    Args:
        texts: List of texts to annotate
        model: Model to use
        temperature: Temperature setting
        log_file: Optional path to save annotation log

    Returns:
        pandas.DataFrame with annotations
    """
    results = []

    print(f"\nAnnotating {len(texts)} texts with {model}...")
    print(f"Temperature: {temperature}\n")

    for i, text in enumerate(texts, 1):
        print(f"[{i}/{len(texts)}] Processing: {text[:50]}...")

        try:
            annotation = annotate_text(text, model=model, temperature=temperature)
            annotation['text'] = text
            annotation['model'] = model
            annotation['temperature'] = temperature
            annotation['timestamp'] = datetime.now().isoformat()
            annotation['success'] = True
            annotation['error'] = None

        except Exception as e:
            print(f"  ✗ Error: {e}")
            annotation = {
                'text': text,
                'model': model,
                'temperature': temperature,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e),
                'stance': None,
                'confidence': None,
                'reasoning': None,
                'policy_domain': None
            }

        results.append(annotation)

        # Optional: save incremental log
        if log_file:
            with open(log_file, 'a') as f:
                f.write(json.dumps(annotation) + '\n')

    df = pd.DataFrame(results)
    print(f"\n✓ Completed: {df['success'].sum()}/{len(df)} successful")

    return df


# Run batch annotation
df = batch_annotate(
    texts,
    model="gpt-4",
    temperature=0,
    log_file="annotations.jsonl"
)

print("\n" + "=" * 60)
print("ANNOTATION RESULTS")
print("=" * 60)

# Display results
print("\nSummary by stance:")
print(df['stance'].value_counts())

print("\nAverage confidence by stance:")
print(df.groupby('stance')['confidence'].mean().round(3))

print("\nSample annotations:")
print(df[['text', 'stance', 'confidence', 'policy_domain']].head(3).to_string(index=False))

print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save to CSV
csv_file = "annotations.csv"
df.to_csv(csv_file, index=False)
print(f"✓ Saved to {csv_file}")

# Save detailed JSON
json_file = "annotations_detailed.json"
with open(json_file, 'w') as f:
    json.dump(df.to_dict('records'), f, indent=2)
print(f"✓ Saved to {json_file}")

print("\n" + "=" * 60)
print("QUALITY CHECKS")
print("=" * 60)

# Check for low confidence predictions
low_confidence = df[df['confidence'] < 0.7]
print(f"\nLow confidence annotations (< 0.7): {len(low_confidence)}")
if len(low_confidence) > 0:
    print(low_confidence[['text', 'stance', 'confidence']].to_string(index=False))

# Check for null values
nulls = df[df['stance'].isna()]
print(f"\nMissing stance labels: {len(nulls)}")

print("\n✓ Batch annotation complete!")
