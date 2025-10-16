"""
Example 7: Validation and Logging for Replication
Shows best practices for reproducibility and validation
"""

import json
import os
import hashlib
from datetime import datetime
from openai import OpenAI
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("=" * 60)
print("REPLICATION & VALIDATION BEST PRACTICES")
print("=" * 60)

# ============================================================================
# 1. COMPREHENSIVE LOGGING
# ============================================================================

print("\n1. Comprehensive Logging\n")

def annotate_with_logging(text, model="gpt-4-0613", temperature=0, seed=42, log_file="annotations.jsonl"):
    """
    Annotate text with complete logging for reproducibility
    """
    prompt = f"""Analyze political stance: {text}

Return JSON with stance, confidence, reasoning."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a political analyst. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        seed=seed
    )

    # Parse result
    result = json.loads(response.choices[0].message.content)

    # Create comprehensive log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "text": text,
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "prompt": prompt,
        "response": result,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        },
        "finish_reason": response.choices[0].finish_reason
    }

    # Append to log file
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

    return result

text = "Expand social safety nets and increase minimum wage"
result = annotate_with_logging(text)
print(f"✓ Annotated: {text}")
print(f"  Stance: {result.get('stance')}")
print(f"  Log saved to annotations.jsonl")

# ============================================================================
# 2. MODEL FINGERPRINTING (DETECT DRIFT)
# ============================================================================

print("\n" + "=" * 60)
print("2. Model Fingerprinting (Detect API Drift)")
print("=" * 60 + "\n")

def model_fingerprint(model, test_prompts, temperature=0, seed=42):
    """
    Create fingerprint to detect if model behavior has changed

    Args:
        model: Model name
        test_prompts: List of test prompts
        temperature: Temperature setting
        seed: Random seed

    Returns:
        str: SHA256 hash of concatenated responses
    """
    responses = []

    for prompt in test_prompts:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            seed=seed
        )
        responses.append(response.choices[0].message.content)

    # Hash concatenated responses
    fingerprint = hashlib.sha256(
        "".join(responses).encode()
    ).hexdigest()

    return fingerprint

# Create test set for fingerprinting
test_prompts = [
    "Classify: 'Cut taxes for businesses' - Progressive/Conservative/Centrist",
    "Classify: 'Expand healthcare coverage' - Progressive/Conservative/Centrist",
    "Classify: 'Balanced budget amendment' - Progressive/Conservative/Centrist"
]

fingerprint = model_fingerprint("gpt-4-0613", test_prompts)
print(f"Model fingerprint: {fingerprint[:16]}...")
print("\n✓ Save this fingerprint and check periodically for drift")
print("✓ If fingerprint changes, model behavior has changed!")

# ============================================================================
# 3. VALIDATION STRATEGIES
# ============================================================================

print("\n" + "=" * 60)
print("3. Validation Strategies")
print("=" * 60 + "\n")

# Simulate human and LLM labels for validation
human_labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])  # 0=Prog, 1=Centrist, 2=Cons
llm_labels = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2, 0])

# 3a. Human-LLM Agreement
print("3a. Human-LLM Agreement (Cohen's Kappa)\n")

kappa = cohen_kappa_score(human_labels, llm_labels)
accuracy = accuracy_score(human_labels, llm_labels)

print(f"Cohen's κ: {kappa:.3f}")
print(f"Accuracy:  {accuracy:.3f}")

if kappa > 0.80:
    print("✓ Substantial agreement")
elif kappa > 0.60:
    print("⚠ Moderate agreement - consider refinement")
else:
    print("✗ Low agreement - significant issues")

# 3b. Test-Retest Reliability
print("\n3b. Test-Retest Reliability\n")

def classify_batch(texts, model="gpt-4-0613", temperature=0, seed=42):
    """Classify batch of texts with fixed settings"""
    labels = []
    for text in texts:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Classify stance: {text}"}],
            temperature=temperature,
            seed=seed
        )
        # Extract label (simplified)
        content = response.choices[0].message.content
        if "Progressive" in content:
            labels.append(0)
        elif "Conservative" in content:
            labels.append(2)
        else:
            labels.append(1)
    return labels

texts = [
    "Expand Medicare",
    "Cut business taxes",
    "Balanced approach"
]

# Note: Actual test-retest would run at different times
print("Test-retest: Run same annotations twice with identical settings")
print("✓ High reliability (κ > 0.90) indicates stable model behavior")
print("⚠ Low reliability suggests temperature/seed not working as expected")

# 3c. Confusion Matrix
print("\n3c. Confusion Matrix\n")

cm = confusion_matrix(human_labels, llm_labels)
print("Confusion Matrix (rows=human, cols=LLM):")
print("              Prog  Cent  Cons")
for i, label in enumerate(["Progressive", "Centrist", "Conservative"]):
    print(f"{label:12}  {cm[i]}")

# ============================================================================
# 4. PROMPTBOOK (DOCUMENTATION)
# ============================================================================

print("\n" + "=" * 60)
print("4. Promptbook (Documentation)")
print("=" * 60 + "\n")

promptbook = {
    "task": "political_stance_classification",
    "date_created": "2024-10-08",
    "version": "1.0",
    "models": [
        {
            "name": "gpt-4-0613",
            "type": "api",
            "provider": "openai",
            "temperature": 0,
            "seed": 42,
            "response_format": "json"
        }
    ],
    "prompt_template": "Analyze political stance: {text}\n\nReturn JSON with stance, confidence, reasoning.",
    "output_schema": {
        "stance": ["Progressive", "Conservative", "Centrist"],
        "confidence": "float (0-1)",
        "reasoning": "string"
    },
    "validation": {
        "method": "human_comparison",
        "sample_size": 200,
        "cohen_kappa": 0.78,
        "accuracy": 0.82,
        "validation_date": "2024-10-08"
    },
    "fingerprint": fingerprint,
    "notes": "Validated on US political tweets. Low confidence (<0.7) texts manually reviewed."
}

promptbook_file = "promptbook.json"
with open(promptbook_file, 'w') as f:
    json.dump(promptbook, f, indent=2)

print(f"✓ Promptbook saved to {promptbook_file}")
print("\nPromptbook includes:")
for key in promptbook.keys():
    print(f"  • {key}")

# ============================================================================
# 5. REPRODUCIBILITY CHECKLIST
# ============================================================================

print("\n" + "=" * 60)
print("5. Reproducibility Checklist")
print("=" * 60 + "\n")

checklist = [
    ("Pin model versions", "Use specific snapshots (gpt-4-0613, not gpt-4)"),
    ("Set temperature to 0", "For deterministic outputs"),
    ("Use seed parameter", "When supported by API"),
    ("Log everything", "Prompts, responses, settings, timestamps"),
    ("Create promptbook", "Document complete annotation pipeline"),
    ("Validate against humans", "Cohen's κ > 0.80 target"),
    ("Test-retest reliability", "Check consistency over time"),
    ("Model fingerprinting", "Detect API drift"),
    ("Share code & configs", "Enable exact replication"),
    ("Use open models when possible", "Fixed weights = perfect reproducibility")
]

for item, description in checklist:
    print(f"☐ {item:30} - {description}")

print("\n✓ Following these practices enables credible, replicable research")
