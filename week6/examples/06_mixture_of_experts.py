"""
Example 6: Mixture of Experts (Multi-Model Ensemble)
Shows how to aggregate predictions across multiple models for robustness
Based on Kraft et al. (2024) approach
"""

import json
import os
import numpy as np
import pandas as pd
from openai import OpenAI

# Try to import ollama for local models
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠ Ollama not available - will use API models only\n")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("=" * 60)
print("MIXTURE OF EXPERTS: MULTI-MODEL ENSEMBLE")
print("=" * 60)

def get_stance_score(text, model="gpt-4", provider="openai"):
    """
    Get ideological position score from a model

    Args:
        text: Text to analyze
        model: Model name
        provider: "openai" or "ollama"

    Returns:
        float: Score from -1 (most progressive) to +1 (most conservative)
    """
    prompt = f"""Rate this text on ideology from -1 (most progressive)
to +1 (most conservative). Return only the number.

Text: {text}"""

    if provider == "openai":
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return float(response.choices[0].message.content.strip())

    elif provider == "ollama" and OLLAMA_AVAILABLE:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return float(response['message']['content'].strip())

    else:
        raise ValueError(f"Unknown provider: {provider}")


def ensemble_stance(text, model_configs):
    """
    Aggregate stance estimates across multiple models

    Args:
        text: Text to analyze
        model_configs: List of dicts with 'model' and 'provider' keys

    Returns:
        dict: Aggregated results with mean, median, std, individual scores
    """
    scores = []
    individual = {}

    for config in model_configs:
        model = config['model']
        provider = config['provider']

        try:
            score = get_stance_score(text, model=model, provider=provider)
            scores.append(score)
            individual[model] = score
            print(f"  {model:15} ({provider:7}): {score:+.3f}")
        except Exception as e:
            print(f"  {model:15} ({provider:7}): Error - {e}")
            continue

    if not scores:
        return None

    return {
        "mean": np.mean(scores),
        "median": np.median(scores),
        "std": np.std(scores),
        "min": np.min(scores),
        "max": np.max(scores),
        "individual": individual,
        "n_models": len(scores)
    }


print("\nExample 1: Single text with multiple models\n")

# Configure models to use
model_configs = [
    {"model": "gpt-4", "provider": "openai"},
    {"model": "gpt-3.5-turbo", "provider": "openai"}
]

# Add local models if available
if OLLAMA_AVAILABLE:
    model_configs.extend([
        {"model": "llama3", "provider": "ollama"},
        {"model": "mixtral", "provider": "ollama"}
    ])

text = "We must protect traditional family values and limit government overreach"
print(f"Text: {text}\n")
print("Individual model scores:")

result = ensemble_stance(text, model_configs)

if result:
    print(f"\nEnsemble results:")
    print(f"  Mean:      {result['mean']:+.3f}")
    print(f"  Median:    {result['median']:+.3f}")
    print(f"  Std dev:   {result['std']:.3f}")
    print(f"  Range:     [{result['min']:+.3f}, {result['max']:+.3f}]")
    print(f"  Agreement: {'High' if result['std'] < 0.3 else 'Medium' if result['std'] < 0.6 else 'Low'}")

print("\n" + "=" * 60)
print("BATCH ANALYSIS WITH ENSEMBLE")
print("=" * 60)

# Sample tweets/texts
tweets = [
    "Expand Medicare to cover everyone",
    "Cut taxes and regulations on businesses",
    "Protect voting rights and access",
    "Secure the border and enforce immigration laws",
    "Invest in public schools and teacher salaries"
]

results = []

print(f"\nAnalyzing {len(tweets)} texts with ensemble...\n")

for i, tweet in enumerate(tweets, 1):
    print(f"[{i}/{len(tweets)}] {tweet}")
    ensemble = ensemble_stance(tweet, model_configs)

    if ensemble:
        results.append({
            "text": tweet,
            "position": ensemble["mean"],
            "uncertainty": ensemble["std"],
            "n_models": ensemble["n_models"],
            **{f"model_{k}": v for k, v in ensemble["individual"].items()}
        })
    print()

df = pd.DataFrame(results)

print("=" * 60)
print("ENSEMBLE RESULTS")
print("=" * 60)

print("\nPosition scores (negative = progressive, positive = conservative):")
print(df[['text', 'position', 'uncertainty']].to_string(index=False))

print("\n" + "=" * 60)
print("WHEN MIXTURE OF EXPERTS WORKS BEST")
print("=" * 60)

use_cases = [
    "✓ Short texts: Tweets, manifestos, speeches",
    "✓ Cross-lingual: Works across 10+ languages",
    "✓ Policy dimensions: Clear ideological spectra",
    "✓ Validation: Can correlate with expert coding",
    "✓ Robustness: Reduces impact of single model quirks",
    "✓ Uncertainty: Std dev indicates agreement level"
]

for use_case in use_cases:
    print(f"  {use_case}")

print("\n" + "=" * 60)
print("KEY INSIGHTS FROM KRAFT ET AL. (2024)")
print("=" * 60)

insights = [
    "• Correlations with expert benchmarks > 0.90",
    "• Works across GPT-4, Llama 3, MiXtral, Aya",
    "• Averaging reduces model-specific biases",
    "• Cost-efficient compared to human coding",
    "• Fast: Can process thousands of texts per day",
    "• Reliable across languages and text types"
]

for insight in insights:
    print(f"  {insight}")

print("\n✓ Ensemble approach provides robust ideological scaling")
