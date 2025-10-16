"""
Example 5: Local Annotation with Ollama
Shows how to run annotations locally using open-source models
"""

import json
import pandas as pd

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠ Ollama not installed. Install with: pip install ollama")
    print("⚠ Also requires Ollama running locally: https://ollama.ai\n")

if OLLAMA_AVAILABLE:
    print("=" * 60)
    print("LOCAL ANNOTATION WITH OLLAMA")
    print("=" * 60)

    # Sample texts
    texts = [
        "We need healthcare reform",
        "Cut taxes and regulations",
        "Protect the environment",
        "Invest in renewable energy",
        "Maintain current policies"
    ]

    def analyze_text_ollama(text, model='llama3'):
        """
        Analyze text using local Ollama model

        Args:
            text: Text to analyze
            model: Ollama model name (e.g., 'llama3', 'mixtral', 'phi3')

        Returns:
            dict: Parsed JSON response
        """
        response = ollama.chat(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a political analyst. Return valid JSON only.'
                },
                {
                    'role': 'user',
                    'content': f"""Analyze this text: {text}

Return JSON with stance, confidence, reasoning."""
                }
            ],
            format='json'  # Force JSON output
        )

        return json.loads(response['message']['content'])

    print("\nExample 1: Single text annotation\n")

    text = texts[0]
    print(f"Text: {text}")

    try:
        result = analyze_text_ollama(text)
        print(f"✓ Result: {json.dumps(result, indent=2)}\n")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure Ollama is running and llama3 is pulled:")
        print("  ollama pull llama3\n")

    print("=" * 60)
    print("BATCH PROCESSING WITH OLLAMA")
    print("=" * 60)

    def batch_annotate_ollama(texts, model='llama3'):
        """Batch annotate texts with local model"""
        results = []

        print(f"\nAnnotating {len(texts)} texts with {model}...\n")

        for i, text in enumerate(texts, 1):
            print(f"[{i}/{len(texts)}] {text[:50]}...")

            try:
                result = analyze_text_ollama(text, model=model)
                result['text'] = text
                result['model'] = model
                results.append(result)
                print(f"  ✓ {result.get('stance', 'N/A')}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append({
                    'text': text,
                    'model': model,
                    'error': str(e)
                })

        return pd.DataFrame(results)

    try:
        df = batch_annotate_ollama(texts)

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(df[['text', 'stance', 'confidence']].to_string(index=False))

    except Exception as e:
        print(f"\n✗ Batch processing failed: {e}")

    print("\n" + "=" * 60)
    print("COMPARING DIFFERENT LOCAL MODELS")
    print("=" * 60)

    # Compare different models
    models = ['llama3', 'mixtral', 'phi3']
    text = "Invest in renewable energy infrastructure"

    print(f"\nText: {text}\n")

    for model in models:
        try:
            response = ollama.chat(
                model=model,
                messages=[{
                    'role': 'user',
                    'content': f'Analyze political stance: {text}'
                }],
                format='json'
            )
            result = json.loads(response['message']['content'])
            stance = result.get('stance', 'N/A')
            confidence = result.get('confidence', 'N/A')
            print(f"{model:10} | Stance: {stance:12} | Confidence: {confidence}")
        except Exception as e:
            print(f"{model:10} | Error: {e}")

    print("\n" + "=" * 60)
    print("LOCAL VS API: TRADEOFFS")
    print("=" * 60)

    tradeoffs = {
        "Cost": {"Local": "Free (after setup)", "API": "Per-token fees"},
        "Privacy": {"Local": "Data stays local", "API": "Sent to provider"},
        "Speed": {"Local": "Depends on hardware", "API": "Fast, optimized"},
        "Reproducibility": {"Local": "Fixed model weights", "API": "Version drift"},
        "Setup": {"Local": "Install + download models", "API": "Just API key"},
        "Quality": {"Local": "Varies by model", "API": "Generally higher"}
    }

    print()
    for aspect, comparison in tradeoffs.items():
        print(f"{aspect:15} | Local: {comparison['Local']:25} | API: {comparison['API']}")

    print("\n✓ Local models excellent for: privacy, cost control, reproducibility")
    print("✓ API models excellent for: quick prototyping, highest accuracy")

else:
    print("\nTo use this example:")
    print("1. Install Ollama: https://ollama.ai")
    print("2. Pull a model: ollama pull llama3")
    print("3. Install Python client: pip install ollama")
    print("4. Run this script again")
