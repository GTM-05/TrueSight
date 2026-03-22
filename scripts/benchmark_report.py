import time
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from fusion.engine import generate_final_verdict_ai
from config import CFG
import ollama

def benchmark_report():
    print(f"--- Benchmarking Report Generation (Model: {CFG.LLM_VERDICT_MODEL}) ---")
    
    # Mock evidence
    evidence = {
        "Image": {"score": 72.5, "confidence": 0.85, "is_strong": True, "reasons": ["[ELA] High noise at (120,450)"]},
        "URL": {"score": 85.0, "confidence": 0.9, "is_strong": True, "reasons": ["[URL] Homograph detected: g0ogle.com"]}
    }
    
    print("Preloading model...")
    start_preload = time.time()
    from llm.llm import llm_preload_model
    success = llm_preload_model()
    end_preload = time.time()
    print(f"Preload success: {success} (Time: {end_preload - start_preload:.2f}s)")
    
    print("\nGenerating report with streaming...")
    start_time = time.time()
    verdict = generate_final_verdict_ai(evidence, skip_llm=False, stream=True)
    explanation_gen = verdict['ai_explanation']
    
    first_token_time = None
    full_text = ""
    
    try:
        for chunk in explanation_gen:
            if first_token_time is None:
                first_token_time = time.time()
                print(f"First chunk received in: {first_token_time - start_time:.2f}s")
            
            if isinstance(chunk, dict) and 'response' in chunk:
                full_text += chunk['response']
            else:
                full_text += str(chunk)
    except Exception as e:
        print(f"Streaming error: {e}")

    end_time = time.time()
    
    print(f"Total generation time: {end_time - start_time:.2f}s")
    print(f"Total characters: {len(full_text)}")
    print(f"Average speed: {len(full_text) / (end_time - start_time):.2f} chars/sec")
    print("\nSample Output (First 100 chars):")
    print(full_text[:100] + "...")

if __name__ == "__main__":
    benchmark_report()
