import os
import sys
import json
import logging
from typing import Any

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from modules.video import analyze_video
from config import CFG

# Disable noise logging 
logging.getLogger("truesight.video").setLevel(logging.ERROR)

def test_samples():
    base_dir = "samples/fake videos"
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        return

    files = [f for f in os.listdir(base_dir) if f.endswith(".mp4")]
    print(f"--- Testing {len(files)} Fake Video Samples ---")
    
    results = []
    
    for filename in files:
        path = os.path.join(base_dir, filename)
        print(f"\nAnalyzing: {filename}...")
        try:
            res = analyze_video(path)
            score = res.get("score", 0)
            verdict = "HIGH" if score >= 60 else "MEDIUM" if score >= 30 else "LOW"
            
            print(f"  Score: {score}% | Verdict: {verdict}")
            print(f"  Confidence: {res.get('confidence', 0)*100:.0f}%")
            
            # Print top reasons (truncated)
            reasons = res.get("reasons", [])
            for r in reasons[:3]:
                print(f"  [!] {r}")
            if len(reasons) > 3:
                print(f"  ... and {len(reasons)-3} more indicators.")
                
            results.append({"file": filename, "score": score, "verdict": verdict})
        except Exception as e:
            print(f"  Error analyzing {filename}: {e}")

    print("\n--- Summary ---")
    high = sum(1 for r in results if r["verdict"] == "HIGH")
    medium = sum(1 for r in results if r["verdict"] == "MEDIUM")
    low = sum(1 for r in results if r["verdict"] == "LOW")
    print(f"Total: {len(results)} | High: {high} | Medium: {medium} | Low: {low}")

if __name__ == "__main__":
    test_samples()
