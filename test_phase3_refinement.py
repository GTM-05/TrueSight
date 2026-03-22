
import sys
import os
import numpy as np

# Add the project root to sys.path
sys.path.append(os.getcwd())

from fusion.engine import generate_final_verdict_ai

def test_fusion_logic():
    print("🧪 Testing Phase 3 Fusion Logic...")
    
    # Case 1: Simple Weighted Average (Low risk)
    evidence_low = {
        'Image': {'score': 10, 'confidence': 0.6, 'is_strong': False, 'reasons': ['Clean']},
        'Audio': {'score': 5, 'confidence': 0.7, 'is_strong': False, 'reasons': ['Clean']},
        'Video': {'score': 8, 'confidence': 0.5, 'is_strong': False, 'reasons': ['Clean']}
    }
    res1 = generate_final_verdict_ai(evidence_low, skip_llm=True)
    print(f"CASE 1 (Low): Score={res1['final_score']} (Expected < 19 due to safety floor)")
    assert res1['final_score'] <= 19
    
    # Case 2: Cross-Modal Disagreement (Red Flag)
    evidence_disagree = {
        'Image': {'score': 10, 'confidence': 0.8, 'is_strong': False, 'reasons': ['Natural Texture']},
        'Audio': {'score': 65, 'confidence': 0.9, 'is_strong': True, 'reasons': ['Monotone Pitch', 'Jitter Anomaly']},
        'Video': {'score': 5, 'confidence': 0.9, 'is_strong': False, 'reasons': ['Liveness OK']}
    }
    res2 = generate_final_verdict_ai(evidence_disagree, skip_llm=True)
    print(f"CASE 2 (Disagree): Score={res2['final_score']} (Expected > 60 due to disagreement bonus)")
    assert res2['final_score'] >= 60

    # Case 3: High Confidence Anchor
    evidence_strong = {
        'Image': {
            'score': 20, 
            'confidence': 0.9, 
            'is_strong': True, 
            'reasons': ['Strong Spectral Slope Anomaly'],
            'metrics': {'spectral_slope': -1.2} # This will trigger the Elite Override in engine.py
        },
        'Audio': {'score': 10, 'confidence': 0.5, 'is_strong': False}
    }
    res3 = generate_final_verdict_ai(evidence_strong, skip_llm=True)
    print(f"CASE 3 (Strong Anchor): Score={res3['final_score']} (Expected >= 75 due to Elite Override)")
    assert res3['final_score'] >= 75

    # Case 4: Weighted Average with Confidence
    evidence_conf = {
        'Image': {'score': 80, 'confidence': 0.2, 'is_strong': False}, # Very shaky signal
        'Audio': {'score': 10, 'confidence': 0.9, 'is_strong': False}, # Strong clean signal
        'Video': {'score': 10, 'confidence': 0.9, 'is_strong': False}
    }
    res4 = generate_final_verdict_ai(evidence_conf, skip_llm=True)
    print(f"CASE 4 (Low Confidence Noise): Score={res4['final_score']} (Expected low score despite 80 in Image)")
    assert res4['final_score'] < 30

    print("✅ All Fusion Logic tests passed!")

if __name__ == "__main__":
    try:
        test_fusion_logic()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
