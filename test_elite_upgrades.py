import sys
import os

# Ensure local modules are importable
sys.path.append(os.getcwd())

from fusion.engine import generate_final_verdict_ai

def test_elite_fusion():
    print("🧪 Testing Elite Forensic Fusion Signals...")
    
    # Test Case 1: Spectral Slope Anomaly (Image)
    evidence_slope = {
        'Image': {
            'score': 20, # Low base score
            'metrics': {'spectral_slope': -1.5} # Significant deviation from -2.2
        }
    }
    res1 = generate_final_verdict_ai(evidence_slope, skip_llm=True)
    print(f"1. Spectral Slope Anomaly (-1.5) -> Final Score: {res1['final_score']}% (Expected: >= 75%)")
    assert res1['final_score'] >= 75
    
    # Test Case 2: Chromatic Alignment Anomaly
    evidence_chrom = {
        'Image': {
            'score': 25,
            'metrics': {'chrom_score': 1.2} # Too perfectly aligned
        }
    }
    res2 = generate_final_verdict_ai(evidence_chrom, skip_llm=True)
    print(f"2. Chromatic Alignment Anomaly (1.2) -> Final Score: {res2['final_score']}% (Expected: >= 65%)")
    assert res2['final_score'] >= 65
    
    # Test Case 3: Iris Jitter (Video)
    evidence_iris = {
        'Video': {
            'score': 19, # Safety floor level
            'metrics': {'iris_jitter_anomaly': True}
        }
    }
    res3 = generate_final_verdict_ai(evidence_iris, skip_llm=True)
    print(f"3. Iris Jitter Anomaly (True) -> Final Score: {res3['final_score']}% (Expected: >= 70%)")
    assert res3['final_score'] >= 70

    print("✅ All Elite Fusion Tests Passed!")

if __name__ == "__main__":
    test_elite_fusion()
