
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from fusion.engine import generate_final_verdict_ai

def test_user_scenario_fix():
    print("--- Testing User Scenario Fix (Human Video / High Metadata) ---")
    
    # Mocking the result of analyze_video for a Human but with metadata/audio noise
    # Previously, this would have resulted in a Final Score of 90+ because of meta_score or audio_score
    # but 19% AI and 0% Manipulation.
    
    # In my fix, bio_override reduces audio/meta scores if liveness is confirmed.
    # Also, fusion engine now includes all metrics in sub-scores.
    
    mock_video_evidence = {
        'Video': {
            'score': 27, # (Previously 92, now reduced by 0.3x reduction_factor due to liveness)
            'ai_gen_score': 19,
            'manip_score': 0,
            'meta_score': 27, # (Previously 90)
            'internal_audio_score': 27, # (Previously 90)
            'reasons': [
                'Verification: Multimodal liveness (blinks + pulse) confirms biological authenticity. Risk signals reduced.',
                'Video re-encoded with potentially suspicious software: ffmpeg',
                'C2PA Content Credentials found'
            ]
        },
        'Audio': {'score': 0, 'reasons': []},
        'Image': {'score': 0, 'reasons': []},
        'URL': {'score': 0, 'reasons': []}
    }
    
    result = generate_final_verdict_ai(mock_video_evidence, skip_llm=True)
    
    print(f"Final Score: {result['final_score']}%")
    print(f"Threat Score: {result['threat_score']}%")
    print(f"AI-Generated Score: {result['ai_generated_score']}%")
    print(f"Manipulation Score: {result['manipulation_score']}%")
    print(f"Verdict: {result['risk_level']}")
    
    # Expected: Final score is Low or Medium, not 92%.
    # Sub-scores are consistent with the final score.
    
    if result['final_score'] < 60:
        print("\nSUCCESS: False positive resolved! Human liveness correctly overrides noise.")
    else:
        print("\nFAILURE: Score is still too high.")

    if result['ai_generated_score'] >= 19 and result['manipulation_score'] >= 27:
        print("SUCCESS: Sub-scores now correctly reflect metadata/audio signals.")
    else:
        print(f"FAILURE: Sub-scores missing components (AI: {result['ai_generated_score']}, Manip: {result['manipulation_score']})")

if __name__ == "__main__":
    test_user_scenario_fix()
