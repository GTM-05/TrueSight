
import os
import cv2
import numpy as np
from modules.video import _get_face_crop, analyze_video
from fusion.engine import generate_final_verdict_ai

def test_face_crop():
    print("--- Testing Face Crop ---")
    video_path = "deepfake_sample.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("test_frame.jpg", frame)
        crop_path = _get_face_crop("test_frame.jpg")
        if crop_path != "test_frame.jpg":
            print(f"SUCCESS: Face detected and cropped to {crop_path}")
            os.remove(crop_path)
        else:
            print("FAILURE: No face detected in the first frame.")
        os.remove("test_frame.jpg")
    cap.release()

def test_fusion_logic():
    print("\n--- Testing Fusion Logic ---")
    # Mock evidence similar to user's case (High Video, Low others)
    mock_evidence = {
        'Video': {'score': 68, 'ai_gen_score': 68, 'manip_score': 48, 'reasons': ['AI suspected']},
        'Audio': {'score': 0, 'reasons': []},
        'Image': {'score': 0, 'reasons': []},
        'URL': {'score': 0, 'reasons': []}
    }
    result = generate_final_verdict_ai(mock_evidence, skip_llm=True)
    print(f"Final Score: {result['final_score']}%")
    print(f"Verdict: {result['risk_level']}")
    if result['final_score'] >= 60:
        print("SUCCESS: Max-Biased fusion works! (Score is High Risk)")
    else:
        print(f"FAILURE: Score is still too low ({result['final_score']}%)")

if __name__ == "__main__":
    test_face_crop()
    test_fusion_logic()
