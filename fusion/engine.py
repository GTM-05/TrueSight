def generate_final_verdict(url_score, image_score, audio_score, video_score):
    """
    Fuses the individual module scores into a single final risk score.
    """
    scores = []
    if url_score is not None: scores.append(url_score)
    if image_score is not None: scores.append(image_score)
    if audio_score is not None: scores.append(audio_score)
    if video_score is not None: scores.append(video_score)
    
    if not scores:
        return {'final_score': 0, 'risk_level': 'Low', 'summary': 'No data analyzed.'}
        
    final_score = sum(scores) / len(scores)
    
    risk_level = 'High' if final_score >= 60 else 'Medium' if final_score >= 30 else 'Low'
    
    return {
        'final_score': int(final_score),
        'risk_level': risk_level,
        'summary': f"Analyzed {len(scores)} modalities. Overall Risk: {risk_level} ({int(final_score)}%)"
    }
