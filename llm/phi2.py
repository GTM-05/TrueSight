import ollama

def generate_ai_explanation(modality_data):
    """
    Connects to the local Phi model via Ollama to generate a clear, 
    forensic explanation of the detected anomalies.
    """
    prompt = f"""
    You are a professional Cyber Forensic Investigator AI evaluating a media file.
    Analyze the provided forensic data and output a structured 3-stage report. Do not make up information.
    
    Data:
    {modality_data}
    
    Structure your exact response using these 3 markdown headings:
    ### Stage 1: Threat Assessment
    (Analyze if there are malicious payloads or steganography).
    ### Stage 2: AI-Generation Assessment
    (Analyze indicators that this was created completely by AI, such as missing EXIF, specific resolutions, or silent video tracks).
    ### Stage 3: Deepfake & Manipulation Assessment
    (Analyze if a real photo/video was edited or spliced).
    
    End with a FINAL VERDICT summary.
    """
    
    try:
        # We use strict temperature for deterministic, factual outputs
        response = ollama.generate(model='phi', prompt=prompt, options={'temperature': 0.1})
        if response and 'response' in response:
            return response['response']
        else:
            return "Failed to parse Ollama response."
    except Exception as e:
        return f"AI Engine Error (Ensure Ollama is running 'ollama run phi'): {str(e)}"
