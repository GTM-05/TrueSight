import os

# Signatures often found in file-based polyglot/malicious attacks (steganography / web shell uploads)
MALICIOUS_SIGNATURES = [
    b'<?php',
    b'eval(',
    b'<script>',
    b'system(',
    b'cmd.exe',
    b'/bin/bash',
    b'powershell'
]

def scan_for_threats(file_path):
    """
    Performs a lightweight binary scan of a file to detect 
    embedded malicious payloads, scripts, or suspicious magic bytes.
    """
    results = {
        'score': 0,
        'reasons': [],
        'threat_level': 'Low'
    }
    
    if not os.path.exists(file_path):
        results['reasons'].append("File not found for threat scan")
        return results
        
    try:
        # Check file size (Arbitrary limit: 100MB)
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024 * 1024:
            results['score'] += 20
            results['reasons'].append(f"File is unusually large ({file_size / (1024*1024):.1f} MB), potentially hiding data.")
            
        with open(file_path, 'rb') as f:
            content = f.read()
            
            # 1. Scan for malicious code strings
            detected_sigs = []
            for sig in MALICIOUS_SIGNATURES:
                if sig in content:
                    detected_sigs.append(sig.decode('ascii', errors='ignore'))
                    
            if detected_sigs:
                results['score'] += 80
                results['reasons'].append(f"CRITICAL: Found embedded suspicious strings/scripts: {', '.join(detected_sigs)}. Potential polyglot or web shell payload!")
                
            # 2. Check for unexpected executable headers (MZ) anywhere after the start
            # Many image formats don't contain MZ. If MZ is found deep in a JPEG, it could be an appended exe. 
            if b'MZ' in content[10:] and b'This program cannot be run in DOS mode' in content:
                results['score'] += 90
                results['reasons'].append("CRITICAL: Embedded Windows Executable (PE/MZ) signature detected inside media file! High probability of malware steganography.")
                
            # 3. Check for embedded ELF binaries (Linux)
            if b'\x7fELF' in content[10:]:
                results['score'] += 90
                results['reasons'].append("CRITICAL: Embedded Linux Executable (ELF) signature detected inside media file! High probability of malware steganography.")
                
    except Exception as e:
        results['reasons'].append(f"Threat scan error: {str(e)}")
        
    final_score = min(100, results['score'])
    results['score'] = final_score
    results['threat_level'] = 'Critical' if final_score >= 80 else 'High' if final_score >= 50 else 'Low'
    
    return results
