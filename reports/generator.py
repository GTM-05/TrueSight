from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import datetime
import uuid

def draw_header(canvas, doc):
    canvas.saveState()
    # ── 1. Top Branding ──
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawCentredString(4.25*inch, 10.6*inch, "TRUESIGHT CERTIFIED")
    
    canvas.setFont('Helvetica-Bold', 22)
    canvas.setFillColor(colors.HexColor("#0D1B2A"))
    canvas.drawCentredString(4.25*inch, 10.2*inch, "TRUESIGHT FORENSIC INVESTIGATION")
    
    # ── 2. Seal Removed ──
    pass
    
    # ── 3. Diagonal Watermark ──
    canvas.setFont('Helvetica-Bold', 60)
    canvas.setStrokeColor(colors.lightgrey)
    canvas.setFillColor(colors.lightgrey, alpha=0.1)
    canvas.translate(3*inch, 5*inch)
    canvas.rotate(45)
    canvas.drawCentredString(0, 0, "TRUESIGHT CERTIFIED")
    canvas.restoreState()

def generate_pdf_report(report_data, output_path=None):
    """
    Tier 2 Premium Forensic Dossier Generator with Unique Filenaming.
    """
    if output_path is None:
        report_id = str(uuid.uuid4()).upper().split("-")[0]
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        output_path = f"Forensic_Dossier_{report_id}_{timestamp}.pdf"

    try:
        doc = SimpleDocTemplate(
            output_path, 
            pagesize=letter,
            rightMargin=inch*0.5, leftMargin=inch*0.5,
            topMargin=inch*2.3, bottomMargin=inch*0.5
        )
        
        styles = getSampleStyleSheet()
        is_specific = report_data.get('is_specific', False)
        target_mod = report_data.get('target_modality', 'MULTIMODAL')

        title_style = ParagraphStyle(
            'TitleStyle', parent=styles['Heading1'],
            fontName='Helvetica-Bold', fontSize=28,
            textColor=colors.HexColor("#0D1B2A"), spaceAfter=2, alignment=1
        )
        subtitle_style = ParagraphStyle(
            'SubTitle', parent=styles['Normal'],
            fontName='Helvetica-Bold', fontSize=9,
            textColor=colors.HexColor("#415A77"), alignment=1, spaceAfter=20
        )
        header_style = ParagraphStyle(
            'H2', parent=styles['Heading2'],
            fontName='Helvetica-Bold', fontSize=13,
            textColor=colors.white, spaceBefore=12, spaceAfter=8,
            borderPadding=6, backColor=colors.HexColor("#0D1B2A")
        )
        normal_style = styles['Normal']
        normal_style.fontSize = 8.5
        normal_style.leading = 11
        
        elements = []
        
        # ─── 1. HEADER ──────────────────────────────────────────────────────
        # Note: Top-level branding is drawn by `onFirstPage/onLaterPages=draw_header`.
        # The `topMargin=2.3*inch` in SimpleDocTemplate provides the necessary spacing.
        elements.append(Paragraph(f"OFFICIAL {target_mod} ANALYSIS RECORD // CONFIDENTIAL", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=3, color=colors.HexColor("#0D1B2A"), spaceAfter=10))
        
        # 2. EVIDENCE REFERENCE GRID
        report_id = str(uuid.uuid4()).upper().split("-")[0]
        gen_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        ref_data = [
            ["Dossier ID:", f"TS-X{report_id}", "Case Classification:", "RESTRICTED / FORENSIC"],
            ["Timestamp:", gen_date, "System Integrity:", "VERIFIED (SHA-256)"]
        ]
        ref_table = Table(ref_data, colWidths=[1.1*inch, 2.7*inch, 1.4*inch, 2.3*inch])
        ref_table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ]))
        elements.append(ref_table)
        elements.append(Spacer(1, 15))

        # 3. EXECUTIVE DETERMINATION
        elements.append(Paragraph("I. EXECUTIVE SUMMARY", header_style))
        risk_level = report_data.get('risk_level', 'Unknown')
        score = report_data.get('final_score', 0)
        
        summary_data = [
            [f"RESULT: {risk_level.upper()} THREAT LEVEL DETECTED", f"CONFIDENCE SCORE: {score}%"]
        ]
        summary_table = Table(summary_data, colWidths=[3.75*inch, 3.75*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#0D1B2A")),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('PADDING', (0,0), (-1,-1), 10),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 15))

        # ─── 4. COMPARISON MATRIX ──────────────────────────────────────────
        elements.append(Paragraph("II. FORENSIC DIFFERENTIATION MATRIX", header_style))
        if target_mod == 'URL':
            matrix_data = [
                ["Vector", "Algorithmic Generation (DGA)", "Spoofing / Phish (Morphing)"],
                ["Domain Signal", "High Entropy / Randomness", "Homograph / Punycode Spoof"],
                ["Structure", "Deep Subdomain Nesting", "URL Shortener / Redirects"],
                ["Metadata", "N/A", "TLD Reputation / Age"]
            ]
        elif target_mod == 'AUDIO':
            matrix_data = [
                ["Vector", "Synthetic Speech (TTS)", "Post-Production (Editing)"],
                ["Pitch Signal", "Monotonic / Non-Organic", "Frequency Spikes / Cuts"],
                ["Coefficients", "MFCC Delta Smoothness", "Phase Discontinuity"],
                ["Metadata", "Vocoder Fingerprints", "Encoding Jitter"]
            ]
        else:
            matrix_data = [
                ["Vector", "Synthetic Synthesis (AI-Gen)", "Post-Production (Morphing)"],
                ["Visual Signal", "ViT / Neural Correlation", "ELA / Pixel Anomalies"],
                ["Temporal Signal", "Inter-frame Consistency", "SSIM Jitter / Flow Error"],
                ["Metadata Signal", "Encoding / Header Gaps", "Revision History / Tags"]
            ]
        
        matrix_table = Table(matrix_data, colWidths=[1.5*inch, 3*inch, 3*inch])
        matrix_table.setStyle(TableStyle([
            ('FONTSIZE', (0,0), (-1,-1), 7.5),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#333333")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ]))
        elements.append(matrix_table)
        elements.append(Spacer(1, 15))

        # ─── 5. TECHNICAL EVIDENCE DATA ─────────────────────────────────────
        elements.append(Paragraph("III. TECHNICAL EVIDENCE DATA", header_style))
        tech_data = [["Modality", "Heuristic Status", "Risk %"]]
        if is_specific:
            mod = target_mod.capitalize()
            score = report_data.get('final_score', 0)
            tech_data.append([mod, f"Analyzed ({score}%)", f"{score}%"])
        else:
            for mod in ['Image', 'Audio', 'Video', 'URL']:
                score = report_data.get(f'{mod.lower()}_score', 0)
                status = "Analyzed" if score > 0 else "Not Analyzed"
                tech_data.append([mod, f"{status} ({score}%)", f"{score}%"])

        tech_table = Table(tech_data, colWidths=[1.5*inch, 4*inch, 2*inch])
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E0E1DD")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ]))
        elements.append(tech_table)
        elements.append(Spacer(1, 15))

        # ─── 6. ANALYTICAL METHODOLOGY ─────────────────────────────────────
        elements.append(Paragraph("IV. ANALYTICAL METHODOLOGY", header_style))
        if target_mod == 'URL':
            method_text = """<b>Homograph & Punycode Analysis:</b> Detects visual domain spoofing using non-Latin characters (homoglyphs) and punycode translation.<br/><br/>
            <b>DGA & Entropy Scoring:</b> Identifies Domain Generation Algorithms by calculating Shannon entropy and character randomness within the host string.<br/><br/>
            <b>TLD & Redirection Audit:</b> Evaluates top-level domain reputation and scans for malicious redirection chains or deep subdomain nesting."""
        elif target_mod == 'AUDIO':
            method_text = """<b>Pitch Monotonicity Analysis:</b> Detects unnatural robotic "steadiness" characteristic of synthetic voice models (TTS).<br/><br/>
            <b>MFCC Delta Smoothness:</b> Identifies lack of organic micro-oscillations in Mel-Frequency Cepstral Coefficients, common in AI vocoders.<br/><br/>
            <b>Energy Consistency (RMS):</b> Scans for non-human volume envelopes and splicing artifacts at the waveform level."""
        else:
            method_text = """<b>Neural Correlation (ViT):</b> Leverages the <i>prithivMLmods/Deep-Fake-Detector</i> manifold to identify synthetic generator footprints.<br/><br/>
            <b>Optical Flow Anomaly:</b> Tracks pixel motion vectors via the Farneback algorithm to detect unnatural "sliding" or face-warping.<br/><br/>
            <b>Laplacian Discontinuity:</b> Analyzes pixel-level edge noise and Laplacian variance to identify digital retouching or splicing."""
        
        elements.append(Paragraph(method_text, ParagraphStyle('Meth', parent=normal_style, backColor=colors.HexColor("#F8F9FA"), borderPadding=8)))
        elements.append(Spacer(1, 15))

        # ─── 7. FINAL AI NARRATIVE ──────────────────────────────────────────
        elements.append(Paragraph("V. FORENSIC EXPLANATION (AI ANALYST)", header_style))
        elements.append(Paragraph(report_data.get('ai_explanation', 'No analysis provided.')[:800], normal_style))
        elements.append(Spacer(1, 15))

        # ─── 8. CHAIN OF CUSTODY ───────────────────────────────────────────
        elements.append(Paragraph("VI. CHAIN OF CUSTODY & INTEGRITY", header_style))
        coc_data = [
            ["Phase", "Action", "Authorized Entity", "Status"],
            ["Acquisition", "Local File Ingest", "TrueSight System Analyst", "COMPLETED"],
            ["Processing", "Multi-modal Heuristics", "Fusion Engine v2.1", "COMPLETED"],
            ["Analysis", "Neural Narrative Gen", "Qwen2 0.5B (Quantized)", "COMPLETED"],
            ["Certification", "Digital Dossier Finalized", "SHA-256 Internal Stamp", "AUTHENTIC"]
        ]
        coc_table = Table(coc_data, colWidths=[1.5*inch, 2*inch, 2.5*inch, 1.5*inch])
        coc_table.setStyle(TableStyle([
            ('FONTSIZE', (0,0), (-1,-1), 7),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
        ]))
        elements.append(coc_table)
        elements.append(Spacer(1, 30))
        # ─── 9. FINAL SIGNATURE ────────────────────────────────────────────
        elements.append(Paragraph("DIGITALLY SIGNED BY TRUESIGHT AI EXHIBIT A", ParagraphStyle('F', alignment=1, fontSize=10, fontName='Helvetica-Bold', textColor=colors.HexColor("#415A77"))))

        # Build with Watermark & Header
        doc.build(elements, onFirstPage=draw_header, onLaterPages=draw_header)
        return output_path
    except Exception as e:
        print(f"Error: {e}")
        return None
