from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import inch
import datetime
import uuid

def generate_pdf_report(report_data, output_path="Forensic_Dossier.pdf"):
    """
    Generates a highly professional Forensic Dossier using ReportLab Platypus.
    """
    try:
        doc = SimpleDocTemplate(
            output_path, 
            pagesize=letter,
            rightMargin=inch, leftMargin=inch,
            topMargin=inch, bottomMargin=inch
        )
        
        styles = getSampleStyleSheet()
        
        # Custom Styles
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=24,
            textColor=colors.HexColor("#1A237E"),
            spaceAfter=12,
            alignment=1 # Center
        )
        
        subtitle_style = ParagraphStyle(
            'SubtitleStyle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.gray,
            alignment=1,
            spaceAfter=20
        )
        
        header_style = ParagraphStyle(
            'HeaderStyle',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=colors.HexColor("#0D47A1"),
            spaceBefore=15,
            spaceAfter=10
        )
        
        normal_style = styles['Normal']
        normal_style.fontSize = 10
        normal_style.leading = 14
        
        elements = []
        
        # 1. DOCUMENT HEADER
        elements.append(Paragraph("TRUESIGHT CYBER FORENSICS", title_style))
        elements.append(Paragraph("OFFICIAL MULTI-MODAL INVESTIGATION DOSSIER", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1A237E"), spaceAfter=20))
        
        # 2. METADATA TABLE
        report_id = str(uuid.uuid4()).upper().split("-")[0]
        gen_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        meta_data = [
            ["Dossier ID:", f"TS-{report_id}", "Classification:", "CONFIDENTIAL / TL: RED"],
            ["Generated:", gen_date, "Target Modalities:", ", ".join([m for m, s in report_data.get('modalities', {}).items() if 'Analyzed' in s])]
        ]
        
        meta_table = Table(meta_data, colWidths=[1*inch, 2*inch, 1.2*inch, 2*inch])
        meta_table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('TEXTCOLOR', (3,0), (3,0), colors.red), # CONFIDENTIAL in red
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 20))
        
        # 3. EXECUTIVE SUMMARY & VERDICT
        elements.append(Paragraph("1. EXECUTIVE SUMMARY & VERDICT", header_style))
        
        risk_level = report_data.get('risk_level', 'Unknown')
        score = report_data.get('final_score', 0)
        
        # Color code the verdict
        bg_color = colors.HexColor("#E8F5E9") # Green
        text_color = colors.HexColor("#2E7D32")
        if risk_level == 'High':
            bg_color = colors.HexColor("#FFEBEE") # Red-ish
            text_color = colors.HexColor("#C62828")
            bg_color = colors.HexColor("#FFEBEE") # Actually #FFEBEE is pink red
        elif risk_level == 'Medium':
            bg_color = colors.HexColor("#FFF8E1") # Orange/Yellow
            bg_color = colors.HexColor("#FFF3E0")
            text_color = colors.HexColor("#E65100")
            
        verdict_data = [
            [Paragraph("<b>FINAL RISK VERDICT</b>", styles['Normal']), Paragraph(f"<b>{risk_level.upper()} RISK ({score}%)</b>", ParagraphStyle('Verdict', textColor=text_color, alignment=1, fontSize=12))]
        ]
        verdict_table = Table(verdict_data, colWidths=[2.5*inch, 3.5*inch])
        verdict_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), bg_color),
            ('BOX', (0,0), (-1,-1), 1, text_color),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 12),
        ]))
        
        elements.append(verdict_table)
        elements.append(Spacer(1, 20))
        
        # 4. MODALITY BREAKDOWN
        elements.append(Paragraph("2. HEURISTIC MODALITY BREAKDOWN", header_style))
        
        mod_table_data = [["Modality", "Status / Local Risk Heuristic"]]
        for mod, status in report_data.get('modalities', {}).items():
            mod_table_data.append([mod, status])
            
        mod_table = Table(mod_table_data, colWidths=[2*inch, 4*inch])
        mod_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E0E0E0")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ]))
        elements.append(mod_table)
        elements.append(Spacer(1, 20))
        
        # 5. AI FORENSIC EXPLANATION (3-STAGE)
        elements.append(Paragraph("3. AI INVESTIGATOR ANALYSIS (PHI-2)", header_style))
        
        ai_raw_text = report_data.get('ai_explanation', 'No AI explanation generated.')
        
        # Parse potential markdown headers (###) from the Phi-2 text into structured paragraphs
        for line in ai_raw_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith('###'):
                # It's a stage header
                header_text = line.replace('###', '').strip()
                h_style = ParagraphStyle('SubH', parent=styles['Heading3'], fontName='Helvetica-Bold', textColor=colors.HexColor("#1565C0"), spaceBefore=10, spaceAfter=5)
                elements.append(Paragraph(header_text, h_style))
            elif line.startswith('-') or line.startswith('*'):
                # Bullet point
                b_style = ParagraphStyle('Bullet', parent=normal_style, leftIndent=20)
                elements.append(Paragraph(line, b_style))
            else:
                # Normal paragraph
                elements.append(Paragraph(line, normal_style))
                elements.append(Spacer(1, 6))
                
        # 6. FOOTER / DISCLAIMER
        elements.append(Spacer(1, 40))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey, spaceAfter=10))
        disclaimer = "CONFIDENTIALITY NOTICE: This forensic dossier was generated automatically by the TrueSight Multimodal Cyber Forensics platform. The heuristic scores and AI-synthesized findings contained within should be reviewed by a certified human analyst before being utilized in legal or disciplinary proceedings."
        elements.append(Paragraph(disclaimer, ParagraphStyle('Disclaimer', fontName='Helvetica-Oblique', fontSize=8, textColor=colors.gray)))
        
        # Build PDF
        doc.build(elements)
        return output_path
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return None
