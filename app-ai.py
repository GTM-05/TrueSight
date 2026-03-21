"""
app-ai.py — TrueSight with LLM-powered fusion reasoning (phi3:mini).

Run with:  streamlit run app-ai.py

The LLM (phi3:mini via Ollama) reasons over ALL raw forensic evidence at the
fusion stage and produces structured per-category scores:
  - Threat Score        (malware / steganography)
  - AI-Generated Score  (likelihood content is fully AI-made)
  - Manipulation Score  (deepfake / splicing / tampering)

Requires: ollama serve  +  ollama pull phi3:mini
Falls back to simple averaging if Ollama is unavailable.

For the original version (LLM only for report narration), run: streamlit run app.py
"""

import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.url import analyze_url
from modules.image import analyze_image
from modules.metadata import check_metadata
from modules.audio import analyze_audio
from modules.video import analyze_video
from fusion.engine_ai import generate_final_verdict_ai
from modules.threats import scan_for_threats
from llm.phi2_ai import generate_ai_explanation
from reports.generator import generate_pdf_report

st.set_page_config(page_title="TrueSight AI · Deep Forensics", layout="wide")

# Initialize session state
if 'url_result' not in st.session_state:
    st.session_state.url_result = None
if 'image_result' not in st.session_state:
    st.session_state.image_result = None
if 'audio_result' not in st.session_state:
    st.session_state.audio_result = None
if 'video_result' not in st.session_state:
    st.session_state.video_result = None

st.title("🧠 TrueSight AI: LLM-Powered Forensics")
st.markdown(
    "**Enhanced mode** — The LLM (phi3:mini) reasons over raw forensic evidence to produce "
    "structured per-category risk verdicts. Analyze **Video, Image, Audio, and URL** inputs.\n\n"
    "> For the standard version, run `streamlit run app.py`"
)

tab1, tab2, tab3, tab4 = st.tabs(["🎥 Video Analysis", "🖼️ Image Analysis", "🔊 Audio Analysis", "🌐 URL Analysis"])

with tab1:
    st.header("Video Forensic Analysis")
    st.write("Extracts frames and audio from a 10s video clip. AI analyzes heuristics to detect deepfakes.")
    video_file = st.file_uploader("Upload a Video file", type=["mp4", "avi", "mov"])
    if video_file is not None:
        ext = os.path.splitext(video_file.name)[1]
        temp_vid = "temp_video" + ext
        with open(temp_vid, "wb") as f:
            f.write(video_file.getbuffer())

        if st.button("Run Video Analysis"):
            with st.spinner("Scanning for malware..."):
                threat_res = scan_for_threats(temp_vid)
                if threat_res['score'] > 0:
                    st.error(f"🚨 THREAT DETECTED: {threat_res['reasons'][0]}")

            with st.spinner("Processing video frames, extracting audio..."):
                vid_res = analyze_video(temp_vid)
                vid_res['threats'] = threat_res
                st.session_state.video_result = vid_res

                st.subheader("Video Analysis Results")
                st.metric("Video Manipulation Risk Score", f"{vid_res['score']}%")
                if threat_res['score'] > 0:
                    st.metric("Malware Threat Score", f"{threat_res['score']}%")

                if vid_res['reasons']:
                    st.write("**Suspicious Indicators found:**")
                    for reason in vid_res['reasons']:
                        st.warning(f"⚠️ {reason}")
                else:
                    st.success("✅ No notable anomalies detected.")

with tab2:
    st.header("Image Forensic Analysis")
    st.write("Uses Error Level Analysis (ELA) and Metadata extraction to detect manipulation.")
    image_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if image_file is not None:
        with open("temp_img.jpg", "wb") as f:
            f.write(image_file.getbuffer())

        if st.button("Run Image Analysis"):
            with st.spinner("Scanning for malware..."):
                threat_res = scan_for_threats("temp_img.jpg")
                if threat_res['score'] > 0:
                    st.error(f"🚨 THREAT DETECTED: {threat_res['reasons'][0]}")

            with st.spinner("Analyzing image and extracting metadata..."):
                img_res = analyze_image("temp_img.jpg")
                meta_res = check_metadata("temp_img.jpg")
                combined_score = (img_res['score'] + meta_res['score']) // 2

                # Store full evidence for LLM reasoning
                st.session_state.image_result = {
                    'score': combined_score,
                    'visual_score': img_res['score'],
                    'meta_score': meta_res['score'],
                    'reasons': img_res['reasons'] + meta_res['reasons'],
                    'ela_metrics': img_res.get('metrics', {}),
                    'exif': meta_res.get('metadata', {}),
                    'ela_map': img_res.get('ela_map'),
                    'threats': threat_res,
                }

                st.subheader("Image Analysis Results")
                st.metric("Manipulation Risk Score", f"{combined_score}%")
                if threat_res['score'] > 0:
                    st.metric("Malware Threat Score", f"{threat_res['score']}%")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Visual Forensics (ELA & Blur)**")
                    for reason in img_res['reasons']:
                        st.warning(f"⚠️ {reason}")
                    if not img_res['reasons']:
                        st.success("✅ No visual anomalies detected.")
                    if img_res.get('ela_map'):
                        st.image(img_res['ela_map'], caption="Error Level Analysis Map", use_container_width=True)

                with col2:
                    st.write("**Metadata Forensics (EXIF)**")
                    for reason in meta_res['reasons']:
                        st.warning(f"⚠️ {reason}")
                    if not meta_res['reasons']:
                        st.success("✅ No metadata anomalies detected.")
                    with st.expander("View Raw EXIF Data"):
                        st.json(meta_res['metadata'])

with tab3:
    st.header("Audio Forensic Analysis")
    st.write("Extracts MFCCs and spectral features to detect synthetic or deepfake audio.")
    audio_file = st.file_uploader("Upload an Audio file", type=["wav", "mp3"])
    if audio_file is not None:
        ext = os.path.splitext(audio_file.name)[1]
        with open("temp_audio" + ext, "wb") as f:
            f.write(audio_file.getbuffer())

        if st.button("Run Audio Analysis"):
            with st.spinner("Scanning for malware..."):
                threat_res = scan_for_threats("temp_audio" + ext)
                if threat_res['score'] > 0:
                    st.error(f"🚨 THREAT DETECTED: {threat_res['reasons'][0]}")
            with st.spinner("Extracting acoustic features..."):
                aud_res = analyze_audio("temp_audio" + ext)

                st.session_state.audio_result = {
                    'score': aud_res['score'],
                    'reasons': aud_res['reasons'],
                    'metrics': aud_res.get('metrics', {}),
                    'threats': threat_res,
                }

                st.subheader("Audio Analysis Results")
                st.metric("Synthetic Audio Risk Score", f"{aud_res['score']}%")
                if threat_res['score'] > 0:
                    st.metric("Malware Threat Score", f"{threat_res['score']}%")

                if aud_res['reasons']:
                    st.write("**Suspicious Acoustic Indicators:**")
                    for reason in aud_res['reasons']:
                        st.warning(f"⚠️ {reason}")
                else:
                    st.success("✅ No acoustic anomalies detected.")

                with st.expander("View Acoustic Metrics"):
                    st.json(aud_res['metrics'])

with tab4:
    st.header("URL Phishing Detection")
    st.write("Uses lexical heuristics and offline feature tracking to detect phishing URLs.")
    url_input = st.text_input("Enter a URL to analyze")
    if st.button("Analyze URL"):
        if url_input:
            with st.spinner(f"Analyzing {url_input}..."):
                result = analyze_url(url_input)

                st.session_state.url_result = {
                    'score': result['score'],
                    'reasons': result['reasons'],
                    'features': result.get('features', {}),
                }

                st.subheader("Analysis Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Phishing Risk Score", f"{result['score']}%")
                with col2:
                    if result['risk_level'] == 'High':
                        st.error(f"Risk Level: {result['risk_level']}")
                    elif result['risk_level'] == 'Medium':
                        st.warning(f"Risk Level: {result['risk_level']}")
                    else:
                        st.success(f"Risk Level: {result['risk_level']}")

                if result['reasons']:
                    st.write("**Suspicious Indicators found:**")
                    for reason in result['reasons']:
                        st.warning(f"⚠️ {reason}")
                else:
                    st.success("✅ No suspicious indicators found.")

                with st.expander("View Raw Feature Data"):
                    st.json(result['features'])
        else:
            st.error("Please enter a URL first.")

st.divider()
st.subheader("Results & Reporting")
st.caption("🧠 phi3:mini will reason over ALL raw evidence to produce per-category scores before generating the report.")

if st.button("Generate Final Forensic Report (AI-Powered)", type="primary"):
    with st.spinner("LLM reasoning over full forensic evidence — this may take 30–60 seconds..."):
        all_evidence = {}
        if st.session_state.url_result:
            all_evidence['URL'] = st.session_state.url_result
        if st.session_state.image_result:
            all_evidence['Image'] = st.session_state.image_result
        if st.session_state.audio_result:
            all_evidence['Audio'] = st.session_state.audio_result
        if st.session_state.video_result:
            all_evidence['Video'] = st.session_state.video_result

        if not all_evidence:
            st.error("No data has been analyzed yet. Please run at least one analysis first.")
        else:
            # LLM reasons over ALL evidence → structured JSON verdict
            verdict = generate_final_verdict_ai(all_evidence)

            # 4-column metric breakdown
            st.subheader("🧠 LLM-Reasoned Risk Breakdown")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🦠 Threat Score", f"{verdict.get('threat_score', 0)}%")
            with col2:
                st.metric("🤖 AI-Generated Score", f"{verdict.get('ai_generated_score', 0)}%")
            with col3:
                st.metric("🎭 Manipulation Score", f"{verdict.get('manipulation_score', 0)}%")
            with col4:
                st.metric("⚡ Final Score", f"{verdict.get('final_score', 0)}%")

            risk_level = verdict.get('risk_level', 'Low')
            confidence = verdict.get('confidence', 'Low')
            key_findings = verdict.get('key_findings', [])

            if risk_level == 'High':
                st.error(f"🚨 **VERDICT: {risk_level} RISK** — Confidence: {confidence}")
            elif risk_level == 'Medium':
                st.warning(f"⚠️ **VERDICT: {risk_level} RISK** — Confidence: {confidence}")
            else:
                st.success(f"✅ **VERDICT: {risk_level} RISK** — Confidence: {confidence}")

            if key_findings:
                with st.expander("🔍 Key LLM Findings"):
                    for finding in key_findings:
                        st.markdown(f"- {finding}")

            with st.spinner("Generating 3-stage AI investigator narrative..."):
                ai_explanation = generate_ai_explanation(verdict, all_evidence)

            report_data = {
                'final_score': verdict['final_score'],
                'risk_level': risk_level,
                'threat_score': verdict.get('threat_score', 0),
                'ai_generated_score': verdict.get('ai_generated_score', 0),
                'manipulation_score': verdict.get('manipulation_score', 0),
                'confidence': confidence,
                'key_findings': key_findings,
                'modalities': {
                    'URL': f"Analyzed ({st.session_state.url_result['score']}%)" if st.session_state.url_result else "Not Analyzed",
                    'Image': f"Analyzed ({st.session_state.image_result['score']}%)" if st.session_state.image_result else "Not Analyzed",
                    'Audio': f"Analyzed ({st.session_state.audio_result['score']}%)" if st.session_state.audio_result else "Not Analyzed",
                    'Video': f"Analyzed ({st.session_state.video_result['score']}%)" if st.session_state.video_result else "Not Analyzed",
                },
                'ai_explanation': ai_explanation
            }

            pdf_path = generate_pdf_report(report_data)

            st.success("Report Generated Successfully!")
            st.info(f"**AI Investigator Report:**\n\n{ai_explanation}")

            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF Report", f, file_name="Forensic_AI_Report.pdf", mime="application/pdf")
