"""
app-ai.py — TrueSight with AI-Enhanced Analysis + LLM Fusion (phi3:mini)

Run with:  streamlit run app-ai.py

Differences from app.py (standard):
  - modules/image_ai.py  → ViT AI-image-detector + ELA (catches Midjourney/DALL-E/SD)
  - modules/audio_ai.py  → Pitch monotonicity, MFCC delta, energy consistency
  - modules/video_ai.py  → Frame-level AI detection + SSIM temporal check
  - modules/url_ai.py    → tldextract, homograph detection, DGA entropy, shorteners
  - fusion/engine_ai.py  → phi3:mini reasons over full raw evidence (not just scores)

Requires:
  ollama serve   +   ollama pull phi3:mini
  pip install transformers torch  (for ViT AI-image detection)
Falls back gracefully if any optional dependency is missing.
"""

import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.url_ai import analyze_url_ai
from modules.image_ai import analyze_image_ai
from modules.metadata import check_metadata
from modules.audio_ai import analyze_audio_ai
from modules.video_ai import analyze_video_ai
from fusion.engine_ai import generate_final_verdict_ai
from modules.threats import scan_for_threats
from llm.phi2_ai import generate_ai_explanation
from reports.generator import generate_pdf_report

st.set_page_config(page_title="TrueSight AI · Deep Forensics", layout="wide")

# Session state
for key in ['url_result', 'image_result', 'audio_result', 'video_result']:
    if key not in st.session_state:
        st.session_state[key] = None

st.title("🧠 TrueSight AI: ML-Powered Forensics")
st.markdown(
    "**Enhanced mode** — Uses real ML models + advanced signal analysis for deeper accuracy.\n\n"
    "| Module | Enhancement |\n"
    "|---|---|\n"
    "| 🖼️ Image | ViT AI-image-detector (catches Midjourney, DALL·E, Stable Diffusion) |\n"
    "| 🔊 Audio | Pitch monotonicity + MFCC delta + energy consistency |\n"
    "| 🎥 Video | Frame-level AI detection + SSIM temporal check |\n"
    "| 🌐 URL | Homograph detection + shortener + DGA entropy + tldextract |\n\n"
    "> For the standard version: `streamlit run app.py`"
)

tab1, tab2, tab3, tab4 = st.tabs(["🎥 Video Analysis", "🖼️ Image Analysis", "🔊 Audio Analysis", "🌐 URL Analysis"])

# ── Tab 1: Video ──────────────────────────────────────────────────────────────
with tab1:
    st.header("Video Forensic Analysis (AI-Enhanced)")
    st.write("Frame-level AI-image detection + pitch/MFCC audio analysis + SSIM temporal consistency.")
    video_file = st.file_uploader("Upload a Video file", type=["mp4", "avi", "mov"])
    if video_file is not None:
        ext = os.path.splitext(video_file.name)[1]
        temp_vid = "temp_video_ai" + ext
        with open(temp_vid, "wb") as f:
            f.write(video_file.getbuffer())

        if st.button("Run Video Analysis"):
            with st.spinner("Scanning for malware..."):
                threat_res = scan_for_threats(temp_vid)
                if threat_res['score'] > 0:
                    st.error(f"🚨 THREAT DETECTED: {threat_res['reasons'][0]}")

            with st.spinner("Running AI-enhanced frame & audio analysis (may take 20–40s)..."):
                vid_res = analyze_video_ai(temp_vid)
                vid_res['threats'] = threat_res
                st.session_state.video_result = vid_res

                st.subheader("Video Analysis Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Manipulation Risk Score", f"{vid_res['score']}%")
                with col2:
                    st.metric("Frames Analyzed", vid_res.get('frames_analyzed', 0))
                with col3:
                    if threat_res['score'] > 0:
                        st.metric("Malware Threat Score", f"{threat_res['score']}%")

                if vid_res.get('ai_frame_scores'):
                    avg_ai = sum(vid_res['ai_frame_scores']) / len(vid_res['ai_frame_scores'])
                    st.info(f"🤖 Average frame AI-generation score: **{avg_ai:.0f}%**")

                ssim = vid_res.get('ssim', {})
                if ssim:
                    st.caption(f"SSIM Temporal: mean={ssim.get('ssim_mean', 'N/A')}, "
                               f"std={ssim.get('ssim_std', 'N/A')} "
                               f"{'⚠️ Anomaly detected' if ssim.get('anomaly') else '✅ Normal'}")

                if vid_res['reasons']:
                    st.write("**Suspicious Indicators:**")
                    for reason in vid_res['reasons']:
                        st.warning(f"⚠️ {reason}")
                else:
                    st.success("✅ No notable anomalies detected.")

# ── Tab 2: Image ──────────────────────────────────────────────────────────────
with tab2:
    st.header("Image Forensic Analysis (AI-Enhanced)")
    st.write("ViT AI-image-detector (Midjourney/DALL·E/SD detection) + ELA + EXIF analysis.")
    image_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if image_file is not None:
        with open("temp_img_ai.jpg", "wb") as f:
            f.write(image_file.getbuffer())

        if st.button("Run Image Analysis"):
            with st.spinner("Scanning for malware..."):
                threat_res = scan_for_threats("temp_img_ai.jpg")
                if threat_res['score'] > 0:
                    st.error(f"🚨 THREAT DETECTED: {threat_res['reasons'][0]}")

            with st.spinner("Running ViT AI-image detector + ELA + metadata (may take 10–20s)..."):
                img_res = analyze_image_ai("temp_img_ai.jpg")
                meta_res = check_metadata("temp_img_ai.jpg")

                # Weight towards AI-detection score if model was used
                if img_res['metrics'].get('model_used'):
                    combined_score = min(100, int(img_res['score'] * 0.7 + meta_res['score'] * 0.3))
                else:
                    combined_score = (img_res['score'] + meta_res['score']) // 2

                st.session_state.image_result = {
                    'score': combined_score,
                    'visual_score': img_res['score'],
                    'meta_score': meta_res['score'],
                    'reasons': img_res['reasons'] + meta_res['reasons'],
                    'ela_metrics': img_res.get('metrics', {}),
                    'ai_detection': img_res.get('ai_detection', {}),
                    'exif': meta_res.get('metadata', {}),
                    'ela_map': img_res.get('ela_map'),
                    'threats': threat_res,
                }

                st.subheader("Image Analysis Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Combined Risk Score", f"{combined_score}%")
                with col2:
                    ai_det = img_res.get('ai_detection', {})
                    st.metric("🤖 AI-Generation Probability", f"{ai_det.get('ai_probability', 0)}%")
                with col3:
                    if threat_res['score'] > 0:
                        st.metric("Malware Threat Score", f"{threat_res['score']}%")

                model_used = img_res['metrics'].get('model_used', False)
                ai_det = img_res.get('ai_detection', {})
                if model_used:
                    st.caption(f"🤖 Model: {ai_det.get('method', 'N/A')} | Label: **{ai_det.get('label', 'N/A')}**")
                else:
                    st.caption(f"⚠️ ML model not loaded — using heuristic fallback | {ai_det.get('method', '')}")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Visual + AI Detection**")
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

# ── Tab 3: Audio ──────────────────────────────────────────────────────────────
with tab3:
    st.header("Audio Forensic Analysis (AI-Enhanced)")
    st.write("Pitch monotonicity, MFCC delta smoothness, and energy consistency analysis.")
    audio_file = st.file_uploader("Upload an Audio file", type=["wav", "mp3"])
    if audio_file is not None:
        ext = os.path.splitext(audio_file.name)[1]
        with open("temp_audio_ai" + ext, "wb") as f:
            f.write(audio_file.getbuffer())

        if st.button("Run Audio Analysis"):
            with st.spinner("Scanning for malware..."):
                threat_res = scan_for_threats("temp_audio_ai" + ext)
                if threat_res['score'] > 0:
                    st.error(f"🚨 THREAT DETECTED: {threat_res['reasons'][0]}")

            with st.spinner("Extracting pitch, MFCC delta, energy patterns..."):
                aud_res = analyze_audio_ai("temp_audio_ai" + ext)

                st.session_state.audio_result = {
                    'score': aud_res['score'],
                    'reasons': aud_res['reasons'],
                    'metrics': aud_res.get('metrics', {}),
                    'threats': threat_res,
                }

                st.subheader("Audio Analysis Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Synthetic Audio Risk Score", f"{aud_res['score']}%")
                with col2:
                    if threat_res['score'] > 0:
                        st.metric("Malware Threat Score", f"{threat_res['score']}%")

                metrics = aud_res.get('metrics', {})
                if metrics:
                    st.caption(
                        f"Pitch std: {metrics.get('pitch_std', 0):.1f} Hz | "
                        f"MFCC delta std: {metrics.get('mfcc_delta_std', 0):.2f} | "
                        f"RMS std: {metrics.get('rms_std', 0):.4f} | "
                        f"Voiced ratio: {metrics.get('voiced_ratio', 0):.2f}"
                    )

                if aud_res['reasons']:
                    st.write("**Suspicious Acoustic Indicators:**")
                    for reason in aud_res['reasons']:
                        st.warning(f"⚠️ {reason}")
                else:
                    st.success("✅ No acoustic anomalies detected.")

                with st.expander("View Full Acoustic Metrics"):
                    st.json(aud_res.get('metrics', {}))

# ── Tab 4: URL ────────────────────────────────────────────────────────────────
with tab4:
    st.header("URL Analysis (AI-Enhanced)")
    st.write("Homograph detection, DGA entropy, URL shortener, deep subdomain nesting, and more.")
    url_input = st.text_input("Enter a URL to analyze")
    if st.button("Analyze URL"):
        if url_input:
            with st.spinner(f"Deep-analyzing {url_input}..."):
                result = analyze_url_ai(url_input)

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
                        st.error(f"Risk Level: 🔴 {result['risk_level']}")
                    elif result['risk_level'] == 'Medium':
                        st.warning(f"Risk Level: 🟡 {result['risk_level']}")
                    else:
                        st.success(f"Risk Level: 🟢 {result['risk_level']}")

                feats = result.get('features', {})
                st.caption(
                    f"Domain: **{feats.get('domain', 'N/A')}** | "
                    f"TLD: **.{feats.get('tld', 'N/A')}** | "
                    f"Entropy: {feats.get('domain_entropy', 0)} | "
                    f"Subdomain depth: {feats.get('subdomain_depth', 0)}"
                )

                if result['reasons']:
                    st.write("**Suspicious Indicators:**")
                    for reason in result['reasons']:
                        st.warning(f"⚠️ {reason}")
                else:
                    st.success("✅ No suspicious indicators found.")

                with st.expander("View Full Feature Analysis"):
                    st.json(result['features'])
        else:
            st.error("Please enter a URL first.")

# ── Final Report ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("Results & Reporting")
st.caption("🧠 phi3:mini will reason over ALL raw forensic evidence to produce per-category risk scores.")

if st.button("Generate Final Forensic Report (AI-Powered)", type="primary"):
    with st.spinner("LLM reasoning over full evidence — may take 30–60 seconds..."):
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
            st.error("No data analyzed yet. Run at least one analysis first.")
        else:
            verdict = generate_final_verdict_ai(all_evidence)

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

            with st.spinner("Generating 3-stage investigator narrative..."):
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
                    st.download_button(
                        "Download PDF Report", f,
                        file_name="Forensic_AI_Report.pdf", mime="application/pdf"
                    )
