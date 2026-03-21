"""
app.py — TrueSight AI Multimodal Cyber Forensics

Run with:  streamlit run app.py

Features:
  - modules/image_ai.py  → ViT AI-image-detector + ELA
  - modules/audio_ai.py  → Pitch, MFCC delta, energy consistency
  - modules/video_ai.py  → Frame-level AI + SSIM temporal check + ffprobe
  - modules/url_ai.py    → tldextract, homograph, DGA entropy
  - fusion/engine.py     → qwen2:0.5b single-pass reasoning

Requires:
  ollama serve   +   ollama pull qwen2:0.5b
  pip install transformers torch
"""

import streamlit as st
import os
import sys
import tempfile
import logging

# ── App Logging (tail -f app.log to watch live) ─────────────────────────────
logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.log'),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("truesight")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.url import analyze_url
from modules.image import analyze_image, _load_detector
from modules.metadata import check_metadata
from modules.audio import analyze_audio
from modules.video import analyze_video
from fusion.engine import generate_final_verdict_ai
from modules.threats import scan_for_threats
from reports.generator import generate_pdf_report

# ── ViT model: cached in RAM, loads lazily on first video/image upload ────────
@st.cache_resource(show_spinner="🧠 Loading AI detector (first run only)...")
def _get_detector():
    return _load_detector()

st.set_page_config(page_title="TrueSight AI · Deep Forensics", layout="wide", page_icon="🧿")

# ── Cyber Theme CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Orbitron:wght@700;900&display=swap');

/* Base */
html, body, [class*="css"], .stApp {
    background-color: #050d1a !important;
    color: #c8e6ff !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Scanline overlay */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,245,255,0.015) 2px, rgba(0,245,255,0.015) 4px);
    pointer-events: none;
    z-index: 9999;
}

/* Title */
h1 { font-family: 'Orbitron', monospace !important; color: #00f5ff !important;
     text-shadow: 0 0 20px #00f5ff, 0 0 40px rgba(0,245,255,0.4); letter-spacing: 2px; }
h2, h3 { color: #00f5ff !important; font-family: 'JetBrains Mono', monospace !important;
          text-shadow: 0 0 8px rgba(0,245,255,0.5); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050d1a 0%, #0a1628 100%) !important;
    border-right: 1px solid rgba(0,245,255,0.25) !important;
}
[data-testid="stSidebar"] * { color: #c8e6ff !important; }

/* Tabs */
[data-baseweb="tab-list"] { background: transparent !important; gap: 4px; }
[data-baseweb="tab"] {
    background: rgba(0,245,255,0.05) !important;
    border: 1px solid rgba(0,245,255,0.2) !important;
    border-radius: 4px !important;
    color: #7ecfff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    transition: all 0.2s ease;
}
[data-baseweb="tab"]:hover {
    background: rgba(0,245,255,0.15) !important;
    border-color: #00f5ff !important;
    color: #00f5ff !important;
}
[aria-selected="true"] {
    background: rgba(0,245,255,0.2) !important;
    border-color: #00f5ff !important;
    color: #00f5ff !important;
    box-shadow: 0 0 12px rgba(0,245,255,0.4), inset 0 0 12px rgba(0,245,255,0.05) !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid #00ff88 !important;
    color: #00ff88 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    border-radius: 3px !important;
    transition: all 0.2s ease;
    text-transform: uppercase;
    font-size: 11px !important;
}
.stButton > button:hover {
    background: rgba(0,255,136,0.15) !important;
    box-shadow: 0 0 20px rgba(0,255,136,0.4) !important;
    color: #ffffff !important;
}
/* Primary button */
.stButton > button[kind="primary"] {
    border-color: #00f5ff !important;
    color: #00f5ff !important;
    box-shadow: 0 0 10px rgba(0,245,255,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    background: rgba(0,245,255,0.15) !important;
    box-shadow: 0 0 25px rgba(0,245,255,0.6) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(0,245,255,0.4) !important;
    border-radius: 6px !important;
    background: rgba(0,245,255,0.03) !important;
    padding: 10px;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00f5ff !important;
    background: rgba(0,245,255,0.07) !important;
    box-shadow: 0 0 20px rgba(0,245,255,0.15) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(0,245,255,0.05) !important;
    border: 1px solid rgba(0,245,255,0.2) !important;
    border-radius: 6px !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"] { color: #00f5ff !important; font-size: 2em !important; }
[data-testid="stMetricLabel"] { color: #7ecfff !important; font-size: 11px !important; }

/* Progress bar */
[data-testid="stProgress"] > div > div > div > div {
    background: linear-gradient(90deg, #00f5ff, #00ff88) !important;
    box-shadow: 0 0 8px #00f5ff !important;
}

/* Text input */
[data-testid="stTextInput"] input {
    background: rgba(0,245,255,0.05) !important;
    border: 1px solid rgba(0,245,255,0.3) !important;
    border-radius: 4px !important;
    color: #c8e6ff !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #00f5ff !important;
    box-shadow: 0 0 12px rgba(0,245,255,0.3) !important;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid rgba(0,245,255,0.2) !important;
    border-radius: 4px !important;
    background: rgba(0,245,255,0.02) !important;
}

/* Checkbox */
[data-testid="stCheckbox"] label { color: #7ecfff !important; }

/* Success / Warning / Error */
[data-testid="stAlert"] { border-radius: 4px !important; font-family: 'JetBrains Mono', monospace !important; }

/* Divider */
hr { border-color: rgba(0,245,255,0.15) !important; }

/* Spinner */
[data-testid="stSpinner"] { color: #00f5ff !important; }

/* Dataframe / tables */
[data-testid="stDataFrame"] { border: 1px solid rgba(0,245,255,0.2) !important; }

/* Caption */
[data-testid="stCaptionContainer"] { color: rgba(0,245,255,0.6) !important; font-size: 11px !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #050d1a; }
::-webkit-scrollbar-thumb { background: rgba(0,245,255,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00f5ff; }
</style>
""", unsafe_allow_html=True)


# Session state
for key in ['url_result', 'image_result', 'audio_result', 'video_result']:
    if key not in st.session_state:
        st.session_state[key] = None

st.title("🧠 TrueSight AI: Multimodal Cyber Forensics")
st.markdown(
    "**Professional mode** — Uses real ML models + advanced signal analysis for deep accuracy.\n\n"
    "| Module | Analysis Stack |\n"
    "|---|---|\n"
    "| 🖼️ Image | ViT AI-image-detector + ELA (catches Midjourney, DALL·E, Stable Diffusion) |\n"
    "| 🔊 Audio | Pitch monotonicity + MFCC delta + energy consistency |\n"
    "| 🎥 Video | Frame-level AI + SSIM temporal check + ffprobe metadata |\n"
    "| 🌐 URL | Homograph detection + shortener + DGA entropy + tldextract |\n"
)

tab1, tab2, tab3, tab4 = st.tabs(["🎥 Video Analysis", "🖼️ Image Analysis", "🔊 Audio Analysis", "🌐 URL Analysis"])

# ── Tab 1: Video ──────────────────────────────────────────────────────────────
with tab1:
    st.header("Video Forensic Analysis (AI-Enhanced)")
    st.write("Frame-level AI-image detection + pitch/MFCC audio analysis + SSIM temporal consistency.")
    
    # NEW: Low Resource Mode for 8GB RAM systems
    low_res = st.sidebar.checkbox("🚀 Low Resource Mode", value=False, help="Enable on low-RAM systems (reduces frame sampling to 1 frame, skips SSIM).")
    deep_scan = st.sidebar.checkbox("🔬 Deep Forensic Scan", value=False, help="Increases frame sampling density (8 frames) for maximum accuracy on long videos.")
    skip_ai = st.sidebar.checkbox("📄 Turbo Report (Instant)", value=False, help="Bypasses LLM reasoning for instant technical report generation.")
    
    video_file = st.file_uploader("Upload a Video file", type=["mp4", "avi", "mov"])
    if video_file is not None:
        suffix = os.path.splitext(video_file.name)[1]
        tmp_vid = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_vid.write(video_file.getbuffer())
        tmp_vid.close()
        temp_vid = tmp_vid.name

        if st.button("Run Video Analysis"):
            with st.spinner("Scanning for malware..."):
                threat_res = scan_for_threats(temp_vid)
                if threat_res['score'] > 0:
                    st.error(f"🚨 THREAT DETECTED: {threat_res['reasons'][0]}")

            with st.spinner("Running AI-enhanced frame & audio analysis (Deep Scan active)..." if deep_scan else "Running standard analysis..."):
                vid_res = analyze_video(temp_vid, low_resource=low_res, deep_scan=deep_scan)
                vid_res['threats'] = threat_res
                st.session_state.video_result = vid_res

                st.subheader("Video Analysis Results")
                vcol1, vcol2, vcol3, vcol4 = st.columns(4)
                with vcol1:
                    st.metric("📹 Total Video Risk", f"{vid_res['score']}%")
                with vcol2:
                    st.metric("🤖 AI Synthesis", f"{vid_res.get('ai_gen_score',0)}%", help="ViT-based detection of Sora/Runway synthetic generation.")
                with vcol3:
                    st.metric("🎭 Morphing", f"{vid_res.get('manip_score',0)}%", help="SSIM temporal inconsistency and metadata morphing indicators.")
                with vcol4:
                    if threat_res['score'] > 0:
                        st.metric("🦠 Malware", f"{threat_res['score']}%")
                    else:
                        st.metric("🛡️ File Safety", "Secure")

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
                
                # --- Specific Report Button ---
                st.divider()
                if st.button("📄 Generate Detailed Video Analysis Report", key="vid_rep"):
                    vid_report_data = {
                        'is_specific': True,
                        'target_modality': 'VIDEO',
                        'final_score': vid_res['score'],
                        'risk_level': 'High' if vid_res['score'] >= 60 else 'Medium' if vid_res['score'] >= 30 else 'Low',
                        'specific_details': {
                            'AI Synthesis Probability': f"{vid_res.get('ai_gen_score',0)}%",
                            'Manipulation/Morphing': f"{vid_res.get('manip_score',0)}%",
                            'Temporal Consistency (SSIM)': ssim.get('ssim_mean', 'N/A') if ssim else "Skipped",
                            'Threat Indicators': ", ".join([r for r in vid_res['reasons']]) if vid_res['reasons'] else "None"
                        },
                        'key_findings': vid_res['reasons'],
                        'ai_explanation': f"Video forensic audit detecting AI synthesis ({vid_res.get('ai_gen_score',0)}%) and post-production morphing ({vid_res.get('manip_score',0)}%)."
                    }
                    path = generate_pdf_report(vid_report_data)
                    if path:
                        with open(path, "rb") as f:
                            st.download_button("Download Detailed Video PDF", f, file_name=os.path.basename(path))

# ── Tab 2: Image ──────────────────────────────────────────────────────────────
with tab2:
    st.header("Image Forensic Analysis (AI-Enhanced)")
    st.write("ViT AI-image-detector (Midjourney/DALL·E/SD detection) + ELA + EXIF analysis.")
    image_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if image_file is not None:
        tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_img.write(image_file.getbuffer())
        tmp_img.close()
        temp_img_path = tmp_img.name

        if st.button("Run Image Analysis"):
            with st.spinner("Scanning for malware..."):
                threat_res = scan_for_threats(temp_img_path)
                if threat_res['score'] > 0:
                    st.error(f"🚨 THREAT DETECTED: {threat_res['reasons'][0]}")

            with st.spinner("Running ViT AI-image detector + ELA + metadata (may take 10–20s)..."):
                img_res = analyze_image(temp_img_path, low_resource=low_res)
                meta_res = check_metadata(temp_img_path)

                # Weight towards AI-detection score if model was used
                # Accuracy Boost: Prioritize the most suspicious evidence
                max_component = max(img_res['score'], meta_res['score'])
                min_component = min(img_res['score'], meta_res['score'])
                
                if img_res['metrics'].get('model_used') and img_res['score'] >= 50:
                    # Model found something, trust it more
                    combined_score = int(img_res['score'] * 0.9 + meta_res['score'] * 0.1)
                else:
                    combined_score = int(max_component * 0.8 + min_component * 0.2)

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
                
                # --- Specific Report Button ---
                st.divider()
                if st.button("📄 Generate Detailed Image Analysis Report", key="img_rep"):
                    img_report_data = {
                        'is_specific': True,
                        'target_modality': 'IMAGE',
                        'final_score': combined_score,
                        'risk_level': 'High' if combined_score >= 60 else 'Medium' if combined_score >= 30 else 'Low',
                        'specific_details': {
                            'AI-Generation Probability': f"{ai_det.get('ai_probability', 0)}% ({ai_det.get('method', 'N/A')})",
                            'ELA Std Dev': f"{st.session_state.image_result.get('ela_metrics', {}).get('ela_std_dev', 0):.2f}",
                            'Metadata/EXIF Signal': f"{meta_res['score']}% Risk",
                            'Notable Indicators': ", ".join(st.session_state.image_result['reasons']) if st.session_state.image_result['reasons'] else "None"
                        },
                        'key_findings': st.session_state.image_result['reasons'],
                        'ai_explanation': "Technical investigation focused on pixel-level anomalies and metadata forensics."
                    }
                    path = generate_pdf_report(img_report_data)
                    if path:
                        with open(path, "rb") as f:
                            st.download_button("Download Detailed Image PDF", f, file_name=os.path.basename(path))

# ── Tab 3: Audio ──────────────────────────────────────────────────────────────
with tab3:
    st.header("Audio Forensic Analysis (AI-Enhanced)")
    st.write("Pitch monotonicity, MFCC delta smoothness, and energy consistency analysis.")
    audio_file = st.file_uploader("Upload an Audio file", type=["wav", "mp3"])
    if audio_file is not None:
        suffix = os.path.splitext(audio_file.name)[1]
        tmp_aud = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_aud.write(audio_file.getbuffer())
        tmp_aud.close()
        temp_aud_path = tmp_aud.name

        if st.button("Run Audio Analysis"):
            with st.spinner("Scanning for malware..."):
                threat_res = scan_for_threats(temp_aud_path)
                if threat_res['score'] > 0:
                    st.error(f"🚨 THREAT DETECTED: {threat_res['reasons'][0]}")

            with st.spinner("Extracting pitch, MFCC delta, energy patterns..."):
                aud_res = analyze_audio(temp_aud_path)

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
                
                # --- Specific Report Button ---
                st.divider()
                if st.button("📄 Generate Detailed Audio Analysis Report", key="aud_rep"):
                    aud_report_data = {
                        'is_specific': True,
                        'target_modality': 'AUDIO',
                        'final_score': aud_res['score'],
                        'risk_level': 'High' if aud_res['score'] >= 60 else 'Medium' if aud_res['score'] >= 30 else 'Low',
                        'specific_details': {
                            'Pitch Instability (Std Dev)': f"{metrics.get('pitch_std', 0):.1f} Hz",
                            'MFCC Delta Jitter (Smoothness)': f"{metrics.get('mfcc_delta_std', 0):.2f}",
                            'RMS Energy Consistency': f"{metrics.get('rms_std', 0):.4f}",
                            'Spectral Flatness': f"{metrics.get('spectral_flatness', 0):.4f}"
                        },
                        'key_findings': aud_res['reasons'],
                        'ai_explanation': "Acoustic signal analysis detecting synthetic speech patterns and TTS signatures."
                    }
                    path = generate_pdf_report(aud_report_data)
                    if path:
                        with open(path, "rb") as f:
                            st.download_button("Download Detailed Audio PDF", f, file_name=os.path.basename(path))

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
                
                # --- Specific Report Button ---
                st.divider()
                if st.button("📄 Generate Detailed URL Analysis Report", key="url_rep"):
                    url_report_data = {
                        'is_specific': True,
                        'target_modality': 'URL',
                        'final_score': result['score'],
                        'risk_level': result['risk_level'],
                        'specific_details': {
                            'Target URL': url_input,
                            'Domain Entropy': feats.get('domain_entropy', 0),
                            'Subdomain Nesting': feats.get('subdomain_depth', 0),
                            'Security Flags': ", ".join(result['reasons']) if result['reasons'] else "None"
                        },
                        'key_findings': result['reasons'],
                        'ai_explanation': f"URL analysis for {url_input} focusing on homograph and DGA entropy."
                    }
                    path = generate_pdf_report(url_report_data)
                    if path:
                        with open(path, "rb") as f:
                            st.download_button("Download Detailed URL PDF", f, file_name=os.path.basename(path))

                with st.expander("View Full Feature Analysis"):
                    st.json(result['features'])
        else:
            st.error("Please enter a URL first.")

# ── Final Report ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("Results & Reporting")
st.caption("⚙️ Fusion Engine will compute the final verdict using weighted mathematical logic. 🧠 AI Analyst (Qwen2) will then generate the forensic explanation report.")

if st.button("Generate Final Forensic Report (CPU-Optimized Fusion)", type="primary"):
    with st.spinner("Computing Weighted Fusion & Generating LLM Explanation..."):
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
            # Simple caching: Check if evidence is the same as last result
            evidence_hash = str(sorted(all_evidence.keys())) + str(sum([v.get('score', 0) for v in all_evidence.values()])) + str(skip_ai)
            
            if "last_verdict" in st.session_state and st.session_state.get("last_evidence_hash") == evidence_hash:
                verdict = st.session_state.last_verdict
                ai_explanation = verdict.get('ai_explanation', '')
                st.success("⚡ Result retrieved from session cache (Instant).")
            else:
                with st.status("🔍 Forensic Fusion Engine: Computing Tier 2 Verdict...", expanded=True) as status:
                    st.write("Synthesizing multi-modal signals...")
                    verdict = generate_final_verdict_ai(all_evidence, skip_llm=skip_ai, stream=not skip_ai)
                    
                    if not skip_ai:
                        st.write("🧠 AI Analyst Expert Mode (Streaming)...")
                        def generator():
                            for chunk in verdict['ai_explanation']:
                                yield chunk['response']
                        ai_explanation = st.write_stream(generator)
                    else:
                        ai_explanation = verdict.get('ai_explanation', 'AI Bypassed.')
                        st.info(ai_explanation)
                    
                    status.update(label="✅ Analysis Complete", state="complete", expanded=False)
                
                # Update cache
                verdict['ai_explanation'] = ai_explanation
                st.session_state.last_verdict = verdict
                st.session_state.last_evidence_hash = evidence_hash

            st.subheader("⚙️ Computed Risk Breakdown")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🦠 Threat Score", f"{verdict.get('threat_score', 0)}%", help="Security threat detection (Malware, etc).")
            with col2:
                st.metric("🤖 AI-Generated Score", f"{verdict.get('ai_generated_score', 0)}%", help="Is this content fully born from an AI model like Midjourney?")
            with col3:
                st.metric("🎭 Manipulation Score", f"{verdict.get('manipulation_score', 0)}%", help="Has a real file been doctored/edited (deepfake swap, Photoshop)?")
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
                with st.expander("🔍 Mathematical Fusion Overrides & Metrics"):
                    for finding in key_findings:
                        st.markdown(f"- {finding}")

            report_data = {
                'final_score': verdict.get('final_score', 0),
                'risk_level': risk_level,
                'threat_score': verdict.get('threat_score', 0),
                'ai_generated_score': verdict.get('ai_generated_score', 0),
                'manipulation_score': verdict.get('manipulation_score', 0),
                'confidence': confidence,
                'key_findings': key_findings,
                'url_score': st.session_state.url_result['score'] if st.session_state.url_result else 0,
                'image_score': st.session_state.image_result['score'] if st.session_state.image_result else 0,
                'audio_score': st.session_state.audio_result['score'] if st.session_state.audio_result else 0,
                'video_score': st.session_state.video_result['score'] if st.session_state.video_result else 0,
                'modalities': {
                    'URL': "Analyzed" if st.session_state.url_result else "Not Analyzed",
                    'Image': "Analyzed" if st.session_state.image_result else "Not Analyzed",
                    'Audio': "Analyzed" if st.session_state.audio_result else "Not Analyzed",
                    'Video': "Analyzed" if st.session_state.video_result else "Not Analyzed",
                },
                'ai_explanation': ai_explanation
            }

            pdf_path = generate_pdf_report(report_data)
            st.success("Report Generated Successfully!")
            st.info(f"**AI Analyst (Qwen2) Technical Evidence Explanation:**\n\n{ai_explanation}")

            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "Download PDF Report", f,
                        file_name=os.path.basename(pdf_path), mime="application/pdf"
                    )
