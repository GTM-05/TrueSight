import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.url import analyze_url
from modules.image import analyze_image
from modules.metadata import check_metadata
from modules.audio import analyze_audio
from modules.video import analyze_video
from fusion.engine import generate_final_verdict
from modules.threats import scan_for_threats
from llm.phi2 import generate_ai_explanation
from reports.generator import generate_pdf_report

st.set_page_config(page_title="TrueSight Forensics", layout="wide")

# Initialize session state for tracking scores across tabs
if 'url_result' not in st.session_state:
    st.session_state.url_result = None
if 'image_result' not in st.session_state:
    st.session_state.image_result = None
if 'audio_result' not in st.session_state:
    st.session_state.audio_result = None
if 'video_result' not in st.session_state:
    st.session_state.video_result = None

st.title("🕵️ TrueSight: Multimodal Cyber Forensics")
st.markdown("Analyze **Video, Image, Audio, and URL** inputs to detect deepfake and phishing threats, generate AI-based forensic explanations, and produce structured investigation reports.")

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
                    
            with st.spinner("Processing video frames, extracting audio, and running AI reasoning..."):
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
                
                st.info(f"**Phi-2 AI Video Verdict:**\\n\\n{vid_res['phi_analysis']}")

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
                
                # Save to session state
                st.session_state.image_result = {
                    'score': combined_score,
                    'details': f"Visual Score: {img_res['score']}%, Meta Score: {meta_res['score']}%",
                    'reasons': img_res['reasons'] + meta_res['reasons'],
                    'threats': threat_res
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
                
                # Save to session state
                st.session_state.audio_result = {
                    'score': aud_res['score'],
                    'details': "Audio analyzed via Librosa",
                    'reasons': aud_res['reasons'],
                    'threats': threat_res
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
                
                # Save to session state
                st.session_state.url_result = {
                    'score': result['score'],
                    'details': f"URL: {url_input}",
                    'reasons': result['reasons']
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

if st.button("Generate Final Forensic Report", type="primary"):
    with st.spinner("Running AI Fusion Engine and generating report..."):
        # Gather scores
        u_score = st.session_state.url_result['score'] if st.session_state.url_result else None
        i_score = st.session_state.image_result['score'] if st.session_state.image_result else None
        a_score = st.session_state.audio_result['score'] if st.session_state.audio_result else None
        v_score = st.session_state.video_result['score'] if st.session_state.video_result else None
        
        if u_score is None and i_score is None and a_score is None and v_score is None:
            st.error("No data has been analyzed yet. Please run at least one analysis first.")
        else:
            verdict = generate_final_verdict(u_score, i_score, a_score, v_score)
            
            # Prepare data for LLM
            modality_data = ""
            if st.session_state.url_result:
                modality_data += f"URL Score: {st.session_state.url_result.get('score')}%. Reasons: {st.session_state.url_result.get('reasons')}\\n"
            if st.session_state.image_result:
                modality_data += f"Image Score: {st.session_state.image_result.get('score')}%. Reasons: {st.session_state.image_result.get('reasons')}\\n"
                if st.session_state.image_result.get('threats', {}).get('score', 0) > 0:
                    modality_data += f"Image Threat Found: {st.session_state.image_result['threats']['reasons']}\\n"
            if st.session_state.audio_result:
                modality_data += f"Audio Score: {st.session_state.audio_result.get('score')}%. Reasons: {st.session_state.audio_result.get('reasons')}\\n"
                if st.session_state.audio_result.get('threats', {}).get('score', 0) > 0:
                    modality_data += f"Audio Threat Found: {st.session_state.audio_result['threats']['reasons']}\\n"
            if st.session_state.video_result:
                modality_data += f"Video Score: {st.session_state.video_result.get('score')}%. Reasons: {st.session_state.video_result.get('reasons')}\\n"
                if st.session_state.video_result.get('threats', {}).get('score', 0) > 0:
                    modality_data += f"Video Threat Found: {st.session_state.video_result['threats']['reasons']}\\n"
                
            ai_explanation = generate_ai_explanation(modality_data)
            
            report_data = {
                'final_score': verdict['final_score'],
                'risk_level': verdict['risk_level'],
                'modalities': {
                    'URL': f"Analyzed ({u_score}%)" if u_score is not None else "Not Analyzed",
                    'Image': f"Analyzed ({i_score}%)" if i_score is not None else "Not Analyzed",
                    'Audio': f"Analyzed ({a_score}%)" if a_score is not None else "Not Analyzed",
                    'Video': f"Analyzed ({v_score}%)" if v_score is not None else "Not Analyzed",
                },
                'ai_explanation': ai_explanation
            }
            
            pdf_path = generate_pdf_report(report_data)
            
            st.success("Report Generated Successfully!")
            st.write(f"**Final Risk Level:** {verdict['risk_level']} ({verdict['final_score']}%)")
            st.info(f"**AI Investigator Explanation:**\n\n{ai_explanation}")
            
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF Report", f, file_name="Forensic_Report.pdf", mime="application/pdf")
