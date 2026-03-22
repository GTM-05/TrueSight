from dataclasses import dataclass, field

@dataclass
class ForensicConfig:

    # ── IMAGE ─────────────────────────────────────────────────────────────
    ELA_QUALITY: int = 85
    ELA_MEAN_THRESHOLD: float = 12.0
    ELA_STD_THRESHOLD: float = 18.0
    # FFmpeg / codec recompression elevates ELA on extracted video frames — separate gates
    ELA_MEAN_THRESHOLD_VIDEO: float = 28.0
    ELA_STD_THRESHOLD_VIDEO: float = 50.0

    # SRM — DIFFERENT thresholds for video vs image (H.264 changes stats)
    SRM_CLEAN_STD_IMAGE: float = 7.5    # standalone images
    SRM_CLEAN_STD_VIDEO: float = 2.8    # video frames — H.264 compresses noise
    SRM_KURTOSIS_IMAGE: float = 9.0     # standalone images
    SRM_KURTOSIS_VIDEO: float = 150.0   # H.264 DCT naturally produces kurtosis 40–120

    # Spectral slope — FIX-3: mid-frequency range only, corrected tolerance
    SPECTRAL_SLOPE_CENTER: float = -2.2
    SPECTRAL_SLOPE_TOLERANCE: float = 0.45   # flag if |slope - (-2.2)| > 0.45
    SPECTRAL_SLOPE_FIT_START: float = 0.10   # start fit at 10% of radial profile
    SPECTRAL_SLOPE_FIT_END: float = 0.75     # end fit at 75% — skip DC and noise floor
    SPECTRAL_MIN_IMAGE_SIZE: int = 128       # skip FFT on images smaller than this

    # Chromatic aberration — FIX-4: sub-pixel phase shift in PIXELS, not arbitrary units
    CHROMA_SHIFT_REAL_MIN: float = 0.3       # real lenses: R/B shift > 0.3 px minimum
    CHROMA_SHIFT_REAL_TYPICAL: float = 1.2   # typical camera: 0.3–1.5 px
    CHROMA_ALIGN_SYNTHETIC_MAX: float = 0.6  # AI generators: shift < 0.6 px = perfect
    CHROMA_WARP_SUSPICIOUS: float = 8.0      # extreme shift > 8 px = compositing warp

    NOISE_FLOOR_SYNTHETIC_MAX: float = 1.4
    NOISE_SPATIAL_CV_MAX: float = 0.8
    DCT_GRID_PEAK_RATIO: float = 110.0
    DCT_BOUNDARY_MISMATCH: float = 0.15
    COPY_MOVE_MIN_MATCHES: int = 12
    # Video frames: periodic texture + H.264 raise SIFT self-matches — require more pairs
    COPY_MOVE_MIN_MATCHES_VIDEO: int = 28

    AI_RESOLUTIONS: set = field(default_factory=lambda: {
        (512,512),(768,768),(1024,1024),(1024,576),(576,1024),
        (1280,720),(832,1216),(1216,832),(1344,768),(768,1344),
        (896,1152),(1152,896),(2048,2048),(640,480),
    })
    AI_SOFTWARE_TAGS: tuple = (
        "stable diffusion","midjourney","dall-e","firefly","imagen",
        "nightcafe","dreamstudio","novelai","comfyui","a1111",
        "automatic1111","invoke ai","leonardo","ideogram","flux",
        "generativeai","adobe firefly","bing image creator",
    )

    # ── AUDIO ─────────────────────────────────────────────────────────────
    F0_MIN_HZ: float = 60.0
    F0_MAX_HZ: float = 400.0
    JITTER_SYNTHETIC_MAX: float = 0.012      # below = robotic TTS
    SHIMMER_SYNTHETIC_MAX: float = 0.04
    PITCH_MONOTONE_STD_MAX: float = 15.0     # Hz std — flat = TTS

    SILENCE_RMS_THRESHOLD: float = 0.0008
    SILENCE_NOISE_STD_MAX: float = 0.00015   # true digital zero (not compressed quiet)
    DIGITAL_SILENCE_RATIO_MIN: float = 0.30
    NEAR_SILENT_SKIP_THRESHOLD: float = 0.005  # entire track quiet = real room

    SPECTRAL_FLATNESS_TTS_MIN: float = 0.13
    HNR_SYNTHETIC_MIN: float = 22.0
    HNR_STRONG_THRESHOLD: float = 28.0
    MFCC_ABRUPT_PERCENTILE: float = 97.0
    MFCC_ABRUPT_RATIO_MAX: float = 0.038
    PHASE_DISC_THRESHOLD: float = 1.4        # radians
    PHASE_DISC_MIN_SPIKES: int = 2
    PHASE_DISC_STRONG_SPIKES: int = 3        # is_strong requires >= 3 spikes
    TTS_SAMPLE_RATES: tuple = (22050, 24000, 16000)
    MIN_AUDIO_DURATION: float = 2.0

    # ── VIDEO / LIVENESS — FIX-5: data guards added ───────────────────────
    MIN_LIVENESS_FACES: int = 3              # rPPG skips if fewer faces detected
    MIN_LIVENESS_SIGNALS: int = 15           # rPPG skips if fewer chrominance samples
    RPPG_SNR_MIN: float = 3.0               # FIX-5: raised from 1.5 — reliable pulse
    RPPG_SNR_ANOMALY: float = 1.0           # below this = clear pulse anomaly
    RPPG_BANDPASS_LOW: float = 0.7
    RPPG_BANDPASS_HIGH: float = 4.0
    BLINK_MIN_HUMAN: int = 2
    IRIS_JITTER_MIN: float = 0.5
    SSIM_STD_THRESHOLD: float = 0.08
    OPTICAL_FLOW_WARP_THRESHOLD: float = 2.5
    # Face ROI adjacent-frame SSIM — dual pathology (config-driven, no literals in detectors)
    SSIM_FACE_MIN_PAIRS: int = 3
    SSIM_FACE_RESIZE: int = 128
    SSIM_FACE_STABLE_MEAN_MIN: float = 0.992
    SSIM_FACE_STABLE_STD_MAX: float = 0.012
    SSIM_FACE_VARIABLE_STD_MIN: float = 0.06
    SSIM_FACE_MORPH_STABLE_SCORE: float = 32.0
    SSIM_FACE_MORPH_VARIABLE_SCORE: float = 36.0
    FACE_WARP_MIN_SAMPLES: int = 3
    FACE_WARP_FLOW_WIN_SIZE: int = 21
    FACE_WARP_MEAN_MAG_SCORE: float = 28.0
    FACE_WARP_STD_MAG_SCORE: float = 18.0
    FACE_WARP_FLOW_STD_THRESHOLD: float = 1.15

    # Fusion morphing index (video spatial + audio phase splices + metadata)
    MORPHING_META_SCALE: float = 0.38
    MORPHING_META_CAP: float = 22.0
    MORPHING_PHASE_POINTS_PER_SPIKE: float = 0.09
    MORPHING_PHASE_SCORE_CAP: float = 48.0
    MORPHING_SPATIAL_WEIGHT: float = 1.0

    # ── ViT — FIX-7: raised threshold, requires frame agreement ───────────
    VIT_STRONG_THRESHOLD_IMAGE: float = 45.0
    VIT_STRONG_THRESHOLD_VIDEO: float = 65.0  # FIX-7: was 45, raised to 65
    VIT_EVIDENCE_THRESHOLD_VIDEO: float = 55.0
    VIT_MIN_FRAME_AGREEMENT: float = 0.40    # 40% of frames must exceed threshold

    # ── URL ───────────────────────────────────────────────────────────────
    URL_IP_SCORE: int = 40
    URL_SHORTENER_SCORE: int = 35
    URL_SUSPICIOUS_TLD_SCORE: int = 25
    URL_HOMOGRAPH_SCORE: int = 45
    URL_FUZZY_SPOOF_SCORE: int = 40
    URL_FUZZY_RATIO_MIN: float = 0.80
    URL_NO_HTTPS_SCORE: int = 15
    URL_KEYWORD_SCORE_PER: int = 10
    URL_KEYWORD_MAX: int = 30
    URL_ENTROPY_HIGH: float = 3.8
    URL_ENTROPY_MED: float = 3.2
    URL_ENTROPY_HIGH_SCORE: int = 20
    URL_ENTROPY_MED_SCORE: int = 10
    URL_SUBDOMAIN_DEEP_SCORE: int = 20
    URL_SUBDOMAIN_MED_SCORE: int = 10
    URL_LENGTH_LONG: int = 150
    URL_LENGTH_MED: int = 100
    URL_LENGTH_LONG_SCORE: int = 15
    URL_LENGTH_MED_SCORE: int = 8
    URL_DIGIT_RATIO_MAX: float = 0.35
    URL_DIGIT_SCORE: int = 15
    URL_REDIRECT_PARAM_SCORE: int = 15

    URL_SHORTENERS: set = field(default_factory=lambda: {
        'bit.ly','tinyurl.com','goo.gl','t.co','ow.ly','buff.ly',
        'is.gd','rb.gy','cutt.ly','short.io','rebrand.ly','bl.ink',
        'shorte.st','adf.ly','bc.vc','clck.ru','qps.ru',
    })
    URL_SUSPICIOUS_TLDS: set = field(default_factory=lambda: {
        'tk','ml','ga','cf','gq','xyz','top','click','loan',
        'work','party','review','cricket','science',
    })
    URL_PHISHING_KEYWORDS: tuple = (
        'login','verify','account','secure','update','confirm',
        'banking','paypal','amazon','apple','microsoft','google',
        'ebay','netflix','signin','wallet','password','credential',
        'suspended','unusual','alert','urgent','immediately',
    )
    URL_GOLDEN_DOMAINS: tuple = (
        'google','amazon','paypal','microsoft','facebook','netflix',
        'apple','icloud','gmail','outlook','chase','fidelity',
        'instagram','twitter','linkedin','dropbox','github',
    )
    URL_REDIRECT_PARAMS: tuple = (
        'redirect','url=','goto=','next=','return=','redir=',
        'forward=','target=','link=','out=',
    )

    # ── FUSION — FIX-1, FIX-2, FIX-6 ─────────────────────────────────────
    # FIX-1: No longer using (max*0.9)+(avg*0.1). See engine.py Step 1.
    FUSION_BOOST_MULTIPLIER: float = 0.12   # weak signal boost per detector
    FUSION_BOOST_MAX: float = 25.0          # total boost cap

    # FIX-6: Cross-modal disagreement penalty (NEW)
    CROSS_MODAL_SPREAD_STRONG: float = 45.0  # spread > 45 → penalty up to +25
    CROSS_MODAL_SPREAD_WEAK: float = 25.0    # spread > 25 → smaller penalty
    CROSS_MODAL_PENALTY_MAX: float = 25.0

    # FIX-2: Graduated safety floor (replaces binary 19% cap)
    SAFETY_CAP_SCORE_LIMIT: float = 75.0    # above this → no floor applied
    SAFETY_FLOOR_STRONG: float = 45.0       # strong anchor → floor at 45
    SAFETY_FLOOR_MEDIUM_BASE: float = 19.0
    SAFETY_FLOOR_MEDIUM_MAX: float = 44.0
    SAFETY_FLOOR_RESULT: float = 19.0       # no evidence → cap at 19

    # FIX-8: Liveness reduction factors
    LIVENESS_CONFIRMED_FACTOR: float = 0.10  # full confirmation = 90% reduction
    LIVENESS_PARTIAL_FACTOR: float = 0.50    # partial = 50% reduction
    LIVENESS_MIN_FLOOR: float = 15.0         # never reduce below 15

    LACK_OF_DATA_PENALTY: int = 15

    # ── LLM verdict narrative (Ollama / Qwen) ────────────────────────────
    # qwen2:0.5b is fast but often incoherent; qwen2.5:3b or qwen2:1.5b recommended.
    LLM_VERDICT_MODEL: str = "qwen2.5:3b"
    LLM_VERDICT_NUM_PREDICT: int = 640
    LLM_VERDICT_TEMPERATURE: float = 0.15
    LLM_VERDICT_TOP_P: float = 0.9
    # Minimum length for --check-llm / quality heuristic (not a hard science)
    LLM_VERDICT_MIN_CHARS: int = 180

    # ── VERDICT BANDS ─────────────────────────────────────────────────────
    HIGH_RISK_THRESHOLD: float = 60.0
    MEDIUM_RISK_THRESHOLD: float = 30.0


CFG = ForensicConfig()
