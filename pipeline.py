# pipeline.py
# NLP Feature Extraction Pipeline
# Cognitive Drift and Academic Stress Detection
# KDU Research — NTPGB Upethra D/BCS/23/0021
#
# Models used:
#
# Emotion Detection:
# j-hartmann/emotion-english-distilroberta-base
# Hartmann et al. (2022)
# Trained on 6 diverse datasets including
# social media text
# Returns 7 emotion labels: anger, disgust,
# fear, joy, neutral, sadness, surprise
# Fear and sadness map to stress indicators
#
# Zero Shot Classification:
# facebook/bart-large-mnli
# Lewis et al. (2020)
# Validated for mental health symptom
# classification in published research
#
# Absolutist Term Detection:
# Rule based word list
# Al-Mosaiwi and Johnstone (2018)
# Validated marker for anxiety and depression
#
# Negative Emotion Lexicon:
# Based on NRC Word Emotion Lexicon approach
# Mohammad and Turney (2013)

import os
import spacy
import re
import io
import wave
import numpy as np
from transformers import pipeline as hf_pipeline

# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────

print("Loading NLP models...")

nlp = spacy.load("en_core_web_sm") # Load spaCy English model for tokenization and lemmatization

# Emotion detection model
# Trained on social media text which is closer to student self-talk domain, No login required
emotion_model = hf_pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None, # return all 7 emotions with their scores
    truncation=True, # cut text in inputs if it's too long
    max_length=512
)

# Zero shot classification
# Validated for mental health NLP tasks
zero_shot_model = hf_pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

print("Models loaded successfully")

# ─────────────────────────────────────────────
# Validated word lists
# ─────────────────────────────────────────────

# Both lists now live in lexicons.py so that explanation.py can
# reuse them for evidence extraction without importing this module
# (which would load the transformer models as a side effect).
# The contents are unchanged apart from removing a duplicate
# "nothing" entry, which was causing that one word to be counted
# twice by get_absolutist_term_count().
from lexicons import (
    ABSOLUTIST_TERMS,
    NEGATIVE_EMOTION_TERMS,
    SHOULD_STATEMENT_TERMS,
    PERSONALIZATION_TERMS,
    FILLER_TERMS,
)

# ─────────────────────────────────────────────
# SENTIMENT SCORE
# Using emotion detection model
# ─────────────────────────────────────────────

# "legacy": (positive - negative) / (positive + negative), the original formula.
# "confidence": neutral share kept in the denominator (see _sentence_sentiment).
# Switch with the environment variable SENTIMENT_MODE. Default stays "legacy"
# until sentiment_check.py shows the other mode is better on your data.
SENTIMENT_MODE = os.environ.get("SENTIMENT_MODE", "legacy").lower()


def _sentence_sentiment(text):
    """Sentiment of ONE short piece of text (-1.0 to +1.0).
    Same formula as before: weighted negative emotions against positive
    ones, ignoring the 'neutral' label. Kept for single sentences only,
    which is the length the emotion model was trained on."""
    results = emotion_model(text)[0]
    emotion_scores = {item["label"]: item["score"] for item in results}
    negative = (
        emotion_scores.get("fear", 0) * 1.0 +
        emotion_scores.get("sadness", 0) * 1.0 +
        emotion_scores.get("disgust", 0) * 0.7 +
        emotion_scores.get("anger", 0) * 0.7
    )
    positive = (
        emotion_scores.get("joy", 0) * 1.0 +
        emotion_scores.get("surprise", 0) * 0.3
    )
    total = negative + positive
    if total == 0:
        return 0.0
    if SENTIMENT_MODE == "confidence":
        # Keep the neutral share in the denominator, so a sentence the model
        # calls mostly neutral scores near 0 instead of being pushed to -1
        # by a small leftover of fear or sadness.
        return (positive - negative) / (total + emotion_scores.get("neutral", 0))
    return (positive - negative) / total


def get_sentiment_score(text):
    """
    Returns float between -1.0 and +1.0.

    The emotion model (Hartmann et al., 2022) was trained on short social
    media posts. Given a long paragraph it can misjudge the overall emotion:
    a calm, reflective 9-sentence stress paragraph was scored 61% joy and
    came out positive. So the text is scored sentence by sentence and the
    sentence scores are averaged. One-sentence input gives exactly the
    same result as before.
    """
    try:
        doc = nlp(text)
        sentences = [x.text.strip() for x in doc.sents if len(x.text.strip()) > 3]
        if not sentences:
            return 0.0
        scores = [_sentence_sentiment(x) for x in sentences]
        return round(sum(scores) / len(scores), 4)
    except Exception as e:
        print(f"Emotion model error: {e}")
        return 0.0


def get_emotion_scores(text):
    """
    Returns individual emotion scores
    Used for richer feature extraction
    beyond simple sentiment polarity
    """
    try:
        results = emotion_model(text)[0]
        return {
            item["label"]: round(item["score"], 4)
            for item in results
        }
    except Exception as e:
        print(f"Emotion scores error: {e}")
        return {}

# ─────────────────────────────────────────────
# NEGATIVE LANGUAGE SCORE
# ─────────────────────────────────────────────

def get_negative_language_score(text):
    """
    Returns float between 0.0 and 1.0

    Based on the NRC Word Emotion Lexicon approach (Mohammad and
    Turney, 2013), extended with common study-stress words.

    Rule set v2 change: the score now counts every occurrence of a
    listed term (word-boundary match, longest term first so "stressed"
    is not also counted as "stress") and scales by 12. The old version
    counted each listed word once and scaled by 5, so even strongly
    distressed paragraphs scored 0.05 to 0.22 and R01 / R02 (which need
    above 0.6 / 0.5) could not fire on ordinary writing. The factor 12
    is provisional and should be calibrated on the evaluation sessions.
    """
    text_lower = text.lower()
    doc = nlp(text_lower)
    tokens = [token.lemma_ for token in doc if not token.is_punct]

    if len(tokens) == 0:
        return 0.0

    negative_count = len(_NEGATIVE_TERM_PATTERN.findall(text_lower))
    score = min(negative_count / max(len(tokens), 1) * 12, 1.0)
    return round(score, 4)

_NEGATIVE_TERM_PATTERN = re.compile(
    r"\b(?:" + "|".join(
        re.escape(term)
        for term in sorted(set(NEGATIVE_EMOTION_TERMS), key=len, reverse=True)
    ) + r")\w*"
)

# ─────────────────────────────────────────────
# ABSOLUTIST TERM COUNT
# ─────────────────────────────────────────────

def get_absolutist_term_count(text):
    """
    Returns integer count of absolutist terms
    Based on validated word list from
    Al-Mosaiwi and Johnstone (2018)
    Elevated absolutist term use is a validated
    marker specific to anxiety and depression
    """
    text_lower = text.lower()
    count = sum(
        1 for term in ABSOLUTIST_TERMS
        if re.search(
            r'\b' + re.escape(term) + r'\b',
            text_lower
        )
    )
    return count

# ─────────────────────────────────────────────
# SHOULD STATEMENT COUNT
# ─────────────────────────────────────────────

def get_should_statement_count(text):
    """
    Returns integer count of rigid-obligation terms
    ("should", "must", "have to", ...), matched the
    same word-boundary way as get_absolutist_term_count().
    Feeds R32_ShouldStatements, which also requires
    negative sentiment so ordinary planning language
    ("I should study more") does not trigger it alone.
    """
    text_lower = text.lower()
    count = sum(
        1 for term in SHOULD_STATEMENT_TERMS
        if re.search(
            r'\b' + re.escape(term) + r'\b',
            text_lower
        )
    )
    return count

# ─────────────────────────────────────────────
# PERSONALIZATION COUNT
# ─────────────────────────────────────────────

def get_personalization_count(text):
    """
    Returns integer count of self-blame phrases
    ("my fault", "i ruined", ...). Matched by
    substring search since these are multi-word
    phrases, the same approach as
    get_negative_language_score() uses for its terms.
    Feeds R33_Personalization.
    """
    text_lower = text.lower()
    count = sum(
        1 for term in PERSONALIZATION_TERMS
        if term in text_lower
    )
    return count

# ─────────────────────────────────────────────
# ZERO SHOT COGNITIVE DISTORTION SCORES
# ─────────────────────────────────────────────

def get_zero_shot_scores(text):
    """
    Returns scores for helplessness,
    catastrophizing, and overgeneralization
    using zero shot classification.
    CBT-grounded labels based on
    Beck (1979) cognitive distortion taxonomy.
    Lewis et al. (2020) BART model.
    Validated for mental health NLP in
    published research.
    """
    labels = [
        "helplessness and hopelessness about studies",
        "catastrophizing and expecting the worst outcome",
        "overgeneralization and all or nothing thinking",
        "neutral academic statement"
    ]

    try:
        result = zero_shot_model(
            text,
            candidate_labels=labels,
            multi_label=True
        )

        scores = dict(zip(
            result["labels"],
            result["scores"]
        ))

        return {
            "helplessness_score": round(
                scores.get(
                    "helplessness and hopelessness about studies",
                    0.0
                ), 4
            ),
            "catastrophizing_score": round(
                scores.get(
                    "catastrophizing and expecting the worst outcome",
                    0.0
                ), 4
            ),
            "overgeneralization_score": round(
                scores.get(
                    "overgeneralization and all or nothing thinking",
                    0.0
                ), 4
            )
        }

    except Exception as e:
        print(f"Zero shot error: {e}")
        return {
            "helplessness_score": 0.0,
            "catastrophizing_score": 0.0,
            "overgeneralization_score": 0.0
        }

# ─────────────────────────────────────────────
# SUBJECT SPECIFIC DETECTION
# ─────────────────────────────────────────────

def is_subject_specific(text):
    """
    Returns True if text refers to a specific
    academic subject, task, or assessment
    """
    subject_keywords = [
        "math", "maths", "mathematics", "physics",
        "chemistry", "biology", "programming", "code",
        "coding", "algorithm", "data structure",
        "database", "network", "assignment", "exam",
        "test", "quiz", "project", "thesis", "essay",
        "report", "lecture", "module", "subject",
        "course", "lab", "practical", "tutorial",
        "computing", "engineering", "science",
        "software", "hardware", "circuit", "calculus",
        "statistics", "research", "dissertation"
    ]
    text_lower = text.lower()
    return any(
        kw in text_lower
        for kw in subject_keywords
    )

# ─────────────────────────────────────────────
# CONFIDENCE FILTER
# ─────────────────────────────────────────────

def apply_confidence_filter(features):
    """
    Suppresses distortion scores when overall
    sentiment is not negative.
    Prevents false positive inferences for
    neutral or positive student text.
    Only assert distortion scores when
    student text is clearly negative.
    """
    is_negative = features["sentiment_score"] < -0.3

    if not is_negative:
        features["helplessness_score"] = 0.0
        features["catastrophizing_score"] = 0.0
        features["overgeneralization_score"] = 0.0

    return features

# ─────────────────────────────────────────────
# SPEECH FEATURE EXTRACTION
#
# Previously a stub: speech_pace and hesitation_count were supplied
# by hand rather than measured, so the system was bimodal in
# practice despite claiming three modalities (HANDOVER.md open
# issue 2). This section takes a real recording — raw WAV bytes,
# as returned by Streamlit's st.audio_input widget — and derives
# a transcript, pace, hesitation count, and a coarse tone signal
# from it directly.
#
# No new heavy dependency is added. st.audio_input already
# produces a standard PCM WAV container, so Python's built-in
# `wave` module reads it without ffmpeg. Resampling to the 16kHz
# Whisper expects is done with a simple numpy linear interpolation
# rather than pulling in librosa/torchaudio — adequate for speech
# recognition, not broadcast-quality, and documented here as a
# scope decision rather than left silent.
# ─────────────────────────────────────────────

# Loaded lazily on first speech submission, not at module import
# like the other three models. A text-only self-talk analysis
# should not pay the cost of loading a fourth transformer model
# it will never use.
_asr_model = None


def _get_asr_model():
    global _asr_model
    if _asr_model is None:
        print("Loading speech-to-text model (first use only)...")
        _asr_model = hf_pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base.en",
        )
    return _asr_model


def _decode_wav_bytes(audio_bytes):
    """
    Decode a WAV byte string into (samples, framerate, duration_seconds).
    `samples` is a mono float32 numpy array scaled to [-1.0, 1.0].
    Handles 8/16/32-bit PCM and mixes multi-channel audio down to mono.
    """
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError("Unsupported WAV sample width: %d bytes" % sample_width)

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    duration_seconds = (n_frames / float(framerate)) if framerate else 0.0
    return samples.astype(np.float32), framerate, duration_seconds


def _resample_to_16k(samples, framerate, target=16000):
    """
    Simple linear-interpolation resample to the sample rate Whisper
    expects. Not sinc-quality, but sufficient for speech recognition
    and avoids adding librosa/torchaudio as a dependency.
    """
    if framerate == target or len(samples) == 0:
        return samples, framerate
    duration = len(samples) / framerate
    target_len = max(int(round(duration * target)), 1)
    x_old = np.linspace(0, duration, num=len(samples), endpoint=False)
    x_new = np.linspace(0, duration, num=target_len, endpoint=False)
    resampled = np.interp(x_new, x_old, samples).astype(np.float32)
    return resampled, target


def _trim_silence(samples, framerate, threshold_ratio=0.03, pad_seconds=0.15,
                   frame_ms=20, absolute_silence_peak=0.02):
    """
    Trims leading/trailing near-silence from a recording before it
    reaches the ASR model.

    Root-cause fix for an observed failure mode: Whisper hallucinates
    repeated boilerplate ("I'm going to stop at this time." looped
    ~100 times) when fed a long stretch of trailing silence after
    the speaker has actually finished — which is exactly what a
    browser mic recording usually has, since people don't cut the
    recording the instant they stop talking. Trimming that silence
    out before transcription removes the input that triggers the
    loop in the first place. get_hesitation_count() and
    _collapse_repetition() only exist as reactive cleanup for
    whatever this doesn't catch.

    `absolute_silence_peak` catches a clip that is uniformly quiet
    throughout (e.g. an empty/muted recording, or just mic noise
    floor) — the relative threshold alone can't detect that case,
    since with no louder segment to compare against, every frame
    looks "voiced" relative to a peak that is itself just noise.

    Returns (trimmed_samples, is_effectively_silent).
    """
    if len(samples) == 0:
        return samples, True

    frame_len = max(int(framerate * frame_ms / 1000), 1)
    n_frames = len(samples) // frame_len
    if n_frames < 2:
        return samples, bool(np.max(np.abs(samples)) < 1e-4)

    frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    peak = frame_rms.max()
    if peak <= 1e-6 or peak < absolute_silence_peak:
        return samples, True

    voiced = frame_rms > (peak * threshold_ratio)
    if not voiced.any():
        return samples, True

    first_voiced = int(np.argmax(voiced))
    last_voiced = len(voiced) - 1 - int(np.argmax(voiced[::-1]))

    pad_frames = max(int(pad_seconds * 1000 / frame_ms), 0)
    start = max(first_voiced - pad_frames, 0) * frame_len
    end = min((last_voiced + pad_frames + 1) * frame_len, len(samples))

    return samples[start:end], False


def _collapse_repetition(text, max_repeats=2):
    """
    Defensive cleanup for residual ASR repetition-loop hallucination
    that _trim_silence() doesn't catch (e.g. the model looping on a
    genuinely spoken phrase, not just trailing silence). Collapses a
    sentence that repeats back-to-back more than `max_repeats` times
    down to a single run of that length, rather than a full n-gram
    dedup algorithm — enough to fix the pattern actually observed
    ("I'm going to stop at this time." repeated ~100 times) without
    over-engineering the general case.
    """
    if not text:
        return text
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(parts) < max_repeats + 1:
        return text

    collapsed = [parts[0]]
    repeat_run = 1
    for part in parts[1:]:
        if part.strip().lower() == collapsed[-1].strip().lower():
            repeat_run += 1
            if repeat_run <= max_repeats:
                collapsed.append(part)
            # else: this is excess repetition beyond max_repeats - drop it
        else:
            repeat_run = 1
            collapsed.append(part)
    return " ".join(collapsed)


def transcribe_audio(samples, framerate):
    """
    Runs the ASR model on already-resampled mono float32 samples.
    Returns an empty string on any failure rather than raising, so a
    bad recording degrades to "no speech modality" instead of
    crashing the analysis.
    """
    try:
        asr = _get_asr_model()
        result = asr({"array": samples, "sampling_rate": framerate})
        return (result.get("text") or "").strip()
    except Exception as e:
        print(f"ASR error: {e}")
        return ""


def compute_speech_pace(word_count, duration_seconds):
    """
    Returns "Slow" / "Normal" / "Fast" from words-per-minute.
    Thresholds (110 / 165 wpm) are a documented assumption based on
    commonly cited conversational speech-rate ranges, not validated
    against a normative corpus for this population — flagged the
    same way as the other constants in HANDOVER.md open issue 4.
    Consumed by R08_SlowedThinking and R20_ModerateDrift_Speech.
    """
    if duration_seconds <= 0:
        return "Normal"
    wpm = (word_count / duration_seconds) * 60.0
    if wpm < 110:
        return "Slow"
    if wpm > 165:
        return "Fast"
    return "Normal"


def get_hesitation_count(transcript):
    """
    Returns the total number of filler/disfluency *occurrences* in
    the transcript (e.g. "um" said three times counts as 3).
    This differs deliberately from get_absolutist_term_count(),
    which counts one hit per list entry regardless of repetition —
    here, repetition itself is the hesitation signal R20 checks
    for (hesitationCount > 3), not lexical variety.
    """
    text_lower = transcript.lower()
    count = 0
    for term in FILLER_TERMS:
        count += len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
    return count


def compute_tone_indicator(samples, framerate):
    """
    Coarse prosodic-flatness proxy: splits the recording into 50ms
    frames, takes RMS energy per frame, and looks at the relative
    variability of that energy across voiced frames. Low variability
    is reported as "Monotone".

    This is a simple energy-based heuristic, not real pitch/prosody
    analysis (that would need a proper acoustic library) — an
    honest first step rather than leaving tone entirely unmeasured.
    It is currently descriptive only: no SWRL rule consumes
    speech_hasToneIndicator yet, matching HANDOVER.md's list of
    unreachable signals.
    """
    if len(samples) == 0 or framerate <= 0:
        return "Normal"
    frame_len = max(int(framerate * 0.05), 1)
    n_frames = len(samples) // frame_len
    if n_frames < 4:
        return "Normal"
    frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    if rms.max() <= 0:
        return "Normal"
    voiced = rms[rms > (rms.max() * 0.1)]
    if len(voiced) < 4:
        return "Normal"
    variability = float(np.std(voiced) / (np.mean(voiced) + 1e-8))
    return "Monotone" if variability < 0.25 else "Normal"


def extract_speech_features(audio_bytes):
    """
    Main speech entry point — takes raw WAV bytes from a recording
    and returns a dict matching the speech_features contract in
    HANDOVER.md section 4.1: transcribed_text, speech_pace,
    tone_indicator, hesitation_count, plus the same six NLP scores
    used for self-talk text (computed here on the transcript, via
    the same functions, so text and speech are scored identically).

    Returns None if the recording is unusable (unreadable, or too
    short to be a meaningful utterance) — callers should treat that
    the same as "no speech provided".
    """
    try:
        samples, framerate, duration = _decode_wav_bytes(audio_bytes)
    except Exception as e:
        print(f"WAV decode error: {e}")
        return None

    if duration < 0.3:
        return None

    # Trim trailing/leading silence BEFORE transcription - see
    # _trim_silence()'s docstring for why this matters (it is the
    # actual fix for the Whisper repetition-loop hallucination, not
    # just a nicety).
    trimmed_samples, is_silent = _trim_silence(samples, framerate)
    if is_silent or (len(trimmed_samples) / framerate) < 0.3:
        # Nothing resembling speech was recorded - treat the same as
        # "no recording provided" rather than transcribing near-silence,
        # which is exactly the input that triggers ASR hallucination.
        return None

    speaking_duration = len(trimmed_samples) / framerate
    samples_16k, sr_16k = _resample_to_16k(trimmed_samples, framerate)
    transcript = transcribe_audio(samples_16k, sr_16k)
    transcript = _collapse_repetition(transcript)

    # Pace is words-per-minute over actual speaking time, not the
    # full (possibly silence-padded) recording length.
    pace = compute_speech_pace(len(transcript.split()), speaking_duration)
    hesitation = get_hesitation_count(transcript)
    tone = compute_tone_indicator(trimmed_samples, framerate)

    if transcript:
        zero_shot = get_zero_shot_scores(transcript)
        speech_scores = {
            "sentiment_score": get_sentiment_score(transcript),
            "negative_language_score": get_negative_language_score(transcript),
            "absolutist_term_count": get_absolutist_term_count(transcript),
            "helplessness_score": zero_shot["helplessness_score"],
            "catastrophizing_score": zero_shot["catastrophizing_score"],
            "overgeneralization_score": zero_shot["overgeneralization_score"],
        }
        # Same confidence-filter policy applied to self-talk text,
        # so speech and text distortion scores follow one rule.
        speech_scores = apply_confidence_filter(speech_scores)
    else:
        speech_scores = {
            "sentiment_score": 0.0,
            "negative_language_score": 0.0,
            "absolutist_term_count": 0,
            "helplessness_score": 0.0,
            "catastrophizing_score": 0.0,
            "overgeneralization_score": 0.0,
        }

    return {
        "transcribed_text": transcript,
        "speech_pace": pace,
        "tone_indicator": tone,
        "hesitation_count": hesitation,
        **speech_scores,
    }


# ─────────────────────────────────────────────
# MAIN EXTRACTION FUNCTION
# ─────────────────────────────────────────────

def get_stress_statement_count(text):
    """Number of plain stress/strain words or phrases (stress, overwhelmed, anxious,
    pressure, too much ...). Counted, not scored: it feeds the stress evidence score."""
    from lexicons import STRESS_STATEMENT_PATTERN
    return len(re.findall(STRESS_STATEMENT_PATTERN, text or "", flags=re.I))


def extract_features(text):
    """
    Main function — takes raw student text
    Returns all NLP feature scores as dictionary
    ready to be written into the ontology

    Returns dict with:
        raw_text: str
        sentiment_score: float (-1.0 to +1.0)
        negative_language_score: float (0.0 to 1.0)
        absolutist_term_count: int
        should_statement_count: int
        personalization_count: int
        helplessness_score: float (0.0 to 1.0)
        catastrophizing_score: float (0.0 to 1.0)
        overgeneralization_score: float (0.0 to 1.0)
        subject_specific: bool
        emotion_scores: dict of 7 emotion labels
    """
    zero_shot_scores = get_zero_shot_scores(text)
    sentiment = get_sentiment_score(text)
    emotions = get_emotion_scores(text)

    features = {
        "raw_text": text,
        "sentiment_score": sentiment,
        "negative_language_score":
            get_negative_language_score(text),
        "absolutist_term_count":
            get_absolutist_term_count(text),
        "should_statement_count":
            get_should_statement_count(text),
        "stress_statement_count":
            get_stress_statement_count(text),
        "personalization_count":
            get_personalization_count(text),
        "helplessness_score":
            zero_shot_scores["helplessness_score"],
        "catastrophizing_score":
            zero_shot_scores["catastrophizing_score"],
        "overgeneralization_score":
            zero_shot_scores["overgeneralization_score"],
        "subject_specific":
            is_subject_specific(text),
        "emotion_scores": emotions
    }

    features = apply_confidence_filter(features)
    return features

# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    test_cases = [
        {
            "label": "High stress — SevereDrift expected",
            "text": """
            I always fail everything. I can never understand this material no matter how
            hard I try. My mind goes completely blank in exams. I am totally useless at this.
            There is no point continuing. This is a disaster and I cannot handle it anymore.
            """
        },
        {
            "label": "Mild stress — MildStress expected",
            "text": """
            I am a bit worried about my exam tomorrow.
            I hope I studied enough. I feel a little
            nervous but I think I can manage it.
            """
        },
        {
            "label": "No stress — No indicators expected",
            "text": """
            Today I finished my assignment and I feel
            good about it. I understood the concepts
            well and I am ready for the next topic.
            """
        },
        {
            "label": "Attention and memory — EarlyDrift expected",
            "text": """
            I keep zoning out during lectures and
            I cannot focus at all. I read the same
            page five times and still cannot remember
            anything. My mind just goes blank every
            time I try to study.
            """
        }
    ]

    for case in test_cases:
        print(f"\n{'='*55}")
        print(f"TEST: {case['label']}")
        print(f"{'='*55}")
        text = case["text"].strip()
        print(f"Input: {text[:80]}...")
        print(f"{'-'*55}")

        features = extract_features(case["text"])

        print(f"Sentiment Score:          "
              f"{features['sentiment_score']}")
        print(f"Negative Language Score:  "
              f"{features['negative_language_score']}")
        print(f"Absolutist Term Count:    "
              f"{features['absolutist_term_count']}")
        print(f"Helplessness Score:       "
              f"{features['helplessness_score']}")
        print(f"Catastrophizing Score:    "
              f"{features['catastrophizing_score']}")
        print(f"Overgeneralization Score: "
              f"{features['overgeneralization_score']}")
        print(f"Subject Specific:         "
              f"{features['subject_specific']}")

        print(f"\nEmotion breakdown:")
        for emotion, score in sorted(
            features["emotion_scores"].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            bar = "█" * int(score * 20)
            print(f"  {emotion:10} {bar} {score:.4f}")

        print(f"\nRules that would fire:")
        s = features["sentiment_score"]
        nl = features["negative_language_score"]
        ab = features["absolutist_term_count"]
        hl = features["helplessness_score"]
        cs = features["catastrophizing_score"]
        og = features["overgeneralization_score"]

        rules_fired = []
        if nl > 0.6 and ab > 2:
            rules_fired.append("R01 ExamAnxiety")
        if cs > 0.65 and s < -0.6:
            rules_fired.append("R10 Catastrophizing")
        if ab > 3 and s < -0.5:
            rules_fired.append("R09 AllOrNothingThinking")
        if og > 0.6 and ab > 2:
            rules_fired.append("R11 Overgeneralization")
        if s < -0.7 and hl > 0.6:
            rules_fired.append("R12 EmotionalReasoning")
        if hl > 0.7:
            rules_fired.append(
                "R04 HelplessnessAboutWorkload"
            )

        if rules_fired:
            for rule in rules_fired:
                print(f"  → {rule}")
        else:
            print(f"  → No text rules fire")
            print(f"     Behavioral checklist may"
                  f" still trigger rules")