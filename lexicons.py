# lexicons.py
# Shared validated word lists
# Cognitive Drift and Academic Stress Detection
# KDU Research - NTPGB Upethra D/BCS/23/0021
#
# These lists were previously defined inside pipeline.py. They now
# live here because explanation.py needs them to pull evidence
# phrases out of the student's own words, and importing pipeline.py
# would load the transformer models as a side effect.
#
# pipeline.py imports them from here, so there is still exactly one
# copy of each list.

# Source: Al-Mosaiwi and Johnstone (2018)
# Peer reviewed validated list. Elevated use of absolutist terms is
# a marker specific to anxiety and depression - more so than
# negative-emotion words alone.
#
# NOTE: "nothing" appeared twice in the original list. Because
# get_absolutist_term_count() counts one per list entry, any text
# containing "nothing" was scored 2 instead of 1, which could push
# a student over the "> 2" and "> 3" thresholds in R01, R09 and
# R11 on a single word. The duplicate has been removed.
ABSOLUTIST_TERMS = [
    "always", "never", "everything", "nothing",
    "everyone", "no one", "completely", "totally",
    "absolutely", "forever", "impossible", "perfect",
    "terrible", "horrible", "worst", "all", "none",
    "every", "constantly", "entirely", "utterly",
    "nobody", "everybody",
]

# Source: based on the NRC Word Emotion Lexicon
# Mohammad and Turney (2013)
NEGATIVE_EMOTION_TERMS = [
    "anxious", "anxiety", "stressed", "stress",
    "worried", "worry", "scared", "fear", "panic",
    "nervous", "overwhelmed", "exhausted", "tired",
    "depressed", "sad", "unhappy", "miserable",
    "frustrated", "angry", "confused", "lost",
    "hopeless", "desperate", "burnout", "burnt out",
    "dread", "uneasy", "tense", "apprehensive",
    "distressed", "helpless", "worthless", "useless",
    "drained", "struggle", "struggling", "pressure", "falling apart",
    "falling behind", "fall behind", "failure", "failed", "fail",
    "cope", "coping", "guilty", "exhausting", "burden", "stuck",
    "pointless", "give up", "disappoint", "ashamed", "afraid", "upset",
    "irritated", "restless", "awful", "terrible", "horrible",
]

# Beck (1979) cognitive distortion taxonomy - "should statements":
# rigid, self-imposed obligation language. Not an externally validated
# word list like the two above (no published lexicon for this
# distortion was found) - a heuristic list kept deliberately narrow
# to common rigid-obligation phrasing, since single words like "must"
# and "should" are frequent in ordinary planning language and only
# become a distortion signal in volume plus negative sentiment. Feeds
# R32_ShouldStatements, which also requires selfTalk_hasSentimentScore
# < -0.4 so ordinary use ("I should study more") does not fire alone.
SHOULD_STATEMENT_TERMS = [
    "should", "shouldn't", "should have", "shouldn't have",
    "must", "mustn't", "have to", "has to", "had to",
    "ought to", "supposed to",
]

# Beck (1979) cognitive distortion taxonomy - "personalization":
# excessive self-blame for events not entirely within the student's
# control. Heuristic multi-word phrase list, matched the same way as
# NEGATIVE_EMOTION_TERMS (substring, not single-word). Feeds
# R33_Personalization, paired with selfTalk_hasSentimentScore < -0.4.
PERSONALIZATION_TERMS = [
    "my fault", "all my fault", "because of me",
    "i ruined", "i failed everyone", "i let everyone down",
    "i let them down", "i caused this", "i'm the reason",
    "i am the reason", "i messed up", "i screwed up",
    "i should have known better",
]

# Heuristic filler/disfluency list for speech transcripts, used by
# pipeline.get_hesitation_count(). Not an externally validated
# list. Deliberately excludes "like" despite it being a common
# filler word in speech-disfluency research, because it is also an
# extremely common ordinary verb/preposition ("I like this
# module") and would produce far too many false positives when
# matched on transcribed text alone without prosodic timing data.
FILLER_TERMS = [
    "um", "umm", "ummm", "uh", "uhh", "erm", "er",
    "you know", "i mean", "hmm",
]


# Added in calibration round 1 (v5). Plain statements of stress or strain.
# Word stems/phrases a student uses to say they are stressed, overwhelmed, etc.
# Chosen from the wording of the pilot data; provisional and pending supervisor review.
STRESS_STATEMENT_PATTERN = (
    r"\b(stress\w*|overwhelm\w*|anxi\w*|pressur\w*|panic\w*|worr\w*|exhaust\w*|depress\w*|"
    r"fed up|burn\w*|struggl\w*|hopeless|give up|too much|so much|rush\w*|nervous)\b"
)
