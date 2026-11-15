# explanation.py#
# It turns the reasoner's output into three things:
#   1. a supportive, non-clinical message for the student
#   2. a structured report for an academic counsellor
#   3. evidence phrases quoted from the student's own words and writes all of it back into the ontology as an Explanation individual

# DESIGN RULES
# 1. No inference happens here. Every claim traces back to a SWRL rule that Pellet entailed and that reasoner.replay_rule()
#    confirmed. This module only chooses words.
# 2. Nothing is diagnostic. The vocabulary below describes study  behaviour and language patterns, never a person and never a
#    condition. explanation_isClinicallyDiagnostic is asserted. False on every Explanation individual.
# 3. Evidence is the student's own text. Phrases are quoted, not paraphrased, so the student can see exactly what the system  responded to.

import re

from lexicons import ABSOLUTIST_TERMS, NEGATIVE_EMOTION_TERMS
from reasoner import entity

# VOCABULARY
# Indicator -> how it is described to a student (second person, about the study session) and to a counsellor (observational).
INDICATOR_WORDS = {
    # Academic stress indicators
    "ExamAnxiety_Ind": (
        "worry that is centred on an upcoming assessment", "Assessment-focused worry in self-talk",
    ),
    "DeadlinePressure_Ind": (
        "pressure tied to a specific deadline", "Negative language linked to a named upcoming event",
    ),
    "PerceivedOverload_Ind": (
        "a workload that is feeling larger than it is manageable", "High reported workload combined with helplessness language",
    ),
    "HelplessnessAboutWorkload_Ind": (
        "a sense of being stuck about where to start", "Helplessness language together with reported task postponement",
    ),
    "PhysicalStressComplaint_Ind": (
        "physical signs such as headaches or disturbed sleep", "Self-reported headache and sleep difficulty in the same session",
    ),
    "SocialAcademicPressure_Ind": (
        "comparison with how others seem to be doing", "Peer-comparison content in self-talk",
    ),
    # Cognitive drift indicators
    "AttentionLoss_Ind": (
        "difficulty holding your attention while studying", "Self-reported focus difficulty and zoning out",
    ),
    "WorkingMemoryLapse_Ind": (
        "needing to re-read and losing the thread of what you read", "Self-reported re-reading and mind going blank",
    ),
    "SlowedThinking_Ind": (
        "thinking that feels slower than it usually does", "Self-reported slowed thinking and difficulty thinking clearly",
    ),
    "WidespreadStrain_Ind": (
        "difficulties showing up in several areas of your study life at once", "Self-reported difficulties in three or more of the five checklist groups",
    ),
    "AllOrNothingThinking_Ind": (
        "all-or-nothing wording such as always or never", "Elevated absolutist term use with strongly negative sentiment",
    ),
    "Catastrophizing_Ind": (
        "expecting the worst possible outcome", "Worst-case framing with strongly negative sentiment",
    ),
    "Overgeneralization_Ind": (
        "drawing a general conclusion from one or two setbacks", "Overgeneralisation with elevated absolutist term use",
    ),
    "EmotionalReasoning_Ind": (
        "treating how something feels as proof of how it is", "Very negative sentiment together with helplessness language",
    ),
    "MentalFilter_Ind": (
        "attention settling on what went wrong rather than what went well", "Selective negative focus in self-talk",
    ),
    "Personalization_Ind": (
        "taking responsibility for things outside your control", "Self-attribution of external outcomes",
    ),
    "ShouldStatements_Ind": (
        "rules about what you should be doing", "Prescriptive self-directed language",
    ),
    # Added in rule set v2 (behavioural checklist items that no rule used before)
    "MentalExhaustion_Ind": (
        "feeling mentally worn out, together with restlessness or trouble staying focused", "Self-reported mental exhaustion with restlessness or distractibility (B7 with B15 or B2)",
    ),
    "Disengagement_Ind": (
        "feeling like giving up on studying early", "Self-reported early giving up with hopelessness about studying (B11 and B12)",
    ),
    "DistractionLapse_Ind": (
        "getting distracted and forgetting things more than usual", "Self-reported distractibility with forgetfulness (B2 and B5)",
    ),
}

# State -> (student wording, counsellor label, opening line)
STATE_WORDS = {
    "MildStress_State": (
        "some study-related stress",
        "Mild academic stress",
        "It looks like your studies are weighing on you a little at the moment.",
    ),
    "ModerateStress_State": (
        "a noticeable level of study stress",
        "Moderate academic stress",
        "It looks like your studies are putting a real amount of pressure on you right now.",
    ),
    "SevereStress_State": (
        "a high level of study stress",
        "Severe academic stress",
        "It looks like you are carrying a heavy amount of study pressure at the moment.",
    ),
    "EarlyDrift_State": (
        "early signs that your concentration is being affected",
        "Early cognitive drift",
        "There are some early signs that the pressure is starting to affect how easily you can concentrate.",
    ),
    "ModerateDrift_State": (
        "clear signs that your concentration and memory are being affected",
        "Moderate cognitive drift",
        "There are clear signs that sustained pressure is affecting your concentration and recall.",
    ),
    "SevereDrift_State": (
        "strong signs that sustained pressure is affecting how you think and study",
        "Severe cognitive drift",
        "There are strong signs that sustained pressure is affecting how you think and study.",
    ),
}

SUGGESTIONS = {
    "StudentAwarenessMessage_Output": [
        "Noticing this early is genuinely useful - it is much easier to adjust now than later.",
        "A short, deliberate break between study blocks often does more than pushing through.",
    ],
    "ResourceRecommendation_Output": [
        "Breaking the next task into one small, clearly finishable piece can restore some momentum.",
        "Your university's academic support or study skills service can help with workload planning.",
        "Protecting sleep is one of the most reliable things that helps concentration recover.",
    ],
    "CounselorAlert_Output": [
        "Talking this through with someone would help - your student counselling service exists for exactly this.",
        "You do not need to be in crisis to use that support, and reaching out early is a reasonable thing to do.",
    ],
}

CLOSING = (
    "This reflects the patterns in what you wrote and ticked today, not a "
    "judgement about you or your ability. It is not a diagnosis of any kind."
)

STATUS_WORDS = {
    "NoIndicatorsDetected_Status": "No indicators were detected in this session.",
    "MonitoringRequired_Status": (
        "Some indicators were detected but they did not combine into a "
        "recognised state. Worth watching across further sessions."
    ),
    "IncompleteInput_Status": (
        "No speech sample was provided, so the speech-dependent rules "
        "(R08, R20) could not be evaluated in this session."
    ),
    "InsufficientHistory_Status": (
        "This is an early session, so there is not yet enough history to "
        "compare against (rule R41). Progression is tracked from the second "
        "session onwards."
    ),
    "PersistentStress_Status": (
        "Stress has stayed at moderate or above across three consecutive "
        "check-ins (rule R37)."
    ),
    "RisingDrift_Status": (
        "Cognitive drift indicators are higher than in the previous session "
        "(rule R38)."
    ),
}

PROGRESSION_WORDS = {
    "Worsening_Progression": "Stress is higher than in the previous session.",
    "Improving_Progression": "Stress is lower than in the previous session.",
    "Stable_Progression": "Stress is at the same level as the previous session.",
}

STUDENT_PROGRESSION = {
    "Worsening_Progression": "Compared with your last check-in, things look a little heavier this time. That is useful to know, and it is worth talking to someone about.",
    "Improving_Progression": "Compared with your last check-in, things look a little lighter this time. Nice to see.",
}

# A student reading their own result may already be under pressure. The full indicator list goes to the counsellor report; the student
# sees the most important few.
MAX_STUDENT_INDICATORS = 4

DISCLAIMER = (
    "This system is a non-clinical academic self-awareness tool. It does not "
    "diagnose, and its outputs are indicators rather than labels."
)

# EVIDENCE PHRASES - the student's own words
def _sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [part.strip() for part in parts if part.strip()]


def evidence_phrases(text, limit=3):
    """Quote the sentences that carry the language the rules reacted to.  Sentences are ranked by how many absolutist and negative-emotion
    terms they contain, using the same lists pipeline.py scored with.
    """
    scored = []
    for sentence in _sentences(text):
        lowered = sentence.lower()
        absolutist = [
            term
            for term in ABSOLUTIST_TERMS
            if re.search(r"\b" + re.escape(term) + r"\b", lowered)
        ]
        negative = [
            term
            for term in NEGATIVE_EMOTION_TERMS
            if re.search(r"\b" + re.escape(term) + r"\b", lowered)
        ]
        weight = len(absolutist) * 2 + len(negative)
        if weight:
            scored.append(
                {
                    "phrase": sentence,
                    "absolutist_terms": absolutist,
                    "negative_terms": negative,
                    "weight": weight,
                }
            )
    scored.sort(key=lambda item: item["weight"], reverse=True)
    return scored[:limit]


# RULE TRACES
def _confirmed(results, conclusion):
    return [
        rule
        for rule in results.get("fired_rules", [])
        if rule["conclusion"] == conclusion and rule["status"] == "confirmed"
    ]


def format_trace(rule):
    """One rule's evidence as readable lines."""
    lines = []
    for item in rule.get("evidence", []):
        if item["kind"] == "test":
            lines.append(
                "%s %s %s" % (item["observed"], item["operator"], item["threshold"])
            )
        elif item.get("property") in ("hasSession", "hasSelfTalkEntry",
                                      "hasSpeechSample", "hasBehavioralSignal"):
            continue  # structural navigation, not evidence
        else:
            lines.append("%s = %s" % (item["property"], item["value"]))
    return lines


def indicator_traces(results):
    """[{indicator, student_wording, counsellor_wording, rules:[...]}]"""
    traces = []
    for name in results.get("inferred_indicators", []):
        student_words, counsellor_words = INDICATOR_WORDS.get(
            name, (name, name)
        )
        traces.append(
            {
                "indicator": name,
                "student_wording": student_words,
                "counsellor_wording": counsellor_words,
                "rules": [
                    {
                        "rule_id": rule["rule_id"],
                        "label": rule["label"],
                        "evidence": format_trace(rule),
                    }
                    for rule in _confirmed(results, name)
                ],
            }
        )
    return traces


# STUDENT-FACING MESSAGE
def student_message(results, phrases=None):
    stress = results.get("primary_stress_state")
    drift = results.get("primary_drift_state")
    headline = drift or stress

    if not headline:
        if results.get("inferred_indicators"):
            body = [
                "Nothing in today's check-in points to a clear pattern, though "
                "a few individual signals did come up:",
                "  - "
                + "\n  - ".join(
                    INDICATOR_WORDS.get(name, (name, name))[0]
                    for name in results["inferred_indicators"]
                ),
                "That is worth keeping an eye on rather than acting on.",
            ]
        else:
            body = [
                "Thanks for checking in. The system did not pick up a specific "
                "pattern of study stress or difficulty concentrating from what "
                "you wrote and ticked today. That does not mean everything is "
                "fine: this tool only sees part of the picture, and you know how "
                "you feel better than it does. If things feel heavy, it is "
                "always worth talking to your counsellor or someone you trust.",
            ]
        return "\n\n".join(body + [CLOSING])

    lines = [STATE_WORDS[headline][2]]

    if drift and stress:
        lines.append(
            "Alongside that, today's check-in also shows %s."
            % STATE_WORDS[stress][0]
        )

    # A student reading this may already be under pressure, so keep the list short. The counsellor report carries the full set.
    noticed = [
        INDICATOR_WORDS.get(name, (name, name))[0]
        for name in results.get("inferred_indicators", [])
    ]
    if noticed:
        shown = noticed[:MAX_STUDENT_INDICATORS]
        text = "What came through in what you wrote and ticked:\n  - " + \
               "\n  - ".join(shown)
        remaining = len(noticed) - len(shown)
        if remaining > 0:
            text += "\n\n(%d further signals were recorded as well.)" % remaining
        lines.append(text)

    if phrases:
        lines.append(
            "Some of your own words that stood out:\n  - "
            + "\n  - ".join('"%s"' % item["phrase"] for item in phrases)
        )

    escalation = results.get("escalation")
    if escalation in SUGGESTIONS:
        lines.append("A couple of things that might help:\n  - "
                     + "\n  - ".join(SUGGESTIONS[escalation]))

    prog = results.get("progression")
    if "RisingDrift_Status" in results.get("session_statuses", []) and prog != "Worsening_Progression":
        # The progression label follows stress only. Do not tell a student things look lighter when concentration signs went up.
        lines.append(
            "Compared with your last check-in, the signs affecting your "
            "concentration look a little stronger this time"
            + (", even though your stress looks lower." if prog == "Improving_Progression" else ".")
        )
    elif prog in STUDENT_PROGRESSION:
        lines.append(STUDENT_PROGRESSION[prog])

    for status in results.get("session_statuses", []):
        if status in ("IncompleteInput_Status", "InsufficientHistory_Status",
                      "PersistentStress_Status", "RisingDrift_Status"):
            continue  # technical or counsellor-facing, not for the student
        if status in STATUS_WORDS and not headline:
            lines.append(STATUS_WORDS[status])

    lines.append(CLOSING)
    return "\n\n".join(lines)


# COUNSELLOR-FACING REPORT
def counselor_report(results, phrases=None):
    out = []
    add = out.append

    add("COGNITIVE DRIFT CHECK-IN - SUMMARY FOR ACADEMIC SUPPORT")
    add("Session: %s" % results.get("student_id", "unknown"))
    add("")

    stress = results.get("primary_stress_state")
    drift = results.get("primary_drift_state")
    add("INFERRED STATE")
    add("  Academic stress : %s"
        % (STATE_WORDS[stress][1] if stress else "none inferred"))
    add("  Cognitive drift : %s"
        % (STATE_WORDS[drift][1] if drift else "none inferred"))
    escalation = results.get("escalation")
    add("  Recommended     : %s" % (escalation or "no action indicated"))
    add("  Ontology-validated resolution: %s"
        % ("yes" if results.get("resolution_validated") else "no"))
    add("")

    rating = results.get("self_rating")
    if rating is not None and rating >= 6 and (not stress or stress == "MildStress_State") and not drift:
        add("NOTE FOR THE COUNSELLOR")
        add("  The student rated their own stress %s/10, but the system found %s."
            % (rating, "no stress or drift state" if not stress else "only mild stress"))
        add("  The system reads wording, speech and the behaviour checklist, so it can miss")
        add("  stress that is written calmly. Worth reading the student's own words below.")
        add("")

    context = results.get("asserted_values", {}).get("session", {})
    if context:
        add("SESSION CONTEXT")
        add("  Activity        : %s" % context.get("activity_type"))
        add("  Workload        : %s" % context.get("workload_level"))
        add("  Tied to event   : %s" % context.get("linked_to_event"))
        add("")

    traces = indicator_traces(results)
    add("INDICATORS AND THE EVIDENCE FOR EACH (%d)" % len(traces))
    if not traces:
        add("  None.")
    for trace in traces:
        add("  %s" % trace["counsellor_wording"])
        for rule in trace["rules"]:
            add("      via %s" % rule["label"])
            for line in rule["evidence"]:
                add("          %s" % line)
    add("")

    states = results.get("inferred_states", [])
    if states:
        add("STATE RULES THAT FIRED")
        for name in states:
            for rule in _confirmed(results, name):
                add("  %-34s -> %s" % (rule["label"], name))
        add("")

    if results.get("state_conflicts"):
        add("NOTE - the rule set entailed several severity levels at once.")
        add("The reported state is the highest of each ladder. Pairs the")
        add("ontology declares mutually disjoint:")
        for clash in results["state_conflicts"]:
            add("  %s / %s" % (clash["states"][0], clash["states"][1]))
        add("")

    checklist = results.get("asserted_values", {}).get("checklist", {})
    ticked = [
        item
        for item, value in sorted(
            checklist.items(), key=lambda pair: int(pair[0][1:])
        )
        if value
    ]
    add("SELF-REPORTED BEHAVIOURS TICKED (%d of 15)" % len(ticked))
    add("  %s" % (", ".join(ticked) if ticked else "none"))
    add("")

    scores = results.get("asserted_values", {}).get("self_talk", {})
    if scores:
        add("LANGUAGE MEASURES")
        for key in sorted(scores):
            add("  %-28s %s" % (key, scores[key]))
        add("")

    if phrases:
        add("EVIDENCE PHRASES (student's own words)")
        for item in phrases:
            add('  "%s"' % item["phrase"])
            markers = item["absolutist_terms"] + item["negative_terms"]
            if markers:
                add("      markers: %s" % ", ".join(markers))
        add("")

    prog = results.get("progression")
    pinfo = results.get("asserted_values", {}).get("progression") or {}
    if prog or pinfo.get("history"):
        add("PROGRESSION (compared with earlier sessions, by SWRL rules R34-R38):")
        if prog:
            add("  %s" % PROGRESSION_WORDS.get(prog, prog))
        for r in pinfo.get("history", []):
            add("  session %s: stress rank %s, drift rank %s"
                % (r.get("seq"), r.get("stress_rank"), r.get("drift_rank")))
        add("  this session: stress rank %s, drift rank %s"
            % (pinfo.get("current_stress_rank"), pinfo.get("current_drift_rank")))
        add("")

    for status in results.get("session_statuses", []):
        if status in STATUS_WORDS:
            add("STATUS: %s" % STATUS_WORDS[status])
    add("")
    add(DISCLAIMER)
    return "\n".join(out)

# WRITING THE EXPLANATION BACK INTO THE ONTOLOGY
def _input_signals(onto, rule):
    """Individuals bound in a rule trace that are InputSignals."""
    signal_class = entity(onto, "InputSignal")
    found = []
    for item in rule.get("evidence", []):
        value = item.get("value")
        if not isinstance(value, str):
            continue
        try:
            individual = entity(onto, value)
        except LookupError:
            continue
        if isinstance(individual, signal_class):
            found.append(individual)
    return found


def write_to_ontology(results, student_text, counselor_text, phrases):
    """Create the Explanation individual and the provenance links. Populates hasExplanation, explanation_*, evidencedBy (indicator
    -> the input signal that produced it), triggeredBy (state -> the indicators in the rule that produced it) and supportedBy
    (state -> input signals).
    """
    session = results.get("_session")
    if not session:
        return None

    onto = session["onto"]
    student = session["student"]
    uid = session["uid"]

    with onto:
        explanation = entity(onto, "Explanation")("Explanation_%s" % uid)
        explanation.explanation_hasStudentFacingMessage = [student_text]
        explanation.explanation_hasCounselorFacingMessage = [counselor_text]
        explanation.explanation_hasEvidencePhrase = [
            item["phrase"] for item in phrases
        ]
        explanation.explanation_hasInferredStateLabel = [
            STATE_WORDS[name][1]
            for name in (
                results.get("primary_stress_state"),
                results.get("primary_drift_state"),
            )
            if name
        ]
        explanation.explanation_hasTriggeredRuleID = sorted(
            {
                rule["rule_id"]
                for rule in results.get("fired_rules", [])
                if rule["status"] == "confirmed"
            }
        )
        # Asserted, not assumed: this system is not diagnostic.
        explanation.explanation_isClinicallyDiagnostic = [False]
        student.hasExplanation = [explanation]

        # evidencedBy: indicator -> the raw input that caused it
        for name in results.get("inferred_indicators", []):
            indicator = entity(onto, name)
            for rule in _confirmed(results, name):
                for signal in _input_signals(onto, rule):
                    if signal not in indicator.evidencedBy:
                        indicator.evidencedBy.append(signal)

        # triggeredBy / supportedBy: state -> indicators and inputs
        for name in results.get("inferred_states", []):
            state = entity(onto, name)
            for rule in _confirmed(results, name):
                for item in rule.get("evidence", []):
                    if item.get("property") != "exhibitsIndicator":
                        continue
                    try:
                        indicator = entity(onto, item["value"])
                    except LookupError:
                        continue
                    if indicator not in state.triggeredBy:
                        state.triggeredBy.append(indicator)
                for signal in _input_signals(onto, rule):
                    if signal not in state.supportedBy:
                        state.supportedBy.append(signal)

    return explanation



# SAFETY WORDING
#
# A short, deliberately narrow list of phrases that suggest a student may be in more distress than a study-stress tool should judge alone.
# A match does NOT mean anything is diagnosed. It only makes the counsellor see the check-in promptly and read the student's own words.
# The list needs review by the supervisor / university counselling service before real use; it will miss indirect wording.
SAFETY_PATTERNS = [
    r"\bwant to disappear\b", r"\bwish i could disappear\b",
    r"\b(?:can'?t|cannot|can not) go on (?:like this|anymore|any more|living)\b",
    r"\bdon'?t want to be here\b", r"\bend it all\b",
    r"\bno reason to (?:live|go on|keep going)\b",
    r"\b(?:hurt|harm) myself\b", r"\bkill myself\b", r"\bend my life\b",
    r"\bbetter off without me\b", r"\bgive up on life\b",
]

SAFETY_STUDENT_LINE = (
    "Some of the words you used suggest things may feel very heavy right now. "
    "Please think about talking to your counsellor or someone you trust today, "
    "rather than waiting."
)


def safety_wording(text):
    """Return the matched phrases (empty list if none)."""
    found = []
    for pattern in SAFETY_PATTERNS:
        for m in re.finditer(pattern, text or "", flags=re.IGNORECASE):
            found.append(m.group(0).lower())
    return sorted(set(found))

# -------------------------------------------------------------
# MAIN ENTRY POINT
# -------------------------------------------------------------

def build_explanation(results, raw_text=None, write_back=True, self_rating=None):
    """Turn reasoner results into student and counsellor output. `raw_text` is the student's self-talk. If omitted it is taken
    from the text the reasoner asserted into the ontology.
    """
    if not results.get("consistent", False):
        return {
            "student_message": (
                "The analysis could not be completed for this session. "
                "Nothing has been recorded."
            ),
            "counselor_report": "Reasoning failed: %s" % results.get("error"),
            "evidence_phrases": [],
            "indicator_traces": [],
            "written_to_ontology": False,
        }

    if raw_text is None:
        session = results.get("_session")
        raw_text = ""
        if session:
            entries = session["student"].hasSession
            if entries and entries[0].hasSelfTalkEntry:
                texts = entries[0].hasSelfTalkEntry[0].selfTalk_hasRawText
                raw_text = texts[0] if texts else ""

    if self_rating is not None:
        results["self_rating"] = self_rating

    phrases = evidence_phrases(raw_text)
    student_text = student_message(results, phrases)
    counselor_text = counselor_report(results, phrases)

    safety = safety_wording(raw_text)
    results["safety_phrases"] = safety
    if safety:
        student_text = student_text.replace(CLOSING, SAFETY_STUDENT_LINE + "\n\n" + CLOSING, 1)
        priority = (
            "PRIORITY - WORDING THAT MAY NEED A PERSONAL CHECK-IN\n"
            "  The student used wording that can signal serious distress: %s.\n"
            "  This is a keyword match, not an assessment. Please read their own words\n"
            "  below and consider contacting them promptly, whatever the rules inferred."
            % ", ".join('"%s"' % p for p in safety)
        )
        counselor_text = counselor_text.replace("\n\nINFERRED STATE", "\n\n" + priority + "\n\nINFERRED STATE", 1)

    written = False
    if write_back:
        written = write_to_ontology(
            results, student_text, counselor_text, phrases
        ) is not None

    return {
        "student_message": student_text,
        "counselor_report": counselor_text,
        "evidence_phrases": phrases,
        "indicator_traces": indicator_traces(results),
        "written_to_ontology": written,
        "safety_flag": bool(safety),
    }


# SELF TEST
if __name__ == "__main__":
    import reasoner

    text = (
        "I always fail everything. I can never understand this material no "
        "matter how hard I try. My mind goes completely blank in exams. "
        "I am totally useless at this subject and I feel hopeless about the "
        "whole thing. There is no point continuing anymore."
    )
    features = {
        "raw_text": text,
        "sentiment_score": -0.85,
        "negative_language_score": 0.75,
        "absolutist_term_count": 4,
        "helplessness_score": 0.85,
        "catastrophizing_score": 0.78,
        "overgeneralization_score": 0.70,
        "subject_specific": True,
    }
    checklist = {
        "B1": True, "B3": True, "B4": True, "B6": True,
        "B7": True, "B8": True, "B10": True,
        "B13": True, "B14": True,
    }
    student_info = {
        "name": "Test Student",
        "discipline": "Computing",
        "year_of_study": 2,
        "activity_type": "ExamPreparation",
        "workload_level": "High",
        "is_linked_to_event": True,
    }

    results = reasoner.run_full_pipeline(
        features, checklist, student_info, verbose=False
    )
    output = build_explanation(results, raw_text=text)

    print("=" * 62)
    print("STUDENT-FACING MESSAGE")
    print("=" * 62)
    print(output["student_message"])
    print()
    print("=" * 62)
    print(output["counselor_report"])
    print()
    print("Written back into the ontology: %s" % output["written_to_ontology"])

    session = results.get("_session")
    if session:
        explanation = session["student"].hasExplanation[0]
        print("Explanation individual : %s" % explanation.name)
        print("Triggered rule IDs     : %s"
              % explanation.explanation_hasTriggeredRuleID)
        print("Clinically diagnostic  : %s"
              % explanation.explanation_isClinicallyDiagnostic)
        indicator = session["onto"].search_one(iri="*#AttentionLoss_Ind")
        print("AttentionLoss evidencedBy: %s"
              % [signal.name for signal in indicator.evidencedBy])
