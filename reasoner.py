# reasoner.py
# Ontology Population and Symbolic Reasoning Module
#
# This is the neural -> symbolic bridge. It takes the numeric features produced by pipeline.py plus the behavioural checklist,
# asserts them into the OWL 2 ontology as data properties on freshly created individuals, runs the Pellet reasoner over the
#  SWRL rules, and reads the inferred indicators, states and support outputs back as Python objects.
#
# It reasons in two passes:
#   Pass 1  assert the observed facts, run Pellet, collect every indicator, state and support output the rules entail. Resolve pick one
#          headline stress state and one headline drift state from the ladders (SWRL has no negation as failure, so the rules cannot do this themselves).
#   Pass 2  assert the resolved outcome back into the ontology as properly typed individuals and run Pellet again. The disjointness axioms then 
#           actually validate the result instead of sitting unused, and a contradictory resolution would be reported as an inconsistency.
#
# DESIGN RULE:
# This module asserts FACTS only. It contains no thresholds and no severity decisions. Every threshold (0.6, 0.65, -0.7, > 3)
# lives in a named SWRL rule inside cognitive_drift.owl, which is the single source of truth for the system's judgements.

import os
import re
import uuid

import owlready2
from owlready2 import World, sync_reasoner_pellet
from owlready2.base import OwlReadyInconsistentOntologyError
from owlready2.rule import Variable

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONTOLOGY_PATH = os.path.join(BASE_DIR, "cognitive_drift.owl")

owlready2.JAVA_EXE = os.environ.get("JAVA_EXE", owlready2.JAVA_EXE or "java")

STRESS_LADDER = ["MildStress_State", "ModerateStress_State", "SevereStress_State"]
DRIFT_LADDER = ["EarlyDrift_State", "ModerateDrift_State", "SevereDrift_State"]
ACTION_LADDER = ["StudentAwarenessMessage_Output", "ResourceRecommendation_Output", "CounselorAlert_Output"]

PROGRESSION_STATUSES = [ "PersistentStress_Status", "RisingDrift_Status", "InsufficientHistory_Status"]
MAX_HISTORY = 2

def severity_rank(state_name, ladder):
    """0 = none, 1 = mild/early, 2 = moderate, 3 = severe."""
    return ladder.index(state_name) + 1 if state_name in ladder else 0


def normalise_history(history):
    """Most recent first, at most MAX_HISTORY earlier sessions, with ranks."""
    items = []
    for entry in history or []:
        items.append(entry)
    items.sort(key=lambda e: e.get("seq") or 0, reverse=True)
    records = []
    for index, entry in enumerate(items[:MAX_HISTORY], start=1):
        stress = entry.get("primary_stress_state")
        drift = entry.get("primary_drift_state")
        records.append({
            "recency": index,
            "seq": entry.get("seq"),
            "stress_state": stress,
            "drift_state": drift,
            "stress_rank": severity_rank(stress, STRESS_LADDER),
            "drift_rank": severity_rank(drift, DRIFT_LADDER),
        })
    return records


# Closed-world session statuses.
#
# OWL is open world: "this student shows no indicators" is not something a reasoner can entail, because absence of evidence is
# not evidence of absence. Those two statuses are therefore  asserted here, in pass 2, where the set of inputs is known to be
# complete. IncompleteInput_Status is different - R29 entails it from an absent speech sample, so it comes out of the reasoner.
STATUS_NO_INDICATORS = "NoIndicatorsDetected_Status"
STATUS_MONITORING = "MonitoringRequired_Status"
STATUS_INCOMPLETE_INPUT = "IncompleteInput_Status"

# Checklist item -> (behaviour individual key, data property name). B1..B15 are the fifteen self-reported micro-behaviours.
CHECKLIST_MAP = {
    "B1": ("attention", "attention_hasFocusDifficultyReport"),
    "B2": ("attention", "attention_hasDistractionReport"),
    "B3": ("attention", "attention_hasZoningOutReport"),
    "B4": ("memory", "memory_hasReReadingReport"),
    "B5": ("memory", "memory_hasForgetfulnessReport"),
    "B6": ("memory", "memory_hasMindBlankReport"),
    "B7": ("energy", "energy_hasMentalExhaustionReport"),
    "B8": ("energy", "energy_hasSlowedThinkingReport"),
    "B9": ("energy", "energy_hasClearThoughtDiffReport"),
    "B10": ("avoidance", "avoidance_hasTaskPostponementReport"),
    "B11": ("avoidance", "avoidance_hasEarlyGiveUpReport"),
    "B12": ("avoidance", "avoidance_hasHopelessnessStudyReport"),
    "B13": ("physical", "physical_hasHeadacheReport"),
    "B14": ("physical", "physical_hasSleepDifficultyReport"),
    "B15": ("physical", "physical_hasPhysicalRestlessnessReport"),
}

BEHAVIOUR_CLASSES = {
    "attention": "AttentionFocusBehavior",
    "memory": "MemoryBehavior",
    "energy": "MentalEnergyBehavior",
    "avoidance": "AvoidanceBehavior",
    "physical": "PhysicalBehavior",
}


# ONTOLOGY HELPERS
def load_ontology(path=ONTOLOGY_PATH):
    """Load the ontology into a fresh in-memory world. A new World per call keeps sessions isolated, which matters
        under Streamlit where the module is imported once but called many times.
    """
    world = World()
    onto = world.get_ontology(path).load()
    onto.name = "cognitive_drift"
    return world, onto


def entity(onto, name):
    """Look up a class, property or individual by local name."""
    found = list(onto.search(iri="*#%s" % name))
    if not found:
        raise LookupError(
            "'%s' not found in the ontology. Check the name against "
            "cognitive_drift.owl." % name
        )
    return found[0]


# SWRL RULE CATALOGUE
# The rules are read out of the ontology itself rather than duplicated in Python, so there is exactly one source of truth.
# The catalogue maps each conclusion (e.g. ExamAnxiety_Ind) to the rule or rules that can produce it, with their body atoms. This
# is what explanation.py should build its evidence traces from.

_HEAD_RE = re.compile(r"^(\w+)\(\?\w+,\s*.*?(\w+)\)$")


def rule_catalogue(onto):
    """Return {conclusion_name: [rule_dict, ...]} read from the ontology."""
    catalogue = {}
    for imp in onto.rules():
        labels = list(getattr(imp, "label", []) or [])
        label = labels[0] if labels else "(unlabelled rule)"
        rule_id = label.split("_")[0] if label.startswith("R") else label
        body = [str(atom) for atom in imp.body]
        for atom in imp.head:
            text = str(atom)
            match = _HEAD_RE.match(text)
            if not match:
                continue
            prop, target = match.groups()
            catalogue.setdefault(target, []).append(
                {
                    "rule_id": rule_id,
                    "label": label,
                    "property": prop,
                    "conclusion": target,
                    "body": body,
                    "head": text,
                    "imp": imp,
                }
            )
    return catalogue


# RULE BODY REPLAY - evidence for the explanation layer
#
# Pellet tells us WHAT was inferred but not WHICH rule produced it, and several rules can share a conclusion (four different
# rules can assert EarlyDrift_State). To turn an inference into an explanation we replay each candidate rule's body against the
# facts that were actually asserted and inferred.
#
# This is not a second copy of the rules. The atoms are read from the ontology at runtime; this is a small generic SWRL body
# evaluator over whatever rules the ontology happens to contain. Edit a threshold in Protege and the explanation follows it.
#
# A rule is reported as:
#   confirmed  - a satisfying variable binding was found
#   eliminated - the body cannot be satisfied, so it did not fire
#   candidate  - could not be decided here (dropped to a maybe)

class _Undecidable(Exception):
    """Raised when an atom cannot be evaluated from the known facts."""


_BUILTINS = {
    "greaterThan": lambda a, b: a > b,
    "greaterThanOrEqual": lambda a, b: a >= b,
    "lessThan": lambda a, b: a < b,
    "lessThanOrEqual": lambda a, b: a <= b,
    "equal": lambda a, b: a == b,
    "notEqual": lambda a, b: a != b,
}


def _resolve(argument, bindings):
    """Turn a rule argument into a concrete value."""
    if isinstance(argument, Variable):
        if argument.name not in bindings:
            raise _Undecidable(argument.name)
        return bindings[argument.name]
    return argument


def _property_values(individual, prop):
    value = getattr(individual, prop.python_name, None)
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _label(value):
    return getattr(value, "name", value)


def _solve(atoms, index, bindings, trace):
    """Depth-first search for a variable binding satisfying the body.

    Returns (bindings, trace) on success or None on failure.
    """
    if index == len(atoms):
        return dict(bindings), list(trace)

    atom = atoms[index]
    step = lambda new_bindings, entry: _solve(  # noqa: E731
        atoms, index + 1, new_bindings, trace + ([entry] if entry else [])
    )

    # Class membership, e.g. Student(?p)
    if atom.class_predicate is not None:
        subject = _resolve(atom.arguments[0], bindings)
        if isinstance(subject, atom.class_predicate):
            return step(bindings, None)
        return None

    # Object or data property, e.g. hasSession(?p, ?sess)
    if atom.property_predicate is not None:
        prop = atom.property_predicate
        subject = _resolve(atom.arguments[0], bindings)
        obj = atom.arguments[1]
        values = _property_values(subject, prop)

        if isinstance(obj, Variable) and obj.name not in bindings:
            for value in values:
                extended = dict(bindings)
                extended[obj.name] = value
                entry = {
                    "atom": str(atom),
                    "kind": "fact",
                    "subject": _label(subject),
                    "property": prop.name,
                    "value": _label(value),
                }
                found = step(extended, entry)
                if found:
                    return found
            return None

        expected = _resolve(obj, bindings)
        if expected in values:
            return step(
                bindings,
                {
                    "atom": str(atom),
                    "kind": "fact",
                    "subject": _label(subject),
                    "property": prop.name,
                    "value": _label(expected),
                },
            )
        return None

    # Comparison, e.g. greaterThan(?score, 0.6)
    if atom.builtin is not None:
        operator = _BUILTINS.get(str(atom.builtin))
        if operator is None:
            raise _Undecidable(str(atom.builtin))
        left, right = (_resolve(a, bindings) for a in atom.arguments)
        if isinstance(left, bool) != isinstance(right, bool):
            raise _Undecidable(str(atom))
        try:
            holds = operator(left, right)
        except TypeError:
            raise _Undecidable(str(atom))
        if holds:
            return step(
                bindings,
                {
                    "atom": str(atom),
                    "kind": "test",
                    "operator": str(atom.builtin),
                    "observed": left,
                    "threshold": right,
                },
            )
        return None

    raise _Undecidable(str(atom))


def replay_rule(imp, student):
    """Replay one SWRL rule body. Returns (status, evidence_trace)."""
    try:
        solved = _solve(list(imp.body), 0, {"p": student}, [])
    except _Undecidable:
        return "candidate", []
    if solved is None:
        return "eliminated", []
    return "confirmed", solved[1]


# POPULATION - the neural -> symbolic assertion step
def populate(onto, nlp_features, checklist, student_info, speech_features=None):
    """Create the session individuals and assert the observed facts. Returns (student_individual, asserted_values_dict).
    """
    uid = str(uuid.uuid4())[:8]
    checklist = checklist or {}
    student_info = student_info or {}
    asserted = {}

    with onto:
        # -- Student -------------------------------------------
        student = entity(onto, "Student")("Student_%s" % uid)
        student.hasName = [str(student_info.get("name", "Anonymous"))]
        student.hasDiscipline = [str(student_info.get("discipline", "Computing"))]
        student.hasYearOfStudy = [int(student_info.get("year_of_study", 1))]
        student.hasSessionCount = [int(student_info.get("session_count", 1))]

        # -- AcademicSession -----------------------------------
        session = entity(onto, "AcademicSession")("Session_%s" % uid)
        session.session_hasActivityType = [
            str(student_info.get("activity_type", "SelfStudy"))
        ]
        session.session_hasWorkloadLevel = [
            str(student_info.get("workload_level", "Moderate"))
        ]
        session.session_isLinkedToSpecificEvent = [
            bool(student_info.get("is_linked_to_event", False))
        ]
        session.session_isStressAcuteOrPersistent = [
            str(student_info.get("is_acute_or_persistent", "Acute"))
        ]
        student.hasSession = [session]
        asserted["session"] = {
            "activity_type": session.session_hasActivityType[0],
            "workload_level": session.session_hasWorkloadLevel[0],
            "linked_to_event": session.session_isLinkedToSpecificEvent[0],
        }

        # -- SelfTalkEntry -------------------------------------
        # Scores are written as Python floats, which Owlready2 serializes as xsd:decimal - matching the ontology ranges.
        nlp_features = nlp_features or {}
        entry = entity(onto, "SelfTalkEntry")("SelfTalk_%s" % uid)
        entry.selfTalk_hasRawText = [str(nlp_features.get("raw_text", ""))[:500]]
        entry.selfTalk_hasSentimentScore = [
            float(nlp_features.get("sentiment_score", 0.0))
        ]
        entry.selfTalk_hasNegativeLanguageScore = [
            float(nlp_features.get("negative_language_score", 0.0))
        ]
        entry.selfTalk_hasAbsolutistTermCount = [
            int(nlp_features.get("absolutist_term_count", 0))
        ]
        entry.selfTalk_hasShouldStatementCount = [
            int(nlp_features.get("should_statement_count", 0))
        ]
        entry.selfTalk_hasPersonalizationCount = [
            int(nlp_features.get("personalization_count", 0))
        ]
        entry.selfTalk_hasHelplessnessScore = [
            float(nlp_features.get("helplessness_score", 0.0))
        ]
        entry.selfTalk_hasCatastrophizingScore = [
            float(nlp_features.get("catastrophizing_score", 0.0))
        ]
        entry.selfTalk_hasOvergeneralizationScore = [
            float(nlp_features.get("overgeneralization_score", 0.0))
        ]
        entry.selfTalk_hasSubjectSpecific = [
            bool(nlp_features.get("subject_specific", False))
        ]
        session.hasSelfTalkEntry = [entry]
        asserted["self_talk"] = {
            "sentiment_score": entry.selfTalk_hasSentimentScore[0],
            "negative_language_score": entry.selfTalk_hasNegativeLanguageScore[0],
            "absolutist_term_count": entry.selfTalk_hasAbsolutistTermCount[0],
            "should_statement_count": entry.selfTalk_hasShouldStatementCount[0],
            "personalization_count": entry.selfTalk_hasPersonalizationCount[0],
            "helplessness_score": entry.selfTalk_hasHelplessnessScore[0],
            "catastrophizing_score": entry.selfTalk_hasCatastrophizingScore[0],
            "overgeneralization_score": entry.selfTalk_hasOvergeneralizationScore[0],
        }

        # -- SpeechSample --------------------------------------
        speech = entity(onto, "SpeechSample")("Speech_%s" % uid)
        sf = speech_features or {}
        available = bool(speech_features)
        speech.speech_hasSpeechAvailable = [available]
        speech.speech_hasTranscribedText = [str(sf.get("transcribed_text", ""))[:500]]
        speech.speech_hasSpeechPace = [str(sf.get("speech_pace", "Normal"))]
        speech.speech_hasToneIndicator = [str(sf.get("tone_indicator", "Normal"))]
        speech.speech_hasHesitationCount = [int(sf.get("hesitation_count", 0))]
        speech.speech_hasSentimentScore = [float(sf.get("sentiment_score", 0.0))]
        speech.speech_hasNegativeLanguageScore = [
            float(sf.get("negative_language_score", 0.0))
        ]
        speech.speech_hasAbsolutistTermCount = [
            int(sf.get("absolutist_term_count", 0))
        ]
        speech.speech_hasHelplessnessScore = [float(sf.get("helplessness_score", 0.0))]
        speech.speech_hasCatastrophizingScore = [
            float(sf.get("catastrophizing_score", 0.0))
        ]
        speech.speech_hasOvergeneralizationScore = [
            float(sf.get("overgeneralization_score", 0.0))
        ]
        session.hasSpeechSample = [speech]
        asserted["speech"] = {
            "available": available,
            "speech_pace": speech.speech_hasSpeechPace[0],
            "hesitation_count": speech.speech_hasHesitationCount[0],
        }

        # -- Self-reported behaviour ---------------------------
        behaviours = {}
        for key, cls_name in BEHAVIOUR_CLASSES.items():
            behaviours[key] = entity(onto, cls_name)(
                "%s_%s" % (key.capitalize(), uid)
            )
        for item, (group, prop) in CHECKLIST_MAP.items():
            setattr(behaviours[group], prop, [bool(checklist.get(item, False))])
        session.hasBehavioralSignal = list(behaviours.values())
        # How many of the five checklist groups have at least one item ticked. Counted here and asserted as a fact; R57 decides what it means.
        groups_ticked = len({
            group for item, (group, prop) in CHECKLIST_MAP.items()
            if checklist.get(item, False)
        })
        student.hasBehaviourGroupCount = [groups_ticked]
        asserted["behaviour_group_count"] = groups_ticked
        # v5 (calibration round 1): total ticked items, and a transparent stress evidence scorescore = 2 x min(stress words, 3) + 1 if sentiment < -0.5 + 1 if 8 or more items ticked
        # Counted here and asserted as facts; R62-R67 decide what they mean.
        tick_count = sum(1 for item in CHECKLIST_MAP if checklist.get(item, False))
        sw = min(int(nlp_features.get("stress_statement_count", 0) or 0), 3)
        sent = float(nlp_features.get("sentiment_score", 0.0) or 0.0)
        stress_score = 2 * sw + (1 if sent < -0.5 else 0) + (1 if tick_count >= 8 else 0)
        student.hasChecklistTickCount = [tick_count]
        student.hasStressEvidenceScore = [stress_score]
        asserted["checklist_tick_count"] = tick_count
        asserted["stress_evidence_score"] = stress_score
        asserted["checklist"] = {
            item: bool(checklist.get(item, False)) for item in CHECKLIST_MAP
        }

    return student, asserted


# -------------------------------------------------------------
# SEVERITY ARBITRATION
#
# NOTE - open research decision.
# The SWRL rules link the student to fixed state individuals (MildStress_State, SevereStress_State, ...). Because the student
# is never typed into a state class, the disjointness axioms in the ontology never constrain anything, and a severe case can
# fire several stress levels and several drift levels at once. Until that is resolved at the ontology level, this function
# applies an explicit, editable "highest severity wins" policy for display only. The complete set of fired states is always
# returned separately so nothing is hidden.

def resolve_severity(names, ladder):
    """Return the highest-severity member of `ladder` present in `names`."""
    present = [step for step in ladder if step in names]
    return present[-1] if present else None


def state_conflicts(onto, state_names):
    """Pairs of fired states the ontology itself declares disjoint. This is a rule-set audit, not a runtime error. It reads the
    disjointness axioms out of the ontology rather than assuming a ladder, so it reports exactly what the ontology considers
    contradictory. Under the current Tier B rules a severe case fires several levels at once, and this is what surfaces it.
    """
    individuals = {}
    for name in state_names:
        try:
            individuals[name] = entity(onto, name)
        except LookupError:
            continue

    def classes_of(individual):
        found = set()
        for cls in individual.is_a:
            if hasattr(cls, "ancestors"):
                found.add(cls)
        return found

    conflicts = []
    names = sorted(individuals)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            for cls_a in classes_of(individuals[left]):
                for cls_b in classes_of(individuals[right]):
                    if _declared_disjoint(cls_a, cls_b):
                        conflicts.append(
                            {
                                "states": [left, right],
                                "classes": [cls_a.name, cls_b.name],
                            }
                        )
    return conflicts


def _declared_disjoint(cls_a, cls_b):
    """True if the ontology declares these two classes disjoint."""
    if cls_a is cls_b:
        return False
    ancestors_a = set(cls_a.ancestors())
    ancestors_b = set(cls_b.ancestors())
    for disjoint in cls_a.disjoints():
        members = set(disjoint.entities)
        if members & ancestors_a and members & ancestors_b:
            return True
    for disjoint in cls_b.disjoints():
        members = set(disjoint.entities)
        if members & ancestors_a and members & ancestors_b:
            return True
    return False


def assert_resolution(onto, student, uid, resolved_states, statuses):
    """Pass 2 - write the resolved outcome back as typed individuals. Each resolved state gets its own individual typed into the
    corresponding state class, so `hasInferredState`'s range and the disjointness axioms are exercised for real when the reasoner runs again.
    """
    minted = []
    with onto:
        for state_name in resolved_states:
            if not state_name:
                continue
            template = entity(onto, state_name)
            state_class = next(
                (cls for cls in template.is_a if hasattr(cls, "ancestors")), None
            )
            if state_class is None:
                continue
            resolved = state_class("Resolved_%s_%s" % (state_class.name, uid))
            student.hasInferredState.append(resolved)
            minted.append(resolved)
        for status_name in statuses:
            student.hasSessionStatus.append(entity(onto, status_name))
    return minted


def assert_progression_facts(onto, student, uid, current_stress, current_drift, records):
    """Pass 2 - give the progression rules (R34-R40) their inputs. Asserts the severity rank of this session's headline states and one
    PreviousSessionRecord per earlier session. Nothing is concluded here:whether things are worsening, stable, improving, persistent or rising
    is decided by the SWRL rules inside the ontology.
    """
    with onto:
        student.hasCurrentStressRank = [severity_rank(current_stress, STRESS_LADDER)]
        student.hasCurrentDriftRank = [severity_rank(current_drift, DRIFT_LADDER)]
        record_class = entity(onto, "PreviousSessionRecord")
        for record in records:
            individual = record_class("Prev%d_%s" % (record["recency"], uid))
            individual.prev_hasRecency = [record["recency"]]
            individual.prev_hasStressRank = [record["stress_rank"]]
            individual.prev_hasDriftRank = [record["drift_rank"]]
            student.hasPreviousSession.append(individual)


# -------------------------------------------------------------
# MAIN ENTRY POINT
def run_full_pipeline(
    nlp_features,
    checklist,
    student_info,
    speech_features=None,
    ontology_path=ONTOLOGY_PATH,
    validate=True,
    verbose=True,
    history=None,
):
    """Assert one session into the ontology, reason, and read back. `history` is an optional list of the student's EARLIER sessions'
    stored conclusions, each {"seq", "primary_stress_state", "primary_drift_state"}. When given, the ontology's progression rules
    compare this session with them (see MAX_HISTORY).

    Returns a dict with the inferred indicators, states and support  outputs, the values that were asserted, and the SWRL rules that
    could have produced each conclusion.
    """
    def log(message):
        if verbose:
            print(message)

    log("Loading ontology ...")
    world, onto = load_ontology(ontology_path)
    catalogue = rule_catalogue(onto)
    log("Ontology loaded (%d SWRL rules)" % len(list(onto.rules())))

    student, asserted = populate(
        onto, nlp_features, checklist, student_info, speech_features
    )
    log("Session asserted as %s" % student.name)

    uid = student.name.split("_")[-1]
    results = {
        "student_id": student.name,
        "consistent": True,
        "error": None,
        "inferred_indicators": [],
        "inferred_states": [],
        "recommended_actions": [],
        "session_statuses": [],
        "primary_stress_state": None,
        "primary_drift_state": None,
        "escalation": None,
        "state_conflicts": [],
        "resolution_validated": False,
        "fired_rules": [],
        "progression": None,
        "asserted_values": asserted,
    }
    history_records = normalise_history(history)

    def reason(pass_name):
        """Run Pellet, returning None on success or an error string."""
        try:
            with onto:
                sync_reasoner_pellet(world, infer_property_values=True, debug=0)
            return None
        except OwlReadyInconsistentOntologyError as exc:
            return (
                "Pellet reported the ontology as inconsistent during %s. " "The usual cause is a datatype mismatch between a value "
                "written from Python and the rdfs:range declared in " "cognitive_drift.owl (Owlready2 writes Python floats as "
                "xsd:decimal). Details: %s" % (pass_name, exc)
            )
        except Exception as exc:  # Java missing, Pellet crash, bad path
            return "Reasoner could not be run during %s: %s: %s" % (
                pass_name,
                type(exc).__name__,
                exc,
            )

    # -- Pass 1: what do the rules entail? -----------------------
    log("Running Pellet (pass 1 of 2) ...")
    failure = reason("pass 1")
    if failure:
        results["consistent"] = False
        results["error"] = failure
        log(failure)
        return results

    results["inferred_indicators"] = sorted(
        ind.name for ind in student.exhibitsIndicator
    )
    results["inferred_states"] = sorted(
        state.name for state in student.hasInferredState
    )
    results["recommended_actions"] = sorted(
        action.name for action in student.hasRecommendedAction
    )
    results["session_statuses"] = sorted(
        status.name for status in student.hasSessionStatus
    )
    log(
        "Pass 1: %d indicators, %d states"
        % (len(results["inferred_indicators"]), len(results["inferred_states"]))
    )

    # -- Resolve --------------------------------------------------
    results["state_conflicts"] = state_conflicts(onto, results["inferred_states"])
    results["primary_stress_state"] = resolve_severity(
        results["inferred_states"], STRESS_LADDER
    )
    results["primary_drift_state"] = resolve_severity(
        results["inferred_states"], DRIFT_LADDER
    )
    results["escalation"] = resolve_severity(
        results["recommended_actions"], ACTION_LADDER
    )

    # Closed-world detection statuses (see STATUS_* above).
    closed_world = []
    if not results["inferred_indicators"] and not results["inferred_states"]:
        closed_world.append(STATUS_NO_INDICATORS)
    elif results["inferred_indicators"] and not results["inferred_states"]:
        closed_world.append(STATUS_MONITORING)

    # -- Pass 2: assert the resolution (and, if there is history, the
    #    progression inputs) and let the ontology check/compare them ----
    if validate or history_records:
        log("Asserting resolved outcome and re-running Pellet (pass 2 of 2) ...")
        if validate:
            assert_resolution(
                onto,
                student,
                uid,
                [results["primary_stress_state"], results["primary_drift_state"]],
                closed_world,
            )
        assert_progression_facts(
            onto, student, uid,
            results["primary_stress_state"], results["primary_drift_state"],
            history_records,
        )
        failure = reason("pass 2 (resolution validation)")
        if failure:
            results["consistent"] = False
            results["error"] = (
                "The resolved outcome is not consistent with the ontology. "
                "This means the severity resolution produced a combination "
                "the disjointness axioms forbid. " + failure
            )
            log(results["error"])
            return results
        results["resolution_validated"] = bool(validate)
        results["session_statuses"] = sorted(
            status.name for status in student.hasSessionStatus
        )
        if not validate:
            results["session_statuses"] = sorted(
                set(results["session_statuses"]) | set(closed_world)
            )
        # Progression rules may have added actions (R39, R40) - re-read.
        results["recommended_actions"] = sorted(
            action.name for action in student.hasRecommendedAction
        )
        results["escalation"] = resolve_severity(
            results["recommended_actions"], ACTION_LADDER
        )
        progression = sorted(p.name for p in student.hasProgressionRecord)
        results["progression"] = progression[0] if progression else None
        log("Resolution validated against the ontology")
    else:
        results["session_statuses"] = sorted(
            set(results["session_statuses"]) | set(closed_world)
        )

    asserted["progression"] = {
        "session_count": int((student_info or {}).get("session_count", 1)),
        "history": history_records,
        "current_stress_rank": severity_rank(results["primary_stress_state"], STRESS_LADDER),
        "current_drift_rank": severity_rank(results["primary_drift_state"], DRIFT_LADDER),
    }

    # -- Work out which rule produced each conclusion ------------
    conclusions = (
        results["inferred_indicators"]
        + results["inferred_states"]
        + results["recommended_actions"]
        + ([results["progression"]] if results["progression"] else [])
        + [x for x in results["session_statuses"] if x in PROGRESSION_STATUSES]
    )
    for conclusion in conclusions:
        for rule in catalogue.get(conclusion, []):
            status, evidence = replay_rule(rule["imp"], student)
            if status == "eliminated":
                continue
            record = {key: value for key, value in rule.items() if key != "imp"}
            record["status"] = status
            record["evidence"] = evidence
            results["fired_rules"].append(record)

    results["fired_rules"].sort(key=lambda r: (r["conclusion"], r["rule_id"]))

    # Live handles for explanation.py, which writes Explanation
    # individuals back into this same world. Stripped by to_json().
    results["_session"] = {
        "world": world,
        "onto": onto,
        "student": student,
        "uid": uid,
    }
    return results


def to_json(results):
    """A serialisable copy of the results (drops the live handles)."""
    return {key: value for key, value in results.items() if not key.startswith("_")}


# -------------------------------------------------------------
# SELF TEST
# -------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.append(BASE_DIR)

    # A severe case, using scores in the range pipeline.py produces for strongly negative absolutist self-talk. Run `python pipeline.py` for the real model outputs.
    demo_features = {
        "raw_text": (
            "I always fail everything. I can never understand this material. "
            "My mind goes completely blank in exams. I am totally useless at "
            "this subject. There is no point continuing anymore."
        ),
        "sentiment_score": -0.85,
        "negative_language_score": 0.75,
        "absolutist_term_count": 4,
        "helplessness_score": 0.85,
        "catastrophizing_score": 0.78,
        "overgeneralization_score": 0.70,
        "subject_specific": True,
    }
    demo_checklist = {
        "B1": True, "B2": False, "B3": True,
        "B4": True, "B5": False, "B6": True,
        "B7": True, "B8": True, "B9": True,
        "B10": True, "B11": False, "B12": False,
        "B13": True, "B14": True, "B15": False,
    }
    demo_student = {
        "name": "Test Student",
        "discipline": "Computing",
        "year_of_study": 2,
        "activity_type": "ExamPreparation",
        "workload_level": "High",
        "is_linked_to_event": True,
        "is_acute_or_persistent": "Acute",
    }

    out = run_full_pipeline(demo_features, demo_checklist, demo_student)

    print("\n" + "=" * 60)
    print("REASONING RESULTS")
    print("=" * 60)
    if not out["consistent"]:
        print("FAILED: %s" % out["error"])
        raise SystemExit(1)

    print("Indicators (%d):" % len(out["inferred_indicators"]))
    for name in out["inferred_indicators"]:
        print("   - %s" % name)
    print("States (%d):" % len(out["inferred_states"]))
    for name in out["inferred_states"]:
        print("   - %s" % name)
    print("Support outputs (%d):" % len(out["recommended_actions"]))
    for name in out["recommended_actions"]:
        print("   - %s" % name)
    print("Session statuses:      %s" % (out["session_statuses"] or "none"))
    print("Headline stress state: %s" % out["primary_stress_state"])
    print("Headline drift state:  %s" % out["primary_drift_state"])
    print("Escalation:            %s" % out["escalation"])
    print("Resolution validated:  %s" % out["resolution_validated"])
    if out["state_conflicts"]:
        print(
            "\nRule-set audit - the ontology declares these fired states "
            "mutually disjoint:"
        )
        for clash in out["state_conflicts"]:
            print("   %s vs %s" % (clash["states"][0], clash["states"][1]))

    print("\nRules that fired, with the evidence for each:")
    for rule in out["fired_rules"]:
        print("\n   %s  [%s]  ->  %s" % (
            rule["label"], rule["status"], rule["conclusion"]
        ))
        for item in rule["evidence"]:
            if item["kind"] == "test":
                print("       %s %s %s" % (
                    item["observed"], item["operator"], item["threshold"]
                ))
            else:
                print("       %s = %s" % (item["property"], item["value"]))
