"""
Analysis worker for the live Cognitive Drift system.

Polls Supabase for check-ins with status='pending', runs the SAME pipeline as the prototype (NLP features -> speech features -> OWL/SWRL/Pellet reasoning ->
explanation) and writes the result to `analyses` (and `alerts` when the ontology infers CounselorAlert_Output).

Run it from the folder that holds pipeline.py / reasoner.py / explanation.py /cognitive_drift.owl (set SYSTEM_DIR if they live elsewhere). Needs Java for Pellet.
"""
import os, sys, json, time, hashlib, traceback, argparse

SYSTEM_DIR = os.environ.get("SYSTEM_DIR", os.getcwd())
sys.path.insert(0, SYSTEM_DIR)

from supabase import create_client            # noqa: E402
from pipeline import extract_features, extract_speech_features   # noqa: E402
from reasoner import run_full_pipeline         # noqa: E402
from explanation import build_explanation      # noqa: E402

URL = os.environ["SUPABASE_URL"]
KEY = os.environ["SUPABASE_SERVICE_KEY"]
BUCKET = "voice-recordings"
POLL_SECONDS = 4
MAX_ATTEMPTS = 3
PIPELINE_VERSION = "live-1"
OWL_PATH = os.path.join(SYSTEM_DIR, "cognitive_drift.owl")

sb = create_client(URL, KEY)


def jsonable(x):
    """Round-trip through JSON so numpy types etc. become plain values."""
    return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def ontology_hash():
    with open(OWL_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def register_ontology_version(h):
    sb.table("ontology_versions").upsert(
        {"ontology_hash": h, "label": "auto-registered by worker"}, on_conflict="ontology_hash",
        ignore_duplicates=True).execute()


def claim_next():
    rows = (sb.table("checkins").select("*").eq("status", "pending")
            .order("submitted_at").limit(1).execute().data)
    if not rows:
        return None
    row = rows[0]
    # atomic claim: only succeeds if it is still pending
    got = (sb.table("checkins").update({"status": "processing", "attempts": (row.get("attempts") or 0) + 1})
           .eq("id", row["id"]).eq("status", "pending").execute().data)
    return got[0] if got else None


def process(ci, h):
    student = sb.table("students").select("discipline,academic_year,counsellor_id").eq("id", ci["student_id"]).single().execute().data
    text = ci["self_talk_text"] or ""
    t_start = time.perf_counter()
    features = extract_features(text)
    t_feat = time.perf_counter() - t_start
    t_speech = 0.0

    speech = None
    if ci.get("has_audio") and ci.get("audio_path"):
        try:
            audio = None
            for attempt in range(3):          # storage download can drop on a flaky connection
                try:
                    audio = sb.storage.from_(BUCKET).download(ci["audio_path"]); break
                except Exception as e:
                    print("  audio download attempt %d failed: %r" % (attempt + 1, e)); time.sleep(2)
            if not audio:
                raise RuntimeError("could not download %s" % ci["audio_path"])
            print("  audio downloaded: %d bytes" % len(audio))
            _ts = time.perf_counter()
            speech = extract_speech_features(audio)
            t_speech = time.perf_counter() - _ts
            if speech is None:
                print("  AUDIO NOT ANALYSED: file was not a readable WAV, or too short / silent (see 'WAV decode error' above if shown)")
            else:
                print("  speech analysed: %d words transcribed" % len((speech.get("transcribed_text") or "").split()))
        except Exception:
            print("  AUDIO NOT ANALYSED - continuing with text only. Reason:")
            traceback.print_exc()

    # previous finished sessions of this student -> loaded into the ontology as history
    hist = (sb.table("checkins").select("seq,analyses(primary_stress_state,primary_drift_state)")
            .eq("student_id", ci["student_id"]).lt("seq", ci.get("seq") or 1)
            .order("seq", desc=True).limit(2).execute().data or [])
    history = []
    for r in hist:
        a = r.get("analyses")
        a = a[0] if isinstance(a, list) and a else a
        if a:
            history.append({"seq": r["seq"], "primary_stress_state": a.get("primary_stress_state"),
                            "primary_drift_state": a.get("primary_drift_state")})

    checklist = {("B%d" % i): bool((ci.get("checklist") or {}).get("B%d" % i)) for i in range(1, 16)}
    _tr = time.perf_counter()
    results = run_full_pipeline(
        nlp_features=features,
        checklist=checklist,
        student_info={   # no real name is sent to the reasoner - only the pseudonymous student id
            "name": ci["student_id"],
            "discipline": student.get("discipline") or "",
            "year_of_study": student.get("academic_year") or 1,
            "session_count": ci.get("seq") or 1,
            "activity_type": ci["activity_type"],
            "workload_level": ci["workload_level"],
            "is_linked_to_event": bool(ci["linked_to_event"]),
        },
        speech_features=speech,
        history=history,
        verbose=False,
    )
    if not results.get("consistent"):
        raise RuntimeError("Ontology inconsistent: %s" % results.get("error"))
    expl = build_explanation(results, raw_text=text, self_rating=ci.get("stress_label"))
    t_reason = time.perf_counter() - _tr
    print("  TIMING  text features %.2fs | speech %.2fs | reasoning+explanation %.2fs | TOTAL %.2fs (%s)" % (
        t_feat, t_speech, t_reason, t_feat + t_speech + t_reason, "text+voice" if speech else "text-only"))

    row = {
        "checkin_id": ci["id"],
        "transcript": (speech or {}).get("transcribed_text"),
        "nlp_features": jsonable(features),
        "speech_features": jsonable(speech) if speech else None,
        "consistent": True,
        "inferred_indicators": jsonable(results.get("inferred_indicators", [])),
        "inferred_states": jsonable(results.get("inferred_states", [])),
        "recommended_actions": jsonable(results.get("recommended_actions", [])),
        "session_statuses": jsonable(results.get("session_statuses", [])),
        "primary_stress_state": results.get("primary_stress_state"),
        "primary_drift_state": results.get("primary_drift_state"),
        "escalation": results.get("escalation"),
        "progression": results.get("progression"),
        "state_conflicts": jsonable(results.get("state_conflicts")),
        "fired_rules": jsonable(results.get("fired_rules")),
        "student_message": expl.get("student_message"),
        "counselor_report": expl.get("counselor_report"),
        "evidence_phrases": jsonable(expl.get("evidence_phrases")),
        "ontology_hash": h,
        "pipeline_version": PIPELINE_VERSION,
    }
    sb.table("analyses").upsert(row, on_conflict="checkin_id").execute()

    if results.get("escalation") == "CounselorAlert_Output" or expl.get("safety_flag"):
        sb.table("alerts").upsert(
            {"checkin_id": ci["id"], "student_id": ci["student_id"],
             "counsellor_id": student["counsellor_id"], "level": "support"},
            on_conflict="checkin_id", ignore_duplicates=True).execute()

    sb.table("checkins").update({"status": "done", "processing_error": None}).eq("id", ci["id"]).execute()
    print("  done  escalation=%s  stress=%s  drift=%s" % (
        results.get("escalation"), results.get("primary_stress_state"), results.get("primary_drift_state")))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--once", action="store_true")
    once = ap.parse_args().once
    h = ontology_hash(); register_ontology_version(h)
    # a crash may have left rows half-processed; this assumes ONE worker is running
    sb.table("checkins").update({"status": "pending"}).eq("status", "processing").execute()
    print("Worker started. ontology hash:", h)
    while True:
        try:
            ci = claim_next()
        except Exception as e:      # network drop (e.g. "Server disconnected"): wait and retry, do not crash
            print("  connection problem, retrying in 10s:", repr(e)); time.sleep(10); continue
        if ci is None:
            if once:
                break
            time.sleep(POLL_SECONDS); continue
        print("Analysing check-in", ci["id"], "(attempt %s)" % ci.get("attempts"))
        try:
            process(ci, h)
        except Exception as e:
            traceback.print_exc()
            final = (ci.get("attempts") or 1) >= MAX_ATTEMPTS
            try:
                sb.table("checkins").update({"status": "failed" if final else "pending",
                                             "processing_error": str(e)[:500]}).eq("id", ci["id"]).execute()
            except Exception as e2:
                print("  could not record failure (connection):", repr(e2)); time.sleep(10)
    print("Queue empty - exiting (--once).")


if __name__ == "__main__":
    main()
