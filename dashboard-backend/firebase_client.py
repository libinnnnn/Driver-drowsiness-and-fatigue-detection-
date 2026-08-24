"""
Firebase Firestore integration for the Driver Fatigue Detection dashboard.

Mirrors alert events and session summaries to a Firestore collection so a
fleet manager / remote viewer can see them outside the local dashboard.

Designed to fail SAFE: if credentials are missing, misconfigured, or the
firebase_admin package errors for any reason, every function here becomes a
no-op that logs a single warning instead of raising. The rest of the app
(local SQLite logging, the socket relay, the REST API) keeps working exactly
as before regardless of whether Firebase is configured.
"""

import os
import time
import threading

_db = None
_init_attempted = False
_enabled = False

# A single background worker handles all Firestore writes so a slow/unreachable
# network (SSL issues, offline, etc.) can never block the live socket/detection loop.
_write_lock = threading.Lock()
_write_busy = False
FIRESTORE_TIMEOUT_SECONDS = 8


def _initialize():
    """Attempt to initialize the Firebase Admin SDK exactly once."""
    global _db, _init_attempted, _enabled

    if _init_attempted:
        return
    _init_attempted = True

    # Default location: firebase-credentials.json sitting right next to this file
    # (dashboard-backend/firebase-credentials.json). The env var, if set, overrides this.
    default_cred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "firebase-credentials.json")
    cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH") or default_cred_path

    if not os.path.isfile(cred_path):
        print(f"[FIREBASE] Credentials file not found at '{cred_path}' - Firestore sync disabled "
              f"(local SQLite logging still active). Set FIREBASE_CREDENTIALS_PATH to override this location.")
        return

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        _db = firestore.client()
        _enabled = True
        print("[FIREBASE] Firestore initialized successfully - alert/session sync enabled.")
    except Exception as exc:
        print(f"[FIREBASE] Initialization failed ({exc}) - Firestore sync disabled, "
              f"app continues normally with local storage only.")
        _db = None
        _enabled = False


def is_enabled():
    """Returns True only if Firestore initialized successfully."""
    if not _init_attempted:
        _initialize()
    return _enabled


def _run_in_background(fn):
    """Runs fn() on a daemon thread so a slow/unreachable Firestore connection
    can never block the caller (the live socket/detection loop)."""
    global _write_busy
    with _write_lock:
        if _write_busy:
            # A previous write is still in flight (e.g. stuck retrying due to a
            # network/SSL issue) - skip this one rather than piling up threads.
            return
        _write_busy = True

    def _worker():
        global _write_busy
        try:
            fn()
        except Exception as exc:
            print(f"[FIREBASE] background write failed (non-fatal): {exc}")
        finally:
            with _write_lock:
                _write_busy = False

    threading.Thread(target=_worker, daemon=True).start()


def push_alert(session_id, alert_data):
    """Mirror a single alert event to Firestore. Never raises, never blocks the caller."""
    if not is_enabled():
        return

    def _do_push():
        doc = dict(alert_data)
        doc["_session_id"] = session_id
        doc["_synced_at"] = time.time()
        _db.collection("driver_alerts").add(doc, timeout=FIRESTORE_TIMEOUT_SECONDS)

    _run_in_background(_do_push)


def push_session_summary(session_id, summary_data):
    """Upsert the current session's summary snapshot to Firestore. Never raises, never blocks the caller."""
    if not is_enabled():
        return

    def _do_push():
        doc = dict(summary_data)
        doc["_updated_at"] = time.time()
        _db.collection("driver_sessions").document(str(session_id)).set(
            doc, merge=True, timeout=FIRESTORE_TIMEOUT_SECONDS
        )

    _run_in_background(_do_push)