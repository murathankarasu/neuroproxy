# Privacy and data protection

**This is an engineering description, not legal advice.** It sets out what the
system does and does not do with participants' data, so that counsel has
something accurate to reason about. Whether a given study is lawful, on which
basis, with which notices and which controller/processor split, is not answered
here and must not be inferred from it.

## What is collected

| | |
|---|---|
| **Video** | Never leaves the participant's device. Never received, never stored. |
| **Per-frame colour traces** | Computed in the browser, consumed in the browser, discarded there. |
| **Derived state** | Heart rate, deviation from that person's own baseline, signal quality, confidence. About one record per second. |
| **Events** | Marks the researcher places on the timeline. |
| **Consent record** | Timestamp, notice version, and the full text the participant was shown. |

Identifiers held: a random session code. No name, email, address or IP is
recorded by this system. A researcher who needs to join sessions to their own
records supplies a pseudonymous `external_ref`; what that points to is theirs,
and so is the lawful basis for holding it.

## Why the video claim is structural, not a promise

Features are extracted in the page (`apps/web_demo/extractor.js`) and only the
result is transmitted -- roughly 100 bytes per frame. The server has no code
path that decodes or writes an image on the feature route, and the store schema
(`api/store.py`) has no column that could hold one.

The older frame-push route still exists for offline replay and clients that
cannot run an extractor. **Studies with participants should use the feature
route.** The distinction is not cosmetic: it is the difference between "we do
not keep video" and "we never have it".

## Special category data

Camera-derived heart rate is data concerning health. In the EU and UK that is
likely special category under GDPR Article 9, which requires a condition beyond
an ordinary lawful basis, and explicit consent is only one of the available
conditions.

**Get advice before running a study with EU/UK participants.** Points that will
come up:

- which Article 9 condition applies, and whether explicit consent is workable
  in the recruitment context (consent given under an imbalance of power, for
  example an employer's study, is generally not freely given);
- controller and processor roles between the platform and the researcher;
- whether a DPIA is required -- for systematic monitoring of a physiological
  signal at scale, assume yes until told otherwise;
- transfers, if any component is hosted outside the region.

## Mechanisms implemented

These are engineering problems and are solved; they do not by themselves make a
study compliant.

- **Affirmative, itemised consent** before any ingest. Both socket routes refuse
  a session with no recorded grant (`_require_consent`), so the gate is on the
  data path rather than on the interface.
- **Versioned notice.** The exact text shown is stored with the grant. Consent
  against a superseded version is rejected rather than silently accepted, so a
  later edit to the notice cannot re-attribute an old agreement.
- **No partial consent.** There is no mode where the camera runs but the
  measurements are not recorded, so no checkbox pretends otherwise.
- **Withdrawal erases.** `POST /v1/sessions/{id}/withdraw` deletes the samples,
  events, consent record and session row. A withdrawal that flags a row and
  relies on every future query to filter it is not a withdrawal.
- **Retention is enforced, not documented.** Each study carries a retention
  window with no unbounded option; `POST /v1/admin/purge` applies it and is
  meant to run on a schedule.
- **Access requests.** `GET /v1/sessions/{id}/export` returns the complete
  record held for a session.

## Uses this system must not be put to

From the design document's own exclusions, and independently required by the
RAIL licence on vendored model code (see [NOTICE.md](../NOTICE.md)), which
obliges these restrictions to be **passed through to downstream users**:

- medical diagnosis, or any clinical claim;
- employee monitoring, hiring or performance decisions;
- insurance pricing or claim decisions;
- inferring emotion, deception or intent;
- covert measurement, or any use without the person's knowledge.

The output contract is built to make overreach harder: state is reported as
deviation from that person's own baseline in bpm, never as an absolute score,
and is withheld entirely when the signal does not support it.
