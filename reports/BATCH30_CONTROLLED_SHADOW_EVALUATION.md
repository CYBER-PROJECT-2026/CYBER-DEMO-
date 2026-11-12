# AEGIS Batch 30 — Controlled Edge-IIoTset V4 Shadow Evaluation

## Final decision

**SHADOW EVALUATION BLOCKED — AUTHORIZATION**

The authorization and unsealed-data gate could not be verified. Per the stop rule, no capture was selected or opened and no inference was run.

## Authorization evidence and blocker

- Requested source identity: Kaggle dataset `mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot`.
- Requested root: `/kaggle/input/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot/Edge-IIoTset dataset`.
- The exact root does not exist in this execution environment (`Test-Path=False`). This was an existence check only; no directory enumeration or packet access occurred.
- Batch 29.1 states that data-source authorization remained outstanding. No owner authorization reference, purpose approval, retention terms, or confirmation that the selected partition is unsealed was supplied.
- The Kaggle dataset metadata endpoint was checked but returned no readable metadata/license content: [Kaggle dataset metadata](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot/metadata). The associated publication describes Edge-IIoTset and reports dataset availability under a Creative Commons license, but that does not establish the applicable terms for this Kaggle copy or authorization for this specific evaluation: [Edge-IIoTset publication](https://doi.org/10.1109/ACCESS.2022.3165809).
- Partition exclusion is unverified. There is no source manifest or authorization evidence establishing that the proposed captures exclude Partition C and Kitsune SSDP holdout.
- Accordingly, no captures were selected. No quarantined/unverified capture was treated as labeled ground truth.

## Execution scope and results

Execution stopped at the authorization gate, before packet contents or capture labels were accessed.

- PCAP scans / packet-content reads: **NO**
- Real dataset access / generation: **NO**
- Classifier loading / inference: **NO**
- Training / fine-tuning / calibration / threshold tuning / promotion: **NO**
- Sealed Partition C or Kitsune SSDP holdout access: **NO**
- Shadow or production incidents / notifications: **NO**
- Automated blocking: **NO** (`automated_blocking_allowed=False` remains required)

No evaluation output was produced. Parser accepted/rejected counts, timestamp/reorder/overflow counts, eligible/partial window counts, protocol/direction breakdowns, model confidence, OOD, abstention, confusion matrix, precision/recall/F1, false-positive rate, and confidence intervals are **not measured**. No claim about Edge-IIoTset V4 performance is made.

The existing reports preserve the frozen V4 constraints for any separately authorized future run: ordered canonical 29 features, exactly 50 packets per eligible window, symmetric bidirectional 3-tuple identity, `len(buf)` packet length, frozen destination-local directionality, corrected causal topology/timestamp handling, and diagnostic/advisory-only partial windows. The separate 39-feature context schema is not for V4 inference. No before/after artifact hash run was performed because execution did not pass authorization; prior Batch 28.2 and 29.1 evidence recorded all seven hashes unchanged.

## Evidence required to resume

Provide and record all of the following before any packet content is read:

1. Exact source/version or immutable dataset hash and the selected capture list.
2. Owner/rights authorization for this evaluation purpose, plus applicable license/usage terms for the Kaggle copy.
3. Explicit written confirmation that the selected data are unsealed and exclude Partition C and Kitsune SSDP holdout.
4. Label availability and independent adjudication provenance, with quarantined/unverified labels excluded.
5. Permitted purpose and retention/storage/deletion constraints.

No evaluation occurred in this batch. No retraining, calibration, threshold tuning, model promotion, or blocking occurred.

**SHADOW EVALUATION BLOCKED — AUTHORIZATION**
