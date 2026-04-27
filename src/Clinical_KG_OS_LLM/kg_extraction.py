"""
Unified KG Extraction Pipeline
==============================
Extract clinical knowledge graphs from transcripts using a single-pass LLM call
(OpenRouter `z-ai/glm-4.7-flash`).

Usage:
    python kg_extraction.py --output baseline_naive/sub_kgs

After extraction, merge with:
    python dump_graph.py --input baseline_naive/sub_kgs --output baseline_naive/
"""

import json
import re
import argparse
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from Clinical_KG_OS_LLM.paths import transcripts_dir

# === Tool definitions ===

GET_TURN_TOOL = {
    "type": "function",
    "function": {
        "name": "get_turn",
        "description": "Retrieve the full text of a specific transcript turn by its ID (e.g. 'D-52', 'P-1').",
        "parameters": {
            "type": "object",
            "properties": {
                "turn_id": {"type": "string", "description": "Turn ID like 'D-52' or 'P-1'"}
            },
            "required": ["turn_id"]
        }
    }
}

SEARCH_TRANSCRIPT_TOOL = {
    "type": "function",
    "function": {
        "name": "search_transcript",
        "description": "Search the transcript for turns containing a keyword. Returns matching turns with their IDs.",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Word or phrase to search for"}
            },
            "required": ["keyword"]
        }
    }
}


PROPOSE_NODE_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_node",
        "description": "Propose adding a new node needed for an edge but missing from the node list. Python verifies it exists in the transcript before adding. Returns the new node ID if added, or status 'not_found' if absent from transcript.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The node text to add"},
                "type": {"type": "string", "description": "Node type: SYMPTOM, DIAGNOSIS, TREATMENT, PROCEDURE, LOCATION, MEDICAL_HISTORY, LAB_RESULT"},
                "reason": {"type": "string", "description": "Which edge this node is needed for"}
            },
            "required": ["text", "type", "reason"]
        }
    }
}


# === Transcript search utilities ===
def check_node_in_transcript(node_text: str, transcript: str, index: 'TranscriptIndex' = None) -> dict:
    """Check if node text appears in transcript — exact, partial-word, or stem match.
    When the match is in a doctor question turn, includes the patient's reply so the
    reviewer can detect negated mentions (e.g. 'joint pains? ... Uh no.')."""
    check_text = re.sub(r'^absent\s+', '', node_text.lower().strip())

    if index is not None:
        candidates = index.matching_turns(check_text)
        if candidates:
            for turn_id in index.turn_order:
                if turn_id not in candidates:
                    continue
                block = index.turn_block[turn_id]
                if check_text in block.lower():
                    if turn_id.startswith('D-'):
                        turn_num = turn_id.split('-')[1]
                        d_text = index.turn_text.get(f"D-{turn_num}", "")
                        p_text = index.turn_text.get(f"P-{turn_num}", "")
                        evidence = f"[D-{turn_num}] {d_text}"
                        if p_text:
                            evidence += f"  →  [P-{turn_num}] {p_text}"
                    else:
                        char_pos = block.lower().find(check_text)
                        evidence = block[max(0, char_pos - 40):char_pos + len(check_text) + 40].strip()
                    return {"matched": True, "match_type": "exact", "evidence": evidence}
            return {"matched": True, "match_type": "partial_words", "evidence": None}
        return {"matched": False, "match_type": "none", "evidence": None}

    trans_lower = transcript.lower()

    if check_text in trans_lower:
        idx = trans_lower.index(check_text)

        # Find which turn contains this match (last turn marker before idx)
        turn_id_match = None
        for m in re.finditer(r'\[([DP]-(\d+))\]', transcript):
            if m.start() > idx:
                break
            turn_id_match = m

        if turn_id_match and turn_id_match.group(1).startswith('D-'):
            # Doctor question — append patient response for negation context
            turn_num = turn_id_match.group(2)
            d_turn = get_turn(f"D-{turn_num}", transcript)
            p_turn = get_turn(f"P-{turn_num}", transcript)
            evidence = f"[D-{turn_num}] {d_turn.get('text', '')}"
            if p_turn.get('text'):
                evidence += f"  →  [P-{turn_num}] {p_turn['text']}"
        else:
            start, end = max(0, idx - 40), min(len(transcript), idx + len(check_text) + 40)
            evidence = transcript[start:end].strip()

        return {"matched": True, "match_type": "exact", "evidence": evidence}

    words = re.findall(r'\b[a-z]{3,}\b', check_text)
    if words:
        if all(re.search(r'\b' + re.escape(w), trans_lower) for w in words):
            return {"matched": True, "match_type": "partial_words", "evidence": None}
        stems = [w[:max(4, len(w) - 2)] for w in words if len(w) > 4]
        if stems and all(s in trans_lower for s in stems):
            return {"matched": True, "match_type": "stem", "evidence": None}

    return {"matched": False, "match_type": "none", "evidence": None}


def get_turn(turn_id: str, transcript: str, index: 'TranscriptIndex' = None) -> dict:
    """Return full text of a transcript turn by ID."""
    if index is not None:
        text = index.turn_text.get(turn_id)
        if text is not None:
            return {"turn_id": turn_id, "text": text}
        return {"turn_id": turn_id, "text": None, "error": "turn not found"}
    pattern = rf'\[{re.escape(turn_id)}\]\s*[DP]:\s*(.+?)(?=\n\n\[|\Z)'
    match = re.search(pattern, transcript, re.DOTALL)
    if match:
        return {"turn_id": turn_id, "text": match.group(1).strip()}
    return {"turn_id": turn_id, "text": None, "error": "turn not found"}


def search_transcript(keyword: str, transcript: str, index: 'TranscriptIndex' = None, include_adjacent: bool = False) -> dict:
    """Return all turns containing a keyword. With include_adjacent=True, each result
    also carries the immediately preceding and following turn blocks for context."""
    if index is not None:
        matching = index.matching_turns(keyword)
        results = []
        for turn_id in index.turn_order:
            if turn_id not in matching:
                continue
            entry = {"turn_id": turn_id, "text": index.turn_block[turn_id]}
            if include_adjacent:
                pos = index.turn_pos[turn_id]
                adj = []
                if pos > 0:
                    adj.append(index.turn_block[index.turn_order[pos - 1]])
                if pos + 1 < len(index.turn_order):
                    adj.append(index.turn_block[index.turn_order[pos + 1]])
                if adj:
                    entry["adjacent_turns"] = adj
            results.append(entry)
        return {"keyword": keyword, "matches": results}
    keyword_lower = keyword.lower()
    results = []
    for block in transcript.split('\n\n'):
        if keyword_lower in block.lower():
            m = re.match(r'\[([DP]-\d+)\]', block.strip())
            results.append({"turn_id": m.group(1) if m else None, "text": block.strip()})
    return {"keyword": keyword, "matches": results}


class TranscriptIndex:
    """Pre-built lookup structures for O(1) turn retrieval and keyword search.

    Build once per transcript; pass into dispatch closures so the LLM tool
    handlers skip the O(T) scan on every call.
    """

    def __init__(self, transcript: str):
        self.transcript = transcript
        self.turn_order: list = []       # turn_ids in document order
        self.turn_pos: dict = {}         # turn_id → position in turn_order
        self.turn_text: dict = {}        # turn_id → text only (no marker prefix)
        self.turn_block: dict = {}       # turn_id → full raw block
        self._inv: dict = {}             # token/stem → set of turn_ids
        self._build(transcript)

    @staticmethod
    def _stem(s: str) -> str:
        return s[:max(4, len(s) - 2)] if len(s) > 5 else s

    def _build(self, transcript: str):
        inv: dict = defaultdict(set)
        for block in transcript.split('\n\n'):
            block = block.strip()
            if not block:
                continue
            m = re.match(r'\[([DP]-\d+)\]', block)
            if not m:
                continue
            turn_id = m.group(1)
            tm = re.match(r'\[[DP]-\d+\]\s*[DP]:\s*(.+)', block, re.DOTALL)
            text = tm.group(1).strip() if tm else block

            pos = len(self.turn_order)
            self.turn_order.append(turn_id)
            self.turn_pos[turn_id] = pos
            self.turn_text[turn_id] = text
            self.turn_block[turn_id] = block

            for tok in re.findall(r'\b[a-z0-9]{3,}\b', block.lower()):
                inv[tok].add(turn_id)
                stem = self._stem(tok)
                if stem != tok:
                    inv[stem].add(turn_id)
        self._inv = dict(inv)

    def matching_turns(self, phrase: str) -> set:
        """Return turn_ids where every token of phrase appears (exact or stem)."""
        tokens = re.findall(r'\b[a-z0-9]{3,}\b', phrase.lower())
        if not tokens:
            return set()
        sets = []
        for tok in tokens:
            stem = self._stem(tok)
            s = self._inv.get(tok, set()) | self._inv.get(stem, set())
            sets.append(s)
        result = sets[0].copy()
        for s in sets[1:]:
            result &= s
        return result


# Edge type patterns derived from human-curated KG ground truth
VALID_EDGE_PATTERNS = {
    ("SYMPTOM", "DIAGNOSIS"):        ["INDICATES", "RULES_OUT"],
    ("SYMPTOM", "LOCATION"):         ["LOCATED_AT"],
    ("SYMPTOM", "MEDICAL_HISTORY"):  ["INDICATES", "RULES_OUT"],
    ("SYMPTOM", "SYMPTOM"):          ["CAUSES"],
    ("TREATMENT", "DIAGNOSIS"):      ["TAKEN_FOR"],
    ("TREATMENT", "MEDICAL_HISTORY"):["TAKEN_FOR"],
    ("TREATMENT", "SYMPTOM"):        ["TAKEN_FOR", "CAUSES"],
    ("PROCEDURE", "DIAGNOSIS"):      ["RULES_OUT", "INDICATES"],
    ("PROCEDURE", "LOCATION"):       ["LOCATED_AT"],
    ("MEDICAL_HISTORY", "DIAGNOSIS"):["CAUSES", "INDICATES"],
    ("MEDICAL_HISTORY", "MEDICAL_HISTORY"): ["CAUSES", "INDICATES"],
    ("MEDICAL_HISTORY", "SYMPTOM"):  ["CAUSES"],
    ("MEDICAL_HISTORY", "LOCATION"): ["LOCATED_AT"],
    ("LAB_RESULT", "SYMPTOM"):       ["CONFIRMS"],
    ("LAB_RESULT", "DIAGNOSIS"):     ["CONFIRMS"],
    ("DIAGNOSIS", "DIAGNOSIS"):      ["CAUSES", "INDICATES"],
    ("DIAGNOSIS", "LOCATION"):       ["LOCATED_AT"],
    ("DIAGNOSIS", "SYMPTOM"):        ["CAUSES"],
}


def validate_edge_type(source_type: str, target_type: str) -> dict:
    """Return allowed edge types for a source-target node type pair."""
    key = (source_type.upper(), target_type.upper())
    allowed = VALID_EDGE_PATTERNS.get(key, [])
    return {
        "source_type": source_type,
        "target_type": target_type,
        "allowed_edge_types": allowed,
        "valid_combination": len(allowed) > 0
    }


def enumerate_candidate_pairs(nodes: list) -> list:
    """Return all schema-valid (source, target) node pairs using VALID_EDGE_PATTERNS."""
    pairs = []
    for src in nodes:
        for tgt in nodes:
            if src["id"] == tgt["id"]:
                continue
            src_type = src.get("type", "").upper()
            tgt_type = tgt.get("type", "").upper()
            allowed = VALID_EDGE_PATTERNS.get((src_type, tgt_type), [])
            if allowed:
                pairs.append({
                    "pair_id": f"{src['id']}→{tgt['id']}",
                    "source": src,
                    "target": tgt,
                    "allowed_types": allowed,
                })
    return pairs


def check_edge_evidence(source_text: str, target_text: str, edge_type: str, transcript: str, index: 'TranscriptIndex' = None) -> dict:
    """Find turns where source and target co-occur (same turn or adjacent turns).

    For INDICATES/CONFIRMS/RULES_OUT: symptoms are discussed in early turns and
    diagnoses in the assessment — they never co-occur. Falls back to checking that
    the source appears anywhere in the transcript AND the target appears in the
    assessment turn (longest doctor turn).

    Uses stem matching so 'isolation' matches 'isolate', 'hydration' matches 'hydrated'.
    """
    if index is not None:
        src_turns = index.matching_turns(source_text)
        tgt_turns = index.matching_turns(target_text)
        matches = []

        same = src_turns & tgt_turns
        for turn_id in index.turn_order:
            if turn_id in same:
                matches.append({"turn_id": turn_id, "text": index.turn_block[turn_id], "same_turn": True})

        for turn_id in index.turn_order:
            if turn_id not in src_turns or turn_id in same:
                continue
            pos = index.turn_pos[turn_id]
            for ni in (pos - 1, pos + 1):
                if 0 <= ni < len(index.turn_order):
                    n_id = index.turn_order[ni]
                    if n_id in tgt_turns:
                        matches.append({
                            "turn_id": turn_id,
                            "text": index.turn_block[turn_id] + "\n" + index.turn_block[n_id],
                            "same_turn": False,
                        })
                        break

        if not matches and edge_type in ("INDICATES", "CONFIRMS", "RULES_OUT"):
            assessment = get_longest_doctor_turn(transcript, index)
            if src_turns and assessment.get("turn_id") in tgt_turns:
                matches.append({
                    "turn_id": assessment.get("turn_id"),
                    "text": f"['{source_text}' mentioned in transcript history; '{target_text}' confirmed in assessment {assessment.get('turn_id')}]",
                    "same_turn": False,
                    "inferred": True,
                })

        return {"source": source_text, "target": target_text, "edge_type": edge_type,
                "evidence_found": bool(matches), "matches": matches}

    src_lower = source_text.lower()
    tgt_lower = target_text.lower()

    def stem(s):
        return s[:max(4, len(s) - 2)] if len(s) > 5 else s

    src_stem = stem(src_lower)
    tgt_stem = stem(tgt_lower)

    def found_in(text, exact, st):
        t = text.lower()
        if exact in t or st in t:
            return True
        # Tokenize hyphenated terms: "covid-19" → ["covid", "19"]; check all tokens present
        tokens = [tok for tok in re.split(r'[^a-z0-9]', exact) if len(tok) >= 3]
        if tokens and all(tok in t for tok in tokens):
            return True
        return False

    blocks = [b for b in transcript.split('\n\n') if b.strip()]
    matches = []

    for i, block in enumerate(blocks):
        src_here = found_in(block, src_lower, src_stem)
        tgt_here = found_in(block, tgt_lower, tgt_stem)

        if src_here and tgt_here:
            m = re.match(r'\[([DP]-\d+)\]', block.strip())
            matches.append({"turn_id": m.group(1) if m else None, "text": block.strip(), "same_turn": True})
        elif src_here or tgt_here:
            neighbor = blocks[i + 1] if i + 1 < len(blocks) else ""
            if (src_here and found_in(neighbor, tgt_lower, tgt_stem)) or \
               (tgt_here and found_in(neighbor, src_lower, src_stem)):
                m = re.match(r'\[([DP]-\d+)\]', block.strip())
                matches.append({
                    "turn_id": m.group(1) if m else None,
                    "text": block.strip() + "\n" + neighbor.strip(),
                    "same_turn": False
                })

    if not matches and edge_type in ("INDICATES", "CONFIRMS", "RULES_OUT"):
        src_anywhere = any(found_in(b, src_lower, src_stem) for b in blocks)
        assessment = get_longest_doctor_turn(transcript)
        tgt_in_assessment = found_in(assessment.get("text", ""), tgt_lower, tgt_stem)
        if src_anywhere and tgt_in_assessment:
            matches.append({
                "turn_id": assessment.get("turn_id"),
                "text": f"['{source_text}' mentioned in transcript history; '{target_text}' confirmed in assessment {assessment.get('turn_id')}]",
                "same_turn": False,
                "inferred": True,
            })

    return {
        "source": source_text,
        "target": target_text,
        "edge_type": edge_type,
        "evidence_found": len(matches) > 0,
        "matches": matches
    }


def get_longest_doctor_turn(transcript: str, index: 'TranscriptIndex' = None) -> dict:
    """Return the assessment/plan turn — longest of the last 6 doctor turns.
    The assessment is always near the end but can be up to 6 doctor turns from last
    (observed max across 20 transcripts: RES0217 D-18, 6th from final D-23)."""
    N = 6
    if index is not None:
        last_n = [(tid, index.turn_text[tid])
                  for tid in index.turn_order if tid.startswith('D-')][-N:]
        if not last_n:
            return {"turn_id": None, "text": ""}
        best = max(last_n, key=lambda x: len(x[1]))
        return {"turn_id": best[0], "text": best[1]}
    turns = [(f"D-{m.group(1)}", m.group(2).strip())
             for m in re.finditer(r'\[D-(\d+)\]\s*D:\s*(.+?)(?=\n\n\[|\Z)', transcript, re.DOTALL)]
    if not turns:
        return {"turn_id": None, "text": ""}
    best = max(turns[-N:], key=lambda x: len(x[1]))
    return {"turn_id": best[0], "text": best[1]}


# === Configuration ===
TRANSCRIPT_DIR = transcripts_dir()
MAX_RETRIES = 3
OPENROUTER_MODEL = "z-ai/glm-4.7-flash"

# === Prompts ===
EXTRACTION_PROMPT = """Extract clinical knowledge graph from transcript.

## NODE TYPES:
- SYMPTOM: Patient-reported or observed symptoms (chest pain, shortness of breath)
- DIAGNOSIS: Active or suspected conditions (COPD exacerbation, pneumonia)
- TREATMENT: Medications, therapies, interventions (Aspirin, Metformin, DASH diet)
- PROCEDURE: Tests, exams, surgeries (ECG, stress test, CT angiography)
- LOCATION: Body parts and anatomical locations (chest, left arm, heart)
- MEDICAL_HISTORY: Pre-existing conditions, risk factors (diabetes, smoking)
- LAB_RESULT: Lab values and vital signs (A1C 7.2%, BP 148/90, BNP elevated)

## EDGE TYPES:
- CAUSES: Risk factor causes condition (smoking CAUSES heart disease)
- INDICATES: Symptom indicates diagnosis (chest pain INDICATES angina)
- LOCATED_AT: Symptom at body location (pain LOCATED_AT chest)
- RULES_OUT: Test rules out condition (ECG RULES_OUT arrhythmia)
- TAKEN_FOR: Treatment for condition (Aspirin TAKEN_FOR angina)
- CONFIRMS: Lab/test confirms diagnosis (elevated BNP CONFIRMS heart failure)

TRANSCRIPT:
{transcript}

## FORMAT REQUIREMENTS:
- Node IDs: "N_001", "N_002", etc.
- turn_id: String format "P-X" or "D-X" (P=Patient, D=Doctor, X=turn number)
  Example: "P-1", "D-39" (from [P-1], [D-39] in transcript)

Output JSON with nodes (id, text, type, evidence, turn_id) and edges (source_id, target_id, type, evidence, turn_id).
Output ONLY valid JSON."""

NODE_EXTRACTION_PROMPT = """You are an experienced clinical physician reviewing a doctor-patient transcript to build a structured medical record. Extract all clinically significant entities.

## NODE TYPES (use ONLY these):
- SYMPTOM: Patient-reported or observed symptoms (chest pain, shortness of breath)
- DIAGNOSIS: Active or suspected conditions (COPD exacerbation, pneumonia)
- TREATMENT: Medications, therapies, interventions (Aspirin, Metformin, DASH diet)
- PROCEDURE: Tests, exams, surgeries (ECG, stress test, CT angiography)
- LOCATION: Body parts and anatomical locations (chest, left arm, heart)
- MEDICAL_HISTORY: Pre-existing conditions, risk factors (diabetes, smoking)
- LAB_RESULT: Lab values and vital signs (A1C 7.2%, BP 148/90, BNP elevated)

## RULES:
- Extract only what is clinically significant — a doctor would document it
- For SYMPTOM nodes: use the patient's own qualifying language when clinically meaningful (e.g. "sharp chest pain" not "chest pain", "persistent dry cough" not "cough"). Only extract symptoms the patient confirmed as present — denied symptoms are not nodes. Do NOT extract vague phrases like "feeling unwell" — use the specific symptom name.
- For DIAGNOSIS nodes: use the full standard name with qualifiers (e.g. "covid-19" not "covid", "viral illness" not "virus"). Extract ALL diagnoses in the assessment including differentials ("could be X", "if not X")
- For PROCEDURE nodes: include what is being tested (e.g. "covid swab" not "swab", "nasal swab" not "swab")
- For TREATMENT nodes: extract the clinical noun, brand name, or the activity phrasing matching transcript wording (e.g. "hydration", "well hydrated", "self-isolation", "isolate for 14 days"). When a patient or doctor names a specific drug (Tylenol, Advil, Ventolin), keep that exact name — do not abstract to a drug class like "analgesics" or "antipyretics".
- For MEDICAL_HISTORY: extract ALL of the following when mentioned — (a) lifestyle negations from "No" answers ("no" to smoking → "non-smoker"; "healthy, no conditions" → "no chronic conditions"); (b) substance use (alcohol use, marijuana use) when confirmed; (c) exposure or contact history (school exposure, contact with confirmed case); (d) family history (family history of epilepsy, family history of diabetes); (e) past injuries or conditions (broken arm, prior hospitalization); (f) medication adherence events (missed medication dose, forgot to take medication); (g) immunization status whether up to date or not.
- For LOCATION nodes: single lowercase anatomical term matching transcript wording (e.g. "chest", "left arm", "throat"). One location per node — do not combine multiple body parts.
- For LAB_RESULT nodes: always include the measured value with units (e.g. "A1C 7.2%", "BP 148/90", "temperature 101 F"). Do not extract a lab name without its value.
- The doctor's final assessment turn is information-dense: extract each diagnosis, treatment, procedure, and lab result as a separate node

TRANSCRIPT:
{transcript}

## FORMAT:
- Node IDs: "N_001", "N_002", etc.
- turn_id: "P-X" (patient) or "D-X" (doctor)
- evidence: exact quote from transcript

Output JSON with ONLY a "nodes" array: [{{"id", "text", "type", "evidence", "turn_id"}}]
Output ONLY valid JSON."""

REVIEW_NODE_PROMPT = """You are a senior clinician verifying extracted clinical KG nodes against a transcript.

Your ONLY job is to decide KEEP or REMOVE for each node. Do NOT rename or normalize node text — canonicalization happens in a separate later pass. Return the same text as given.

For EACH node:
1. Call get_turn(turn_id) to read the source turn.
2. If that turn is a doctor turn (D-X): also call get_turn("P-X") to read the patient's reply at the same number.
3. Read both turns and decide by node type:
    - SYMPTOM: KEEP if the patient confirmed it as present; REMOVE if denied ("No", "not really", "I don't think so")
    - DIAGNOSIS: KEEP if mentioned as active, suspected, or differential ("could be", "rule out", "if not X")
    - TREATMENT / PROCEDURE: KEEP if ordered, recommended, or reported by patient or doctor
    - MEDICAL_HISTORY: KEEP if the patient confirmed it — this includes lifestyle (smoking, alcohol), exposure history, family history, past injuries, medication adherence. REMOVE if the patient explicitly denied it.
    - LOCATION: KEEP if the location appears in a confirmed patient statement or alongside a confirmed symptom. If found only in a doctor's question, check the patient's reply — if the patient denied the associated symptom, REMOVE.
    - LAB_RESULT: KEEP if a numeric value is present in the turn text; REMOVE if no value found (call search_transcript first to verify)
4. If get_turn returns no content or the turn is unrelated: call search_transcript with a related keyword to find supporting context. KEEP if found, REMOVE if nothing supports it.

NODES:
{nodes}

Return ONLY valid JSON with ONLY "id", "text", "type" per node (same text as input — do not rename):
{{"nodes": [{{"id": "N_001", "text": "...", "type": "SYMPTOM"}}]}}"""


ASSESSMENT_NODE_PROMPT = """You are a senior clinician checking whether the doctor's assessment mentions any clinical entities not yet captured in the extracted node list.

DOCTOR'S ASSESSMENT ({assessment_turn_id}):
{assessment_text}

NODES ALREADY EXTRACTED:
{nodes}

The assessment is the most information-dense turn. Add ONLY nodes that:
- Clearly appear in the assessment text above
- Are NOT already covered by an existing node (including paraphrases)
- Are clinically significant (diagnosis, treatment, procedure, lab result)
- Do NOT add SYMPTOM nodes from the assessment — symptoms must come from patient speech. The doctor may mention symptoms the patient "might get" or hypothetical future symptoms; these are not confirmed findings and must not be added.

Assign new IDs continuing from the highest existing ID (e.g. if last is N_017, start at N_018).

Return ONLY valid JSON — empty list if nothing is missing:
{{"nodes": [{{"id": "N_018", "text": "...", "type": "DIAGNOSIS", "evidence": "...", "turn_id": "{assessment_turn_id}"}}]}}"""


EDGE_EXTRACTION_PROMPT = """You are an experienced clinical physician finding relationships between clinical entities.

Use get_turn(turn_id) and search_transcript(keyword) to retrieve evidence from the transcript as needed.
Assessment turn (doctor's final summary): {assessment_turn_id}

## VALID EDGE SCHEMA — hard constraints on direction and type:
Only create edges where the combination appears in this table. Source is LEFT, target is RIGHT.

Source type       | Target type       | Allowed edge types
SYMPTOM           | DIAGNOSIS         | INDICATES, RULES_OUT
SYMPTOM           | LOCATION          | LOCATED_AT
SYMPTOM           | MEDICAL_HISTORY   | INDICATES, RULES_OUT
SYMPTOM           | SYMPTOM           | CAUSES
TREATMENT         | DIAGNOSIS         | TAKEN_FOR
TREATMENT         | MEDICAL_HISTORY   | TAKEN_FOR
TREATMENT         | SYMPTOM           | TAKEN_FOR, CAUSES
PROCEDURE         | DIAGNOSIS         | RULES_OUT, INDICATES
PROCEDURE         | LOCATION          | LOCATED_AT
MEDICAL_HISTORY   | DIAGNOSIS         | CAUSES, INDICATES
MEDICAL_HISTORY   | MEDICAL_HISTORY   | CAUSES, INDICATES
MEDICAL_HISTORY   | SYMPTOM           | CAUSES
MEDICAL_HISTORY   | LOCATION          | LOCATED_AT
LAB_RESULT        | SYMPTOM           | CONFIRMS
LAB_RESULT        | DIAGNOSIS         | CONFIRMS
DIAGNOSIS         | DIAGNOSIS         | CAUSES, INDICATES
DIAGNOSIS         | LOCATION          | LOCATED_AT

## KEY RULES:
- A test ORDERED to exclude a diagnosis → RULES_OUT (not CONFIRMS). CONFIRMS is ONLY valid for LAB_RESULT → DIAGNOSIS or LAB_RESULT → SYMPTOM. Never use CONFIRMS for PROCEDURE nodes.
- INDICATES: only create when the doctor explicitly links a symptom to a specific diagnosis. For alternative/differential diagnoses introduced with "could be" or "if not X", do NOT duplicate INDICATES edges — they share implied symptoms with the primary diagnosis
- TAKEN_FOR: check BOTH early patient turns (patient-reported medications) AND the assessment turn (doctor-recommended treatments)
- LOCATED_AT: the clinical entity (SYMPTOM, PROCEDURE, DIAGNOSIS) is always the SOURCE; LOCATION is always the TARGET. Never put a LOCATION as source.
- If an edge requires a node not in the list: call propose_node(text, type, reason) — Python verifies it exists in the transcript. Only use the returned ID if status is "added"

## SYSTEMATIC NODE CHECKS (do these before finishing):
1. PROCEDURE nodes: for each, call search_transcript(procedure_text) — find what condition it was ordered to test/exclude → RULES_OUT or INDICATES
2. TREATMENT nodes: for each, verify you have a TAKEN_FOR edge. If missing, call search_transcript(treatment_text) → TAKEN_FOR
3. MEDICAL_HISTORY nodes: for each, call search_transcript(history_text) — check if it CAUSES any DIAGNOSIS node
4. LOCATION nodes: for each, call search_transcript(location_text) — find which SYMPTOM was being discussed → LOCATED_AT with SYMPTOM as source, LOCATION as target
5. SYMPTOM nodes: for each, call search_transcript(symptom_text) and check adjacent_turns for a diagnostic link. If no explicit link found, call get_turn({assessment_turn_id}) — doctors often link all symptoms collectively (e.g. "your symptoms overlap with covid-19"). If the assessment names a diagnosis and this patient symptom was reported → SYMPTOM INDICATES DIAGNOSIS.
6. LAB_RESULT nodes: for each, call search_transcript(lab_text) — find the diagnosis the doctor links the result to → CONFIRMS

NODES:
{nodes}

Use the tools to look up evidence, then output ONLY valid JSON:
{{"edges": [{{"source_id": "N_001", "target_id": "N_002", "type": "INDICATES", "evidence": "...", "turn_id": "D-52"}}]}}"""



EDGE_PAIR_CLASSIFICATION_PROMPT = """You are an experienced clinical physician extracting relationships between clinical entities from a doctor-patient transcript.

Python has already enumerated every schema-valid node pair. Your job is to evaluate each candidate pair and return only those where a real clinical relationship exists in the transcript.

## DECISION RULES (apply strictly):
- INDICATES (SYMPTOM→DIAGNOSIS): doctor explicitly links this specific symptom to this diagnosis — in the assessment or during the exam. Do NOT use just because both appear in the transcript.
- INDICATES (MEDICAL_HISTORY→DIAGNOSIS): history is stated as a contributing factor or risk for the diagnosis.
- RULES_OUT: a test or finding was used specifically to exclude this diagnosis.
- TAKEN_FOR: treatment/procedure was prescribed, recommended, or reported for this condition or symptom. Check BOTH patient-reported medications (early turns) AND doctor recommendations (assessment turn).
- CAUSES: explicit causal event or mechanism stated — e.g. "missed medication caused the seizure", "school exposure led to infection". NOT just co-occurrence.
- LOCATED_AT: clinical entity is at this anatomical site — must be stated or clearly implied.
- CONFIRMS: lab result directly confirms this diagnosis or explains this symptom — must be explicitly linked by the doctor.

## HOW TO EVALUATE:
For each candidate pair:
1. Read the source node's evidence turn and the target node's evidence turn.
2. Check the assessment turn for explicit connections.
3. If a relationship is supported by transcript text → include in edges with an exact evidence quote.
4. If no clear support exists → omit the pair entirely.

Only one edge type per pair (the most specific supported one).

## PROPOSING MISSING NODES:
If you find an edge that needs a node not in the list, add it to "proposed_nodes" — Python will verify it against the transcript.

NODES (with evidence):
{nodes}

CANDIDATE PAIRS (evaluate each):
{pairs}

TRANSCRIPT:
{transcript}

Output ONLY valid JSON:
{{"edges": [{{"pair_id": "N_001→N_003", "source_id": "N_001", "target_id": "N_003", "type": "INDICATES", "evidence": "<exact quote>", "turn_id": "D-5"}}],
 "proposed_nodes": [{{"text": "...", "type": "MEDICAL_HISTORY", "evidence": "<quote>", "turn_id": "P-9"}}]}}
Omit "proposed_nodes" if none needed."""


EDGE_EXTRACTION_PROMPT_FULL = """You are an experienced clinical physician extracting relationships between clinical entities from a doctor-patient transcript.

You have the complete transcript and a validated node list with evidence. Use both to reason about relationships — edges often require understanding the full conversation arc, not just adjacent turns.

## VALID EDGE SCHEMA — hard constraints (Source LEFT, Target RIGHT):
Source type       | Target type       | Allowed edge types
SYMPTOM           | DIAGNOSIS         | INDICATES, RULES_OUT
SYMPTOM           | LOCATION          | LOCATED_AT
SYMPTOM           | MEDICAL_HISTORY   | INDICATES, RULES_OUT
SYMPTOM           | SYMPTOM           | CAUSES
TREATMENT         | DIAGNOSIS         | TAKEN_FOR
TREATMENT         | MEDICAL_HISTORY   | TAKEN_FOR
TREATMENT         | SYMPTOM           | TAKEN_FOR, CAUSES
PROCEDURE         | DIAGNOSIS         | RULES_OUT, INDICATES
PROCEDURE         | LOCATION          | LOCATED_AT
MEDICAL_HISTORY   | DIAGNOSIS         | CAUSES, INDICATES
MEDICAL_HISTORY   | MEDICAL_HISTORY   | CAUSES, INDICATES
MEDICAL_HISTORY   | SYMPTOM           | CAUSES
MEDICAL_HISTORY   | LOCATION          | LOCATED_AT
LAB_RESULT        | SYMPTOM           | CONFIRMS
LAB_RESULT        | DIAGNOSIS         | CONFIRMS
DIAGNOSIS         | DIAGNOSIS         | CAUSES, INDICATES
DIAGNOSIS         | LOCATION          | LOCATED_AT
DIAGNOSIS         | SYMPTOM           | CAUSES

## KEY RULES:
- INDICATES (SYMPTOM→DIAGNOSIS): only when the doctor explicitly links this specific symptom to a diagnosis. For collective language ("your symptoms overlap with X"), only create INDICATES if the assessment co-mentions this symptom and the diagnosis in the same sentence. Do NOT create INDICATES just because both appear in the transcript.
- RULES_OUT: a test ORDERED to exclude a diagnosis → RULES_OUT. CONFIRMS is ONLY valid for LAB_RESULT nodes.
- TAKEN_FOR: check BOTH patient-reported medications (early turns) AND doctor-recommended treatments (assessment turn).
- CAUSES: requires an explicit causal event or mechanism — e.g. missed medication dose → seizure, school exposure → infection. Do NOT use CAUSES just because two entities co-occur.
- LOCATED_AT: clinical entity is always SOURCE, LOCATION is always TARGET.

## SYSTEMATIC CHECKS — go through each node type before finishing:
1. PROCEDURE: what condition was it ordered to test/exclude? → RULES_OUT or INDICATES
2. TREATMENT: does it have a TAKEN_FOR edge? Check both early patient turns and the assessment.
3. MEDICAL_HISTORY: does it CAUSE any DIAGNOSIS or SYMPTOM? Look for explicit causal events.
4. LOCATION: which SYMPTOM was being discussed when this location was mentioned? → LOCATED_AT
5. SYMPTOM: is there a doctor turn explicitly linking it to a specific diagnosis? → INDICATES
6. LAB_RESULT: which diagnosis or symptom does the doctor tie the result to? → CONFIRMS

## PROPOSING MISSING NODES:
If an edge requires a node not in the list, add it to "proposed_nodes" — Python will verify it exists in the transcript before accepting it.

NODES:
{nodes}

TRANSCRIPT:
{transcript}

Output ONLY valid JSON:
{{"edges": [{{"source_id": "N_001", "target_id": "N_002", "type": "INDICATES", "evidence": "<exact quote>", "turn_id": "D-52"}}],
 "proposed_nodes": [{{"text": "...", "type": "MEDICAL_HISTORY", "evidence": "<exact quote>", "turn_id": "P-9"}}]}}
Omit "proposed_nodes" if none are needed."""


EDGE_REVIEW_PROMPT = """You are a senior clinician reviewing extracted clinical KG edges for clinical plausibility.

Edge schema has already been validated in Python. Your ONLY job is to remove edges that are clinically nonsensical — where the relationship makes no medical sense regardless of transcript content.

KEEP an edge unless it clearly fails one of these tests:
- A negative medical history fact ("non-smoker", "no X", "never X") INDICATES or CAUSES a specific acute diagnosis → REMOVE (a negative fact cannot indicate an infection)
- An administrative or public health procedure (contact tracing, reporting) INDICATES a diagnosis → REMOVE
- A treatment CAUSES the exact symptom/condition it was prescribed to treat → REMOVE (prescribing Tylenol does not cause headache)
- Exact duplicate: same source_id, target_id, and type already appeared earlier in the list → REMOVE the second occurrence

If an edge has a non-empty evidence field and makes clinical sense → KEEP. Do not second-guess transcript evidence.

EDGES:
{edges}

Return ONLY valid JSON with all kept edges:
{{"edges": [{{"source_id": "N_001", "target_id": "N_002", "type": "INDICATES", "evidence": "...", "turn_id": "..."}}]}}"""


CANONICAL_NODE_PROMPT = """You are a clinical terminologist. Rename each node's text to its standard clinical form.
All output text must be lowercase unless it is a proper brand name or abbreviation.

## Rules and examples drawn from human-curated clinical KGs:

SYMPTOM — standard clinical term, preserve meaningful qualifiers. Absent/denied symptoms use "absent X" format.
  "tiredness" → "fatigue"
  "liquid stools" → "diarrhea"
  "persistent headache" → "headache"
  "can't breathe well" → "difficulty breathing"
  "no fever" → "absent fever"
  "denied chest pain" → "absent chest pain"
  Keep: "dry cough", "loss of smell", "shortness of breath", "chest tightness", "nasal congestion"

DIAGNOSIS — full lowercase standardized name with qualifiers where relevant.
  "covid" → "covid-19"
  "flu" → "influenza"
  "cold" → "viral infection / common cold"
  "ruled out asthma" → "asthma ruled out"
  "possible infection" → "suspected infection"
  Keep: "copd exacerbation", "upper respiratory infection", "viral illness", "lyme disease"

TREATMENT — clinical noun, lowercase. Keep brand names when that is how the treatment is known.
  "isolate for 14 days" → "14-day isolation"
  "well hydrated" → "hydration"
  "anti-inflammatories" → "nsaids"
  "analgesics" → "nsaids"
  "acetaminophen" → "tylenol"
  "ibuprofen" → "advil"
  Keep: "tylenol", "advil", "claritin", "ventolin", "salbutamol", "antibiotics", "steroids", "hydration", "isolation"

PROCEDURE — lowercase, include what is being tested.
  "swab" → "covid swab"
  "listening to lungs" → "lung auscultation"
  Keep: "chest x-ray", "pulse oximetry", "vital signs", "physical exam", "cbc", "lyme serology"

LOCATION — single lowercase anatomical term, no articles or prepositions.
  "the chest area" → "chest"
  "left side of chest" → "chest"
  "in the throat" → "throat"
  Keep: "chest", "throat", "forehead", "top of head", "lungs", "sinuses", "nose"

MEDICAL_HISTORY — lowercase clinical status or condition.
  Keep: "non-smoker", "no chronic conditions", "hypertension", "type 1 diabetes", "cannabis use", "no allergies"

LAB_RESULT — measurement name + value + unit, lowercase.
  "fever 101" → "temperature ~101 f"
  "temp 37.4" → "temperature 37.4 c"

Do NOT change node IDs or types. Only change text.

NODES:
{nodes}

Return ONLY valid JSON:
{{"nodes": [{{"id": "N_001", "text": "canonical name", "type": "SYMPTOM"}}]}}"""




# === Model Client ===
class OpenRouterClient:
    """Client for OpenRouter API (GLM, etc.)"""

    def __init__(self, api_key: str, model: str = OPENROUTER_MODEL):
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model

    def generate(self, prompt: str) -> tuple:
        """Generate response. Returns (content, usage_dict)."""
        for attempt in range(MAX_RETRIES):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    stream=True
                )

                content = ""
                last_chunk = None
                for chunk in stream:
                    last_chunk = chunk
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content += delta.content

                usage = None
                if last_chunk and hasattr(last_chunk, 'usage') and last_chunk.usage:
                    u = last_chunk.usage
                    usage = {
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                    }

                if content:
                    return content, usage

            except Exception as e:
                print(f"(error: {e}, retry {attempt + 1})", end=" ", flush=True)
                time.sleep(2 ** attempt)

        return "", None

    def generate_with_tools(self, prompt: str, tools: list, dispatch: callable, limit: int = 80) -> tuple:
        """Multi-turn generation with tool calling. dispatch(name, args) -> result dict.

        Graceful degradation: at WARN_AT iterations inject a stop signal so the model
        wraps up cleanly. Falls back to last partial assistant output if limit is hit.
        """
        LIMIT = limit
        WARN_AT = LIMIT - 5

        messages = [{"role": "user", "content": prompt}]
        total_prompt = total_completion = 0
        last_content = ""

        for i in range(LIMIT):
            if i == WARN_AT:
                messages.append({
                    "role": "user",
                    "content": "Stop calling tools now. Output your final JSON result immediately."
                })

            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.7
                )
            except Exception as e:
                print(f"(tool-call error: {e})", end=" ", flush=True)
                break

            if resp.usage:
                total_prompt += resp.usage.prompt_tokens or 0
                total_completion += resp.usage.completion_tokens or 0

            msg = resp.choices[0].message
            if msg.content:
                last_content = msg.content
            messages.append(msg)

            if not msg.tool_calls:
                usage = {"prompt_tokens": total_prompt, "completion_tokens": total_completion}
                return msg.content or "", usage

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = dispatch(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result)
                })

        usage = {"prompt_tokens": total_prompt, "completion_tokens": total_completion}
        if last_content:
            print("(warn: tool-call limit reached, using last partial output)", end=" ", flush=True)
            return last_content, usage
        return "", usage


def get_client(api_keys: dict) -> OpenRouterClient:
    key = api_keys.get("openrouter")
    if not key:
        raise SystemExit("api_keys.json must contain a non-empty \"openrouter\" key")
    return OpenRouterClient(key, OPENROUTER_MODEL)


# === Utilities ===
def read_transcript(file_path: Path) -> str:
    with open(file_path, 'r') as f:
        return f.read()


def extract_json_from_response(response_text: str) -> dict:
    """Extract JSON from LLM response."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    if response_text.strip().startswith('```'):
        parts = response_text.split('```')
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith('json'):
                inner = inner[4:]
            inner = inner.strip()
            try:
                return json.loads(inner)
            except json.JSONDecodeError:
                pass

    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_str)
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                pass

    return None


def validate_knowledge_graph(kg: dict) -> dict:
    """Validate and fix knowledge graph integrity."""
    if not kg or 'nodes' not in kg or 'edges' not in kg:
        return kg

    node_ids = {node['id'] for node in kg.get('nodes', [])}

    valid_edges = []
    invalid_count = 0

    for edge in kg.get('edges', []):
        source = edge.get('source_id')
        target = edge.get('target_id')
        if source in node_ids and target in node_ids:
            valid_edges.append(edge)
        else:
            invalid_count += 1

    if invalid_count > 0:
        print(f"    Removed {invalid_count} invalid edges")
        kg['edges'] = valid_edges

    return kg


def get_transcript_files():
    """Get all transcript files."""
    files = []
    for res_dir in sorted(TRANSCRIPT_DIR.glob("RES*")):
        if res_dir.is_dir():
            txt_file = res_dir / f"{res_dir.name}.txt"
            if txt_file.exists():
                files.append(txt_file)
    return files


def review_nodes_with_tool(nodes: list, transcript: str, client: OpenRouterClient, index: 'TranscriptIndex' = None) -> tuple:
    """Review extracted nodes using tool-based transcript verification."""
    idx = index if index is not None else TranscriptIndex(transcript)
    nodes_json = json.dumps(
        [{"id": n["id"], "text": n["text"], "type": n["type"], "turn_id": n.get("turn_id", "")} for n in nodes],
        indent=2
    )
    prompt = REVIEW_NODE_PROMPT.format(nodes=nodes_json)

    def dispatch(name, args):
        if name == "get_turn":
            return get_turn(args["turn_id"], transcript, index=idx)
        if name == "search_transcript":
            return search_transcript(args["keyword"], transcript, index=idx)
        return {"error": f"unknown tool: {name}"}

    content, usage = client.generate_with_tools(prompt, [GET_TURN_TOOL, SEARCH_TRANSCRIPT_TOOL], dispatch, limit=60)

    if content:
        result = extract_json_from_response(content)
        if result and "nodes" in result:
            original_map = {n["id"]: n for n in nodes}
            merged = []
            for n in result["nodes"]:
                orig = original_map.get(n["id"])
                if orig is None:
                    continue  # reject hallucinated IDs not in original
                merged.append({**orig, "text": n["text"]})
            return merged, usage or {}

    return nodes, usage or {}


def extract_edges_with_tools(nodes: list, transcript: str, client: OpenRouterClient, index: 'TranscriptIndex' = None) -> tuple:
    """Extract edges using tool-based transcript lookup. Supports propose_node for mid-pass node discovery.

    Returns (edges, proposed_ids, usage) where proposed_ids tracks nodes added via propose_node
    so the caller can run them through review_nodes_with_tool for type/canonicalization correction.
    """
    idx = index if index is not None else TranscriptIndex(transcript)
    assessment_turn_id = get_longest_doctor_turn(transcript, index=idx).get("turn_id", "unknown")

    def _node_num(n):
        try:
            return int(n["id"].split("_")[1])
        except (IndexError, ValueError):
            return 0

    next_id_ref = [max((_node_num(n) for n in nodes), default=0) + 1]
    proposed_ids = set()
    node_map = {n["id"]: n for n in nodes}

    nodes_summary = json.dumps(
        [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in nodes],
        indent=2
    )
    prompt = EDGE_EXTRACTION_PROMPT.format(nodes=nodes_summary, assessment_turn_id=assessment_turn_id)

    def dispatch(name, args):
        if name == "get_turn":
            return get_turn(args["turn_id"], transcript, index=idx)
        if name == "search_transcript":
            # include_adjacent=True so the LLM sees neighboring turns for relationship context
            return search_transcript(args["keyword"], transcript, index=idx, include_adjacent=True)
        if name == "propose_node":
            text = args.get("text", "").strip()
            node_type = args.get("type", "").upper()
            # Return existing node if already present
            for n in nodes:
                if n["text"].lower() == text.lower() and n["type"] == node_type:
                    return {"status": "already_exists", "id": n["id"]}
            result = check_node_in_transcript(text, transcript, index=idx)
            if result["matched"]:
                new_id = f"N_{next_id_ref[0]:03d}"
                next_id_ref[0] += 1
                new_node = {
                    "id": new_id, "text": text, "type": node_type,
                    "evidence": result.get("evidence") or "",
                    "turn_id": "",
                }
                nodes.append(new_node)
                node_map[new_id] = new_node
                proposed_ids.add(new_id)
                return {"status": "added", "id": new_id}
            return {"status": "not_found", "message": "not in transcript — do not create edges to this node"}
        return {"error": f"unknown tool: {name}"}

    content, usage = client.generate_with_tools(
        prompt, [GET_TURN_TOOL, SEARCH_TRANSCRIPT_TOOL, PROPOSE_NODE_TOOL], dispatch
    )

    if content:
        result = extract_json_from_response(content)
        if isinstance(result, list):
            result = {"edges": result}
        if result and "edges" in result:
            return result["edges"], proposed_ids, usage or {}

    return [], proposed_ids, usage or {}

def schema_filter_edges(edges: list, nodes: list) -> tuple:
    """Remove edges whose (source_type, target_type, edge_type) is not in VALID_EDGE_PATTERNS.
    Returns (valid_edges, dropped_edges)."""
    node_map = {n["id"]: n for n in nodes}
    valid, dropped = [], []
    for e in edges:
        src_type = node_map.get(e.get("source_id"), {}).get("type", "").upper()
        tgt_type = node_map.get(e.get("target_id"), {}).get("type", "").upper()
        allowed = VALID_EDGE_PATTERNS.get((src_type, tgt_type), [])
        if e.get("type", "").upper() in allowed:
            valid.append(e)
        else:
            dropped.append(e)
    return valid, dropped


def review_edges(nodes: list, edges: list, client: OpenRouterClient) -> tuple:
    """Clinical plausibility review of schema-valid edges. No tools — schema already validated in Python."""
    node_map = {n["id"]: n for n in nodes}
    enriched = []
    for e in edges:
        src = node_map.get(e.get("source_id"), {})
        tgt = node_map.get(e.get("target_id"), {})
        enriched.append({
            "source_id": e["source_id"],
            "source_text": src.get("text", ""),
            "source_type": src.get("type", ""),
            "target_id": e["target_id"],
            "target_text": tgt.get("text", ""),
            "target_type": tgt.get("type", ""),
            "type": e["type"],
            "evidence": e.get("evidence", ""),
            "turn_id": e.get("turn_id", "")
        })

    prompt = EDGE_REVIEW_PROMPT.format(edges=json.dumps(enriched, indent=2))
    content, usage = client.generate(prompt)

    if content:
        result = extract_json_from_response(content)
        if isinstance(result, list):
            result = {"edges": result}
        if result and "edges" in result:
            clean = []
            for e in result["edges"]:
                clean.append({
                    "source_id": e["source_id"],
                    "target_id": e["target_id"],
                    "type": e["type"],
                    "evidence": e.get("evidence", ""),
                    "turn_id": e.get("turn_id", "")
                })
            return clean, usage or {}

    return edges, usage or {}


def canonicalize_nodes(nodes: list, client: OpenRouterClient) -> tuple:
    """Rename node texts to standard clinical form. No transcript access needed."""
    nodes_json = json.dumps(
        [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in nodes],
        indent=2
    )
    prompt = CANONICAL_NODE_PROMPT.format(nodes=nodes_json)
    content, usage = client.generate(prompt)
    if not content:
        return nodes, usage or {}
    result = extract_json_from_response(content)
    if isinstance(result, list):
        result = {"nodes": result}
    if not result or "nodes" not in result:
        return nodes, usage or {}
    canonical_map = {n["id"]: n["text"] for n in result["nodes"] if n.get("id") and n.get("text")}
    return [{**n, "text": canonical_map.get(n["id"], n["text"])} for n in nodes], usage or {}


def check_assessment_for_nodes(nodes: list, transcript: str, client: OpenRouterClient, index: 'TranscriptIndex' = None) -> tuple:
    """Additive step: inject assessment text and ask LLM to add any missing nodes."""
    assessment = get_longest_doctor_turn(transcript, index=index)
    if not assessment.get("text"):
        return [], {}

    def _node_num(n):
        try:
            return int(n["id"].split("_")[1])
        except (IndexError, ValueError):
            return 0
    next_id = max((_node_num(n) for n in nodes), default=0) + 1
    nodes_summary = json.dumps(
        [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in nodes], indent=2
    )
    prompt = ASSESSMENT_NODE_PROMPT.format(
        assessment_turn_id=assessment["turn_id"],
        assessment_text=assessment["text"],
        nodes=nodes_summary,
    )
    content, usage = client.generate(prompt)
    if not content:
        return [], usage or {}

    result = extract_json_from_response(content)
    if isinstance(result, list):
        result = {"nodes": result}
    if not result or "nodes" not in result:
        return [], usage or {}

    existing_keys = {(n["text"].lower().strip(), n["type"]) for n in nodes}
    new_nodes = []
    for n in result["nodes"]:
        key = (n.get("text", "").lower().strip(), n.get("type", ""))
        if key in existing_keys or not n.get("text") or not n.get("type"):
            continue
        n["id"] = f"N_{next_id:03d}"
        next_id += 1
        existing_keys.add(key)
        new_nodes.append(n)
    return new_nodes, usage or {}


def deduplicate_edges(edges: list) -> list:
    """Remove duplicate edges with the same source, target, and type."""
    seen = set()
    unique = []
    for e in edges:
        key = (e.get("source_id"), e.get("target_id"), e.get("type"))
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


def extract_naive(transcript: str, client: OpenRouterClient) -> tuple:
    """Single-pass naive extraction."""
    prompt = EXTRACTION_PROMPT.format(transcript=transcript)
    content, usage = client.generate(prompt)

    if content:
        kg = extract_json_from_response(content)
        if kg:
            kg = validate_knowledge_graph(kg)
        return kg, usage
    return None, usage

def extract_edges_full_context(nodes: list, transcript: str, client: OpenRouterClient, index: 'TranscriptIndex' = None) -> tuple:
    """Edge extraction via node-pair enumeration.

    Python enumerates all schema-valid (source, target) pairs; two parallel LLM agents
    classify each pair. Results are merged on (source_id, target_id, type).
    Returns (edges, added_nodes, usage).
    """
    idx = index if index is not None else TranscriptIndex(transcript)

    candidate_pairs = enumerate_candidate_pairs(nodes)
    if not candidate_pairs:
        return [], [], {}

    nodes_json = json.dumps(
        [{"id": n["id"], "text": n["text"], "type": n["type"],
          "evidence": n.get("evidence", ""), "turn_id": n.get("turn_id", "")}
         for n in nodes],
        indent=2
    )

    pairs_text = ""
    for i, p in enumerate(candidate_pairs, 1):
        src, tgt = p["source"], p["target"]
        pairs_text += (
            f'[{i}] {p["pair_id"]}: "{src["text"]}" ({src["type"]}) → "{tgt["text"]}" ({tgt["type"]})\n'
            f'    Allowed types: {", ".join(p["allowed_types"])}\n'
            f'    Source evidence: {str(src.get("evidence", ""))[:120]}\n'
            f'    Target evidence: {str(tgt.get("evidence", ""))[:120]}\n\n'
        )

    prompt = EDGE_PAIR_CLASSIFICATION_PROMPT.format(
        nodes=nodes_json,
        pairs=pairs_text,
        transcript=transcript,
    )

    def _parse(content):
        if not content:
            return {"edges": [], "proposed_nodes": []}
        r = extract_json_from_response(content)
        if isinstance(r, list):
            r = {"edges": r}
        if not r or "edges" not in r:
            return {"edges": [], "proposed_nodes": []}
        if "proposed_nodes" not in r:
            r["proposed_nodes"] = []
        return r

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(client.generate, prompt)
        f2 = pool.submit(client.generate, prompt)
        (content1, usage1), (content2, usage2) = f1.result(), f2.result()

    combined_usage = {
        "prompt_tokens": (usage1 or {}).get("prompt_tokens", 0) + (usage2 or {}).get("prompt_tokens", 0),
        "completion_tokens": (usage1 or {}).get("completion_tokens", 0) + (usage2 or {}).get("completion_tokens", 0),
    }

    res1, res2 = _parse(content1), _parse(content2)

    # Merge proposed_nodes first (dedup on text+type, validate, assign IDs)
    added_nodes = []
    existing_keys = {(n["text"].lower().strip(), n.get("type", "")) for n in nodes}
    try:
        next_id = max(int(n["id"].split("_")[1]) for n in nodes if "_" in n.get("id", "")) + 1
    except ValueError:
        next_id = len(nodes) + 1

    all_proposed = res1.get("proposed_nodes", []) + res2.get("proposed_nodes", [])
    for pn in all_proposed:
        text = pn.get("text", "").strip()
        node_type = pn.get("type", "").upper()
        if not text or not node_type:
            continue
        key = (text.lower(), node_type)
        if key in existing_keys:
            continue
        check = check_node_in_transcript(text, transcript, index=idx)
        if check["matched"]:
            new_node = {
                "id": f"N_{next_id:03d}",
                "text": text,
                "type": node_type,
                "evidence": pn.get("evidence", check.get("evidence") or ""),
                "turn_id": pn.get("turn_id", ""),
            }
            nodes.append(new_node)
            added_nodes.append(new_node)
            existing_keys.add(key)
            next_id += 1

    # Merge edges: union on (source_id, target_id, type), first-occurrence wins
    seen_edges = {}
    for edge in res1["edges"] + res2["edges"]:
        src = edge.get("source_id", "")
        tgt = edge.get("target_id", "")
        etype = edge.get("type", "")
        key = (src, tgt, etype)
        if key not in seen_edges:
            seen_edges[key] = edge

    edges = list(seen_edges.values())
    combined_usage["_agent1_edge_count"] = len(res1["edges"])
    combined_usage["_agent2_edge_count"] = len(res2["edges"])
    combined_usage["_candidate_pairs"] = len(candidate_pairs)
    return edges, added_nodes, combined_usage


def extract_with_node_edge_agents(transcript: str, client: OpenRouterClient) -> tuple:
    """Six-pass extraction pipeline:
    node extraction → assessment node check → node review → dedup
    → edge extraction → edge review → canonicalization
    """
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    debug = {}

    def add_usage(u):
        if u:
            total_usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += u.get("completion_tokens", 0)

    # Build transcript index once — reused by all tool-calling passes
    idx = TranscriptIndex(transcript)

    # Pass 1: two parallel node extraction agents — merged on (text, type) to maximise recall
    prompt_p1 = NODE_EXTRACTION_PROMPT.format(transcript=transcript)
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(client.generate, prompt_p1)
        f2 = pool.submit(client.generate, prompt_p1)
        (content1, usage1), (content2, usage2) = f1.result(), f2.result()
    add_usage(usage1)
    add_usage(usage2)

    def _parse_nodes(content):
        if not content:
            return []
        r = extract_json_from_response(content)
        if isinstance(r, list):
            r = {"nodes": r}
        return r["nodes"] if r and "nodes" in r else []

    raw1, raw2 = _parse_nodes(content1), _parse_nodes(content2)
    if not raw1 and not raw2:
        print("(pass1: both agents returned no content)", end=" ", flush=True)
        return None, total_usage, {}

    # Merge: first-occurrence wins on (text.lower(), type); renumber IDs to avoid conflicts
    seen_p1: set = set()
    merged: list = []
    for n in raw1 + raw2:
        key = (n.get("text", "").lower().strip(), n.get("type", ""))
        if key[0] and key[1] and key not in seen_p1:
            seen_p1.add(key)
            merged.append(n)
    nodes = [{**n, "id": f"N_{i:03d}"} for i, n in enumerate(merged, 1)]

    debug["pass1_agent1"] = [{"text": n.get("text"), "type": n.get("type")} for n in raw1]
    debug["pass1_agent2"] = [{"text": n.get("text"), "type": n.get("type")} for n in raw2]
    debug["pass1_nodes"] = [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in nodes]

    # Pass 2: assessment node check — inject assessment text, add missing nodes (additive)
    added_nodes, usage = check_assessment_for_nodes(nodes, transcript, client, index=idx)
    add_usage(usage)
    if added_nodes:
        nodes.extend(added_nodes)
    debug["pass2_nodes_added"] = [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in added_nodes]

    # Pass 3: node review — verify each node via get_turn; limit=60 (bounded by node count)
    reviewed_nodes, usage = review_nodes_with_tool(nodes, transcript, client, index=idx)
    add_usage(usage)
    reviewed_ids = {n["id"] for n in reviewed_nodes}
    dropped_nodes = [n for n in nodes if n["id"] not in reviewed_ids]

    debug["pass3_nodes_dropped"] = [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in dropped_nodes]

    # Dedup nodes with same (text, type) — keeps first occurrence
    seen_node_keys = set()
    deduped = []
    for n in reviewed_nodes:
        key = (n["text"].lower().strip(), n.get("type", ""))
        if key not in seen_node_keys:
            seen_node_keys.add(key)
            deduped.append(n)
    dedup_removed = [n for n in reviewed_nodes if n["id"] not in {d["id"] for d in deduped}]
    reviewed_nodes = deduped

    debug["pass3_nodes_kept"] = [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in reviewed_nodes]
    if dedup_removed:
        debug["pass3_dedup_removed"] = [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in dedup_removed]

    # Pass 4: canonicalize node text to standard clinical form — before edge extraction to reduce pair count
    pre_canon_map = {n["id"]: n["text"] for n in reviewed_nodes}
    reviewed_nodes, usage = canonicalize_nodes(reviewed_nodes, client)
    add_usage(usage)
    debug["pass4_nodes_canonical"] = [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in reviewed_nodes]
    debug["pass4_renames"] = [
        {"id": n["id"], "before": pre_canon_map[n["id"]], "after": n["text"], "type": n["type"]}
        for n in reviewed_nodes
        if pre_canon_map.get(n["id"]) != n["text"]
    ]

    # Post-canonical dedup — merges nodes with same (text, type) after normalization
    seen_node_keys = set()
    deduped_canon = []
    for n in reviewed_nodes:
        key = (n["text"].lower().strip(), n.get("type", ""))
        if key not in seen_node_keys:
            seen_node_keys.add(key)
            deduped_canon.append(n)
    if len(deduped_canon) < len(reviewed_nodes):
        canon_dedup_removed = [n for n in reviewed_nodes if n["id"] not in {d["id"] for d in deduped_canon}]
        debug["pass4_canon_dedup_removed"] = [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in canon_dedup_removed]
    reviewed_nodes = deduped_canon

    # node_map built here — authoritative post-canon; extended with proposed nodes after Pass 5
    node_map = {n["id"]: n for n in reviewed_nodes}

    # Pass 5: node-pair enumeration edge extraction — two parallel agents on canonical node set
    edges, pass5_proposed, usage = extract_edges_full_context(reviewed_nodes, transcript, client, index=idx)
    debug["pass5_agent1_edges"] = usage.pop("_agent1_edge_count", 0)
    debug["pass5_agent2_edges"] = usage.pop("_agent2_edge_count", 0)
    debug["pass5_candidate_pairs"] = usage.pop("_candidate_pairs", 0)
    add_usage(usage)
    if pass5_proposed:
        node_map.update({n["id"]: n for n in pass5_proposed})
        debug["pass5_proposed_nodes"] = [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in pass5_proposed]

    if not edges and len(reviewed_nodes) > 5:
        print("(warn: pass5 0 edges)", end=" ", flush=True)
    debug["pass5_edges"] = [
        {
            "source": node_map.get(e.get("source_id"), {}).get("text", e.get("source_id")),
            "target": node_map.get(e.get("target_id"), {}).get("text", e.get("target_id")),
            "type": e.get("type"),
            "evidence": e.get("evidence", "")[:80],
        }
        for e in edges
    ]

    # Pass 5a: Python schema filter — drop edges not in VALID_EDGE_PATTERNS
    schema_valid_edges, schema_dropped = schema_filter_edges(edges, reviewed_nodes)
    debug["pass5a_schema_dropped"] = [
        {
            "source": node_map.get(e.get("source_id"), {}).get("text", e.get("source_id")),
            "target": node_map.get(e.get("target_id"), {}).get("text", e.get("target_id")),
            "type": e.get("type"),
        }
        for e in schema_dropped
    ]

    # Pass 5b: clinical plausibility review (no tools — schema already validated)
    reviewed_edges, usage = review_edges(reviewed_nodes, schema_valid_edges, client)
    add_usage(usage)
    kept_edge_keys = {(e.get("source_id"), e.get("target_id"), e.get("type")) for e in reviewed_edges}
    debug["pass5b_edges_kept"] = [
        {
            "source": node_map.get(e.get("source_id"), {}).get("text", e.get("source_id")),
            "target": node_map.get(e.get("target_id"), {}).get("text", e.get("target_id")),
            "type": e.get("type"),
        }
        for e in reviewed_edges
    ]
    debug["pass5b_edges_dropped"] = [
        {
            "source": node_map.get(e.get("source_id"), {}).get("text", e.get("source_id")),
            "target": node_map.get(e.get("target_id"), {}).get("text", e.get("target_id")),
            "type": e.get("type"),
            "evidence": e.get("evidence", "")[:80],
        }
        for e in schema_valid_edges
        if (e.get("source_id"), e.get("target_id"), e.get("type")) not in kept_edge_keys
    ]

    reviewed_edges = deduplicate_edges(reviewed_edges)

    kg = {"nodes": reviewed_nodes, "edges": reviewed_edges}
    kg = validate_knowledge_graph(kg)

    return kg, total_usage, debug


def process_one(
    txt_path: Path,
    client: OpenRouterClient,
    output_dir: Path,
    suffix: str,
    method: str = "reflect",
) -> tuple:
    """Process single transcript."""
    res_id = txt_path.stem
    output_file = output_dir / f"{res_id}_{suffix}.json"

    # Skip if already exists
    if output_file.exists():
        return res_id, "SKIP", 0, 0, None

    try:
        transcript = read_transcript(txt_path)
        print(f"  {res_id}...", end=" ", flush=True)
        debug = {}
        if method == "node_edge":
            kg, usage, debug = extract_with_node_edge_agents(transcript, client)
        else:
            kg, usage = extract_naive(transcript, client)

        if not kg:
            print("FAILED")
            return res_id, "FAILED", 0, 0, None

        n, e = len(kg.get('nodes', [])), len(kg.get('edges', []))

        kg['_usage'] = usage

        with open(output_file, 'w') as f:
            json.dump(kg, f, indent=2, ensure_ascii=False)

        if debug:
            debug_file = output_dir / f"{res_id}_{suffix}_debug.json"
            with open(debug_file, 'w') as f:
                json.dump(debug, f, indent=2, ensure_ascii=False)

        # Print pass-by-pass summary
        p1 = len(debug.get("pass1_nodes", []))
        p2a = len(debug.get("pass2_nodes_added", []))
        p3k = len(debug.get("pass3_nodes_kept", []))
        p3d = len(debug.get("pass3_nodes_dropped", []))
        p4k = len(debug.get("pass4_nodes_canonical", []))
        p5 = len(debug.get("pass5_edges", []))
        p5a_d = len(debug.get("pass5a_schema_dropped", []))
        p5k = len(debug.get("pass5b_edges_kept", []))
        p5d = len(debug.get("pass5b_edges_dropped", []))
        assess_n = f" +{p2a}@assess" if p2a else ""
        schema_drop = f" schema-{p5a_d}" if p5a_d else ""
        print(f"({n}n/{e}e) | nodes: {p1}{assess_n}→{p3k} (-{p3d})→canon{p4k} | edges: {p5}{schema_drop}→{p5k} (-{p5d})")
        return res_id, "OK", n, e, usage

    except Exception as ex:
        print(f"ERROR: {ex}")
        return res_id, f"ERROR: {ex}", 0, 0, None


def main():
    parser = argparse.ArgumentParser(description="KG Extraction Pipeline (naive, GLM via OpenRouter)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for sub-KG JSON files")
    parser.add_argument("--res-ids", nargs="+", default=None,
                        help="Optional list of patient IDs to process (e.g. RES0198 RES0199). Processes all if omitted.")
    parser.add_argument("--method", type=str, default="node_edge",
                        choices=["naive", "reflect", "node_edge"],
                        help="Extraction method: naive, node_edge (6-pass, default)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open("api_keys.json") as f:
        api_keys = json.load(f)

    client = get_client(api_keys)
    suffix = {"naive": "naive_glm", "reflect": "reflect_glm", "node_edge": "node_edge_glm"}[args.method]

    transcript_files = get_transcript_files()
    if args.res_ids:
        transcript_files = [f for f in transcript_files if f.parent.name in args.res_ids]
    print("KG Extraction Pipeline")
    print(f"Method: {args.method} ({OPENROUTER_MODEL})")
    print(f"Output: {output_dir}")
    print(f"Processing {len(transcript_files)} transcripts")
    print("=" * 60)

    success = 0
    failed = 0
    total_tokens = {"prompt": 0, "completion": 0}
    all_stats = []

    for txt_path in transcript_files:
        res_id, status, nodes, edges, usage = process_one(
            txt_path, client, output_dir, suffix, method=args.method
        )
        if status == "OK":
            success += 1
            if usage:
                total_tokens["prompt"] += usage.get("prompt_tokens", 0)
                total_tokens["completion"] += usage.get("completion_tokens", 0)
                all_stats.append({"res_id": res_id, "nodes": nodes, "edges": edges, **usage})
        elif status == "SKIP":
            print(f"  {res_id}: SKIP (exists)")
            success += 1
        else:
            failed += 1
        time.sleep(0.3)

    stats_file = output_dir / "_stats.json"
    with open(stats_file, 'w') as f:
        json.dump({
            "method": args.method,
            "model": OPENROUTER_MODEL,
            "total_tokens": total_tokens,
            "success": success,
            "failed": failed,
            "details": all_stats
        }, f, indent=2)

    print("=" * 60)
    print(f"Done! Success: {success}, Failed: {failed}")
    print(f"Total tokens: {total_tokens['prompt'] + total_tokens['completion']}")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
    
