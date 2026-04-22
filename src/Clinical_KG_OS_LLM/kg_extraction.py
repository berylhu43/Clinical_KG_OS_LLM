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
from pathlib import Path

from Clinical_KG_OS_LLM.paths import transcripts_dir

# === Tool definitions ===
CHECK_NODE_TOOL = {
    "type": "function",
    "function": {
        "name": "check_node_in_transcript",
        "description": "Check if a clinical entity text appears in the transcript. Returns match type (exact, stem, partial, none) and an evidence snippet.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "node_text": {"type": "string", "description": "The node text to search for"}
            },
            "required": ["node_id", "node_text"]
        }
    }
}

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

CHECK_EDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "check_edge_evidence",
        "description": "Find transcript turns where both source and target entities appear (same turn or adjacent turns). Returns co-occurrence evidence supporting the relationship.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_text": {"type": "string"},
                "target_text": {"type": "string"},
                "edge_type": {"type": "string", "description": "The proposed edge type"}
            },
            "required": ["source_text", "target_text", "edge_type"]
        }
    }
}

VALIDATE_EDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "validate_edge_type",
        "description": "Returns allowed edge types for a source-target node type pair, based on patterns in clinical KGs.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_type": {"type": "string", "description": "Node type of source (SYMPTOM, DIAGNOSIS, TREATMENT, PROCEDURE, LOCATION, MEDICAL_HISTORY, LAB_RESULT)"},
                "target_type": {"type": "string", "description": "Node type of target"}
            },
            "required": ["source_type", "target_type"]
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
    """Return the longest doctor turn — typically the assessment/plan."""
    if index is not None:
        best_id, best_text, best_len = None, "", 0
        for turn_id in index.turn_order:
            if not turn_id.startswith('D-'):
                continue
            text = index.turn_text[turn_id]
            if len(text) > best_len:
                best_id, best_text, best_len = turn_id, text, len(text)
        return {"turn_id": best_id, "text": best_text}
    best = {"turn_id": None, "text": "", "length": 0}
    for m in re.finditer(r'\[D-(\d+)\]\s*D:\s*(.+?)(?=\n\n\[|\Z)', transcript, re.DOTALL):
        text = m.group(2).strip()
        if len(text) > best["length"]:
            best = {"turn_id": f"D-{m.group(1)}", "text": text, "length": len(text)}
    return {"turn_id": best["turn_id"], "text": best["text"]}


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
- Use lowercase, short canonical text matching standard clinical terminology
- Do NOT extract vague phrases like "feeling unwell" — use the specific symptom name
- Do NOT extract denied/absent symptoms — these belong as edge relations, not nodes
- Preserve clinical qualifiers in symptom text (e.g. "dry cough" not "cough")
- For DIAGNOSIS nodes: use the full standard name with qualifiers (e.g. "covid-19" not "covid", "viral illness" not "virus"). Extract ALL diagnoses in the assessment including differentials ("could be X", "if not X")
- For PROCEDURE nodes: include what is being tested (e.g. "covid swab" not "swab", "nasal swab" not "swab")
- For TREATMENT nodes: extract the clinical noun concept, not the activity phrasing (e.g. "hydration" not "well hydrated", "self-isolation" not "isolate for 14 days", "nutrition" not "eating nutritious food", "rest" not "sleeping well", "analgesics" not "taking Tylenol for pain")
- For MEDICAL_HISTORY: extract lifestyle facts inferred from negative answers (patient says "no" to smoking → extract "non-smoker"; says "I'm pretty healthy, no conditions" → extract "no chronic conditions"). Extract substance use facts (marijuana use, alcohol use) when confirmed. Do NOT extract immunization status unless a deficiency was noted.
- For LOCATION nodes: single lowercase anatomical term matching transcript wording (e.g. "chest", "left arm", "throat"). One location per node — do not combine multiple body parts.
- For LAB_RESULT nodes: always include the measured value with units (e.g. "A1C 7.2%", "BP 148/90", "temperature 101°F"). Do not extract a lab name without its value.
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

For EACH node, call check_node_in_transcript to verify textual support. Use these rules:
- Match found in patient turn → KEEP
- Match found in doctor question turn → check the patient reply in the evidence:
  - For SYMPTOM nodes: if patient denied it (e.g. "No", "not really") → REMOVE
  - For MEDICAL_HISTORY nodes representing a negative state (text starts with "non-", "no ", "never "): patient saying "No" CONFIRMS the node → KEEP
  - Otherwise if confirmed → KEEP
- No match → call search_transcript with a related keyword to look for supporting context (e.g. for "non-smoker" search "smoke"). If context supports the node as a valid clinical inference → KEEP. If nothing supports it → REMOVE.
- For SYMPTOM/PROCEDURE: normalize text using the wording from the transcript — keep brand names and colloquial terms as said (e.g. "Tylenol" not "acetaminophen")
- For TREATMENT nodes: use the clinical noun form — NOT the activity phrasing (e.g. "hydration" not "well hydrated", "self-isolation" not "isolate", "nutrition" not "eating nutritious food", "rest" not "sleeping well")
- For DIAGNOSIS: use the full standardized disease name as it would appear in a medical record. Expand informal shorthand to the proper clinical name (e.g. "covid-19" not "covid", "influenza" not "flu"). Do not use informal abbreviations even if that is what the transcript says.
- Keep diagnosis nodes introduced conditionally ("could be", "if not X") — these are valid differentials
- MEDICAL_HISTORY: only keep lifestyle facts and past conditions that are clinically relevant. Immunization status is not MEDICAL_HISTORY unless the patient is behind on vaccinations — remove it if the patient is up to date.

NODES:
{nodes}

Return ONLY valid JSON with ONLY "id", "text", "type" per node:
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

Assign new IDs continuing from the highest existing ID (e.g. if last is N_017, start at N_018).

Return ONLY valid JSON — empty list if nothing is missing:
{{"nodes": [{{"id": "N_018", "text": "...", "type": "DIAGNOSIS", "evidence": "...", "turn_id": "{assessment_turn_id}"}}]}}"""


ASSESSMENT_EDGE_PROMPT = """You are a senior clinician checking whether the doctor's assessment contains relationships not yet captured in the extracted edge list.

DOCTOR'S ASSESSMENT ({assessment_turn_id}):
{assessment_text}

NODES:
{nodes}

EDGES ALREADY EXTRACTED:
{edges}

Add ONLY edges that:
- Are clearly supported by the assessment text above
- Are NOT already present (check source_id + target_id + type)
- Use ONLY the node IDs listed above

Return ONLY valid JSON — empty list if nothing is missing:
{{"edges": [{{"source_id": "N_001", "target_id": "N_002", "type": "TAKEN_FOR", "evidence": "...", "turn_id": "{assessment_turn_id}"}}]}}"""


EDGE_EXTRACTION_PROMPT = """You are an experienced clinical physician finding relationships between clinical entities.

Use get_turn(turn_id) and search_transcript(keyword) to retrieve evidence from the transcript as needed.

## EDGE TYPES — choose precisely:
- INDICATES: symptom/finding → suspected diagnosis (headache INDICATES covid-19)
- CONFIRMS: completed test/lab result → confirmed diagnosis
- RULES_OUT: test ordered to exclude a diagnosis, OR absent finding argues against it (covid swab RULES_OUT covid-19)
- TAKEN_FOR: treatment/medication/supportive care → condition or symptom (Tylenol TAKEN_FOR headache)
- LOCATED_AT: symptom → body location
- CAUSES: risk factor → condition

## KEY RULES:
- A test ORDERED to exclude a diagnosis → RULES_OUT (not CONFIRMS)
- INDICATES: only create when the doctor explicitly links a symptom to a specific diagnosis. For alternative/differential diagnoses introduced with "could be" or "if not X", do NOT duplicate INDICATES edges — they share implied symptoms with the primary diagnosis
- TAKEN_FOR: check BOTH early patient turns (patient-reported medications they are already taking) AND the assessment turn (doctor-recommended treatments). A patient saying "I take Tylenol for my headache" → Tylenol TAKEN_FOR headache. Doctor-recommended supportive care in the assessment → TAKEN_FOR the primary diagnosis.
- LOCATED_AT: symptom → body location. Mandatory for every LOCATION node — see systematic check #4 below.
- If an edge requires a node not in the list below: call propose_node(text, type, reason) — Python will verify it exists in the transcript and return its new ID. Only use the returned ID if status is "added"

## SYSTEMATIC NODE CHECKS (do these before finishing):
1. PROCEDURE nodes: for each, call search_transcript(procedure_text) — find what condition it was ordered to test/exclude → RULES_OUT (ordered to exclude) or CONFIRMS (result confirmed a diagnosis)
2. TREATMENT nodes: for each, verify you have a TAKEN_FOR edge. If missing, call search_transcript(treatment_text) to find what condition/symptom it was given for → TAKEN_FOR
3. MEDICAL_HISTORY nodes: for each, check if it CAUSES any DIAGNOSIS node — call search_transcript(history_text) if needed
4. LOCATION nodes: for each, call search_transcript(location_text) — find which SYMPTOM node was being discussed in that context → LOCATED_AT (mandatory, do NOT skip)
5. SYMPTOM nodes: for each, call search_transcript(symptom_text) — find the doctor's assessment turn where a specific diagnosis is named alongside it → INDICATES. Only add if the doctor explicitly links this symptom to a named diagnosis; check adjacent_turns in the result for context.
6. LAB_RESULT nodes: for each, call search_transcript(lab_text) — find the diagnosis the doctor links the result to → CONFIRMS

NODES:
{nodes}

Use the tools to look up evidence, then output ONLY valid JSON:
{{"edges": [{{"source_id": "N_001", "target_id": "N_002", "type": "INDICATES", "evidence": "...", "turn_id": "D-52"}}]}}"""


EDGE_REVIEW_PROMPT = """You are a senior clinician validating extracted clinical KG edges for correctness.

For EACH edge:
1. Call validate_edge_type(source_type, target_type) — if the edge type is NOT in the allowed list → REMOVE (hard rule, no exceptions)
2. Call check_edge_evidence(source_text, target_text, edge_type) — use the result to improve the evidence field:
   - If co-occurrence found → update evidence with the transcript text from the tool result
   - If no co-occurrence found but the edge already has a non-empty evidence field → KEEP with the original evidence
   - If no co-occurrence found AND the edge has no evidence → REMOVE

NODES (for reference):
{nodes}

EDGES (each includes source/target text and type for tool calls):
{edges}

Return ONLY valid JSON:
{{"edges": [{{"source_id": "N_001", "target_id": "N_002", "type": "INDICATES", "evidence": "...", "turn_id": "..."}}]}}"""


REVIEW_PROMPT = """You are a senior clinical physician doing a final review of an extracted knowledge graph. Compare it against the original transcript and correct any issues.

TRANSCRIPT:
{transcript}

EXTRACTED KG:
{kg}

Review checklist:
1. Missing clinically significant entities — add them
2. Missing relationships between existing nodes — add them
3. Incorrect node text — normalize to short canonical clinical terms (lowercase, standard names)
4. Incorrect node or edge types — remap to the allowed types below
5. Denied/absent symptoms must use "absent [symptom]" format (e.g. "absent fever", "absent chest pain")

## ALLOWED NODE TYPES (no exceptions):
- SYMPTOM: Symptoms present or absent. Absent ones: "absent fever", "absent cough"
- DIAGNOSIS: Conditions active, suspected, or ruled out: "viral infection / common cold", "asthma ruled out"
- TREATMENT: Medications and interventions. Generic names: "tylenol", "decongestants", "hydration"
- PROCEDURE: Tests and exams: "covid swab", "chest x-ray", "pulse oximetry"
- LOCATION: Body parts only, single concise word: "nose", "chest", "throat"
- MEDICAL_HISTORY: Past conditions, exposures, family history, lifestyle, allergies
- LAB_RESULT: Lab values with measurements: "temperature ~101 f"

## ALLOWED EDGE TYPES (no exceptions):
- CAUSES: Risk factor/exposure causes condition
- INDICATES: Symptom/finding indicates diagnosis
- LOCATED_AT: Symptom located at body part
- RULES_OUT: Test/finding/absent symptom rules out diagnosis
- TAKEN_FOR: Treatment prescribed for condition
- CONFIRMS: Lab/test confirms diagnosis

Return the complete improved JSON with the same format (nodes + edges). Output ONLY valid JSON."""



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

    def generate_with_tools(self, prompt: str, tools: list, dispatch: callable) -> tuple:
        """Multi-turn generation with tool calling. dispatch(name, args) -> result dict.

        Graceful degradation: at WARN_AT iterations inject a stop signal so the model
        wraps up cleanly. Falls back to last partial assistant output if limit is hit.
        """
        LIMIT = 60
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
                    tool_choice="auto"
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


def review_nodes_with_tool(nodes: list, transcript: str, client: OpenRouterClient) -> tuple:
    """Review extracted nodes using tool-based transcript verification."""
    idx = TranscriptIndex(transcript)
    nodes_json = json.dumps(
        [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in nodes],
        indent=2
    )
    prompt = REVIEW_NODE_PROMPT.format(nodes=nodes_json)

    def dispatch(name, args):
        if name == "check_node_in_transcript":
            return check_node_in_transcript(args["node_text"], transcript, index=idx)
        if name == "search_transcript":
            return search_transcript(args["keyword"], transcript, index=idx)
        return {"error": f"unknown tool: {name}"}

    content, usage = client.generate_with_tools(prompt, [CHECK_NODE_TOOL, SEARCH_TRANSCRIPT_TOOL], dispatch)

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


def extract_edges_with_tools(nodes: list, transcript: str, client: OpenRouterClient) -> tuple:
    """Extract edges using tool-based transcript lookup. Supports propose_node for mid-pass node discovery.

    Returns (edges, proposed_ids, usage) where proposed_ids tracks nodes added via propose_node
    so the caller can run them through review_nodes_with_tool for type/canonicalization correction.
    """
    idx = TranscriptIndex(transcript)

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
    prompt = EDGE_EXTRACTION_PROMPT.format(nodes=nodes_summary)

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


def review_edges_with_tool(nodes: list, edges: list, transcript: str, client: OpenRouterClient) -> tuple:
    """Review extracted edges using validate_edge_type and check_edge_evidence tools."""
    idx = TranscriptIndex(transcript)
    node_map = {n["id"]: n for n in nodes}

    # Enrich edges with node text/type so the agent can call tools without extra lookups
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

    nodes_summary = json.dumps(
        [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in nodes], indent=2
    )
    prompt = EDGE_REVIEW_PROMPT.format(nodes=nodes_summary, edges=json.dumps(enriched, indent=2))

    def dispatch(name, args):
        if name == "check_edge_evidence":
            return check_edge_evidence(args["source_text"], args["target_text"], args["edge_type"], transcript, index=idx)
        if name == "validate_edge_type":
            return validate_edge_type(args["source_type"], args["target_type"])
        return {"error": f"unknown tool: {name}"}

    content, usage = client.generate_with_tools(prompt, [CHECK_EDGE_TOOL, VALIDATE_EDGE_TOOL], dispatch)

    if content:
        result = extract_json_from_response(content)
        if isinstance(result, list):
            result = {"edges": result}
        if result and "edges" in result:
            # Strip enrichment fields — keep only original edge fields
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


def check_assessment_for_nodes(nodes: list, transcript: str, client: OpenRouterClient) -> tuple:
    """Additive step: inject assessment text and ask LLM to add any missing nodes."""
    assessment = get_longest_doctor_turn(transcript)
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


def check_assessment_for_edges(nodes: list, edges: list, transcript: str, client: OpenRouterClient) -> tuple:
    """Additive step: inject assessment text and ask LLM to add any missing edges."""
    assessment = get_longest_doctor_turn(transcript)
    if not assessment.get("text"):
        return [], {}

    node_ids = {n["id"] for n in nodes}
    nodes_summary = json.dumps(
        [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in nodes], indent=2
    )
    edges_summary = json.dumps(
        [{"source_id": e["source_id"], "target_id": e["target_id"], "type": e["type"]} for e in edges], indent=2
    )
    prompt = ASSESSMENT_EDGE_PROMPT.format(
        assessment_turn_id=assessment["turn_id"],
        assessment_text=assessment["text"],
        nodes=nodes_summary,
        edges=edges_summary,
    )
    content, usage = client.generate(prompt)
    if not content:
        return [], usage or {}

    result = extract_json_from_response(content)
    if isinstance(result, list):
        result = {"edges": result}
    if not result or "edges" not in result:
        return [], usage or {}

    existing_keys = {(e["source_id"], e["target_id"], e["type"]) for e in edges}
    new_edges = []
    for e in result["edges"]:
        src, tgt, etype = e.get("source_id"), e.get("target_id"), e.get("type")
        if not src or not tgt or not etype:
            continue
        if src not in node_ids or tgt not in node_ids:
            continue  # reject hallucinated IDs
        key = (src, tgt, etype)
        if key in existing_keys:
            continue
        src_type = next((n["type"] for n in nodes if n["id"] == src), "")
        tgt_type = next((n["type"] for n in nodes if n["id"] == tgt), "")
        if etype not in validate_edge_type(src_type, tgt_type)["allowed_edge_types"]:
            continue
        new_edges.append({
            "source_id": src, "target_id": tgt, "type": etype,
            "evidence": e.get("evidence", ""),
            "turn_id": e.get("turn_id", assessment["turn_id"]),
        })
        existing_keys.add(key)
    return new_edges, usage or {}


def restore_dropped_medical_history(dropped_nodes: list, transcript: str) -> list:
    """Re-add MEDICAL_HISTORY nodes dropped by Pass 2 when transcript evidence exists.

    Handles:
    - Negative-state nodes ("non-smoker", "no diabetes"): search for base keyword
    - Substance use nodes ("marijuana use", "alcohol use"): search for substance keyword
    - Any other MEDICAL_HISTORY: try exact match then first-significant-word search
    """
    restored = []
    for node in dropped_nodes:
        if node.get("type") != "MEDICAL_HISTORY":
            continue
        text = node.get("text", "").lower()

        # Try exact match first
        result = check_node_in_transcript(text, transcript)
        if result["matched"]:
            restored.append(node)
            continue

        # For negative-state nodes, search for the base keyword
        keyword = None
        for prefix in ("non-", "no ", "never ", "absent "):
            if text.startswith(prefix):
                keyword = text[len(prefix):].strip()
                break

        # Fallback: first significant word (handles "marijuana use", "alcohol use", etc.)
        if keyword is None:
            words = [w for w in re.findall(r'\b[a-z]{4,}\b', text)
                     if w not in ("with", "from", "that", "this", "have", "been", "history", "past")]
            keyword = words[0] if words else None

        if keyword:
            search_result = search_transcript(keyword, transcript)
            if search_result["matches"]:
                restored.append(node)

    return restored


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

def extract_with_node_edge_agents(transcript: str, client: OpenRouterClient) -> tuple:
    """Six-pass extraction pipeline:
    node extraction → assessment node check → node review → dedup
    → edge extraction → assessment edge check → edge review
    """
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    debug = {}

    def add_usage(u):
        if u:
            total_usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += u.get("completion_tokens", 0)

    # Pass 1: node extraction
    content, usage = client.generate(NODE_EXTRACTION_PROMPT.format(transcript=transcript))
    add_usage(usage)
    if not content:
        print("(pass1: no content)", end=" ", flush=True)
        return None, total_usage, {}
    node_result = extract_json_from_response(content)
    if isinstance(node_result, list):
        node_result = {"nodes": node_result}
    if not node_result or "nodes" not in node_result:
        print(f"(pass1: bad JSON: {content[:80]})", end=" ", flush=True)
        return None, total_usage, {}
    nodes = node_result["nodes"]
    debug["pass1_nodes"] = [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in nodes]

    # Pass 2: assessment node check — inject assessment text, add missing nodes (additive)
    added_nodes, usage = check_assessment_for_nodes(nodes, transcript, client)
    add_usage(usage)
    if added_nodes:
        nodes.extend(added_nodes)
    debug["pass2_nodes_added"] = [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in added_nodes]

    # Pass 3: node review — verify/normalize only (no full transcript in prompt)
    reviewed_nodes, usage = review_nodes_with_tool(nodes, transcript, client)
    add_usage(usage)
    reviewed_ids = {n["id"] for n in reviewed_nodes}

    debug["pass3_nodes_kept"] = [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in reviewed_nodes]
    debug["pass3_nodes_dropped"] = [
        {"id": n["id"], "text": n["text"], "type": n["type"]}
        for n in nodes if n["id"] not in reviewed_ids
    ]

    # Dedup nodes with same (text, type) — keeps first occurrence
    seen_node_keys = set()
    deduped = []
    for n in reviewed_nodes:
        key = (n["text"].lower().strip(), n.get("type", ""))
        if key not in seen_node_keys:
            seen_node_keys.add(key)
            deduped.append(n)
    reviewed_nodes = deduped

    # Pass 4: edge extraction — tool-based transcript lookup
    edges, proposed_ids, usage = extract_edges_with_tools(reviewed_nodes, transcript, client)
    add_usage(usage)

    # Pass 4b: review any nodes proposed mid-pass through the same node review gate
    # (type correction + canonicalization — same logic as Pass 3)
    if proposed_ids:
        proposed = [n for n in reviewed_nodes if n["id"] in proposed_ids]
        reviewed_proposed, usage = review_nodes_with_tool(proposed, transcript, client)
        add_usage(usage)
        # Replace proposed nodes with their reviewed versions
        reviewed_proposed_map = {n["id"]: n for n in reviewed_proposed}
        reviewed_nodes = [
            reviewed_proposed_map.get(n["id"], n) if n["id"] in proposed_ids else n
            for n in reviewed_nodes
        ]

    # Rebuild node_map after propose_node mutations and Pass 4b corrections
    node_map = {n["id"]: n for n in reviewed_nodes}
    if not edges and len(reviewed_nodes) > 5:
        print("(warn: pass4 0 edges — possible tool-call limit hit)", end=" ", flush=True)
    debug["pass4_edges"] = [
        {
            "source": node_map.get(e.get("source_id"), {}).get("text", e.get("source_id")),
            "target": node_map.get(e.get("target_id"), {}).get("text", e.get("target_id")),
            "type": e.get("type"),
            "evidence": e.get("evidence", "")[:80],
        }
        for e in edges
    ]

    # Pass 5: assessment edge check — inject assessment text, add missing edges (additive)
    added_edges, usage = check_assessment_for_edges(reviewed_nodes, edges, transcript, client)
    add_usage(usage)
    if added_edges:
        edges.extend(added_edges)
        edges = deduplicate_edges(edges)
    debug["pass5_edges_added"] = [
        {
            "source": node_map.get(e.get("source_id"), {}).get("text", e.get("source_id")),
            "target": node_map.get(e.get("target_id"), {}).get("text", e.get("target_id")),
            "type": e.get("type"),
        }
        for e in added_edges
    ]

    # Pass 6: edge review — validate type + verify transcript evidence per edge
    reviewed_edges, usage = review_edges_with_tool(reviewed_nodes, edges, transcript, client)
    add_usage(usage)
    kept_edge_keys = {(e.get("source_id"), e.get("target_id"), e.get("type")) for e in reviewed_edges}
    debug["pass6_edges_kept"] = [
        {
            "source": node_map.get(e.get("source_id"), {}).get("text", e.get("source_id")),
            "target": node_map.get(e.get("target_id"), {}).get("text", e.get("target_id")),
            "type": e.get("type"),
        }
        for e in reviewed_edges
    ]
    debug["pass6_edges_dropped"] = [
        {
            "source": node_map.get(e.get("source_id"), {}).get("text", e.get("source_id")),
            "target": node_map.get(e.get("target_id"), {}).get("text", e.get("target_id")),
            "type": e.get("type"),
            "evidence": e.get("evidence", "")[:80],
        }
        for e in edges
        if (e.get("source_id"), e.get("target_id"), e.get("type")) not in kept_edge_keys
    ]

    reviewed_edges = deduplicate_edges(reviewed_edges)

    kg = {"nodes": reviewed_nodes, "edges": reviewed_edges}
    kg = validate_knowledge_graph(kg)

    return kg, total_usage, debug


def extract_with_reflection(transcript: str, client: OpenRouterClient) -> tuple:
    """Two-pass extraction with self-critique."""
    # Pass 1: naive extraction
    kg, usage1 = extract_naive(transcript, client)
    if not kg:
        return None, usage1

    # Pass 2: reflection
    prompt = REVIEW_PROMPT.format(
        transcript=transcript,
        kg=json.dumps(kg, indent=2)
    )
    content, usage2 = client.generate(prompt)

    improved_kg = None
    if content:
        improved_kg = extract_json_from_response(content)
        if improved_kg:
            improved_kg = validate_knowledge_graph(improved_kg)

    # merge usage
    combined_usage = None
    if usage1 or usage2:
        combined_usage = {
            "prompt_tokens": (usage1 or {}).get("prompt_tokens", 0) + (usage2 or {}).get("prompt_tokens", 0),
            "completion_tokens": (usage1 or {}).get("completion_tokens", 0) + (usage2 or {}).get("completion_tokens", 0),
        }

    return improved_kg or kg, combined_usage


def process_one(txt_path: Path, client: OpenRouterClient, output_dir: Path, suffix: str, method: str = "reflect") -> tuple:
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
        elif method == "naive":
            kg, usage = extract_naive(transcript, client)
        else:
            kg, usage = extract_with_reflection(transcript, client)

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
        p4 = len(debug.get("pass4_edges", []))
        p5a = len(debug.get("pass5_edges_added", []))
        p6k = len(debug.get("pass6_edges_kept", []))
        p6d = len(debug.get("pass6_edges_dropped", []))
        assess_n = f" +{p2a}@assess" if p2a else ""
        assess_e = f" +{p5a}@assess" if p5a else ""
        print(f"({n}n/{e}e) | nodes: {p1}{assess_n}→{p3k} (-{p3d}) | edges: {p4}{assess_e}→{p6k} (-{p6d})")
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
                        help="Extraction method: naive, reflect (2-pass), node_edge (6-pass, default)")
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
    