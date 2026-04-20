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


# === Transcript search utilities ===
def check_node_in_transcript(node_text: str, transcript: str) -> dict:
    """Check if node text appears in transcript — exact, partial-word, or stem match."""
    check_text = re.sub(r'^absent\s+', '', node_text.lower().strip())
    trans_lower = transcript.lower()

    if check_text in trans_lower:
        idx = trans_lower.index(check_text)
        start, end = max(0, idx - 40), min(len(transcript), idx + len(check_text) + 40)
        return {"matched": True, "match_type": "exact", "evidence": transcript[start:end].strip()}

    words = re.findall(r'\b[a-z]{3,}\b', check_text)
    if words:
        if all(re.search(r'\b' + re.escape(w), trans_lower) for w in words):
            return {"matched": True, "match_type": "partial_words", "evidence": None}
        stems = [w[:max(4, len(w) - 2)] for w in words if len(w) > 4]
        if stems and all(s in trans_lower for s in stems):
            return {"matched": True, "match_type": "stem", "evidence": None}

    return {"matched": False, "match_type": "none", "evidence": None}


def get_turn(turn_id: str, transcript: str) -> dict:
    """Return full text of a transcript turn by ID."""
    pattern = rf'\[{re.escape(turn_id)}\]\s*[DP]:\s*(.+?)(?=\n\n\[|\Z)'
    match = re.search(pattern, transcript, re.DOTALL)
    if match:
        return {"turn_id": turn_id, "text": match.group(1).strip()}
    return {"turn_id": turn_id, "text": None, "error": "turn not found"}


def search_transcript(keyword: str, transcript: str) -> dict:
    """Return all turns containing a keyword."""
    keyword_lower = keyword.lower()
    results = []
    for block in transcript.split('\n\n'):
        if keyword_lower in block.lower():
            m = re.match(r'\[([DP]-\d+)\]', block.strip())
            results.append({"turn_id": m.group(1) if m else None, "text": block.strip()})
    return {"keyword": keyword, "matches": results}


# === Configuration ===
TRANSCRIPT_DIR = transcripts_dir()
MAX_RETRIES = 3
OPENROUTER_MODEL = "z-ai/glm-4.7-flash"
OUTPUT_SUFFIX = "naive_glm"

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
- The doctor's final assessment turn is information-dense: extract each diagnosis, treatment, and procedure as a separate node

TRANSCRIPT:
{transcript}

## FORMAT:
- Node IDs: "N_001", "N_002", etc.
- turn_id: "P-X" (patient) or "D-X" (doctor)
- evidence: exact quote from transcript

Output JSON with ONLY a "nodes" array: [{{"id", "text", "type", "evidence", "turn_id"}}]
Output ONLY valid JSON."""

REVIEW_NODE_PROMPT = """You are a senior clinician verifying extracted clinical KG nodes against a transcript.

For EACH node below, call check_node_in_transcript to verify it has textual support.
Then return a cleaned nodes list:
- exact / stem / partial match → KEEP (normalize text to clinical canonical form if needed, e.g. "covid" → "COVID-19")
- no match but valid clinical inference (e.g. "non-smoker" inferred from patient denying smoking) → KEEP
- no match and appears hallucinated → REMOVE

NODES:
{nodes}

Return ONLY valid JSON with ONLY "id", "text", "type" per node:
{{"nodes": [{{"id": "N_001", "text": "...", "type": "SYMPTOM"}}]}}"""


EDGE_EXTRACTION_PROMPT = """You are an experienced clinical physician finding relationships between clinical entities.

Use get_turn(turn_id) and search_transcript(keyword) to retrieve evidence from the transcript as needed.

## EDGE TYPES — choose precisely:
- INDICATES: symptom/finding → suspected diagnosis (headache INDICATES COVID-19)
- CONFIRMS: completed test/lab result → confirmed diagnosis
- RULES_OUT: test ordered to exclude a diagnosis, OR absent finding argues against it (COVID swab RULES_OUT COVID-19)
- TAKEN_FOR: treatment/medication/supportive care → condition or symptom (Tylenol TAKEN_FOR headache)
- LOCATED_AT: symptom → body location
- CAUSES: risk factor → condition

## KEY RULES:
- A test ORDERED to exclude a diagnosis → RULES_OUT (not CONFIRMS)
- Each key symptom the doctor links to a diagnosis needs its own INDICATES edge
- Use ONLY node IDs listed below — never invent IDs

NODES:
{nodes}

Use the tools to look up evidence, then output ONLY valid JSON:
{{"edges": [{{"source_id": "N_001", "target_id": "N_002", "type": "INDICATES", "evidence": "...", "turn_id": "D-52"}}]}}"""


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
        """Multi-turn generation with tool calling. dispatch(name, args) -> result dict."""
        messages = [{"role": "user", "content": prompt}]
        total_prompt = total_completion = 0

        for _ in range(30):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
            except Exception as e:
                print(f"(tool-call error: {e})", end=" ", flush=True)
                return "", None

            if resp.usage:
                total_prompt += resp.usage.prompt_tokens or 0
                total_completion += resp.usage.completion_tokens or 0

            msg = resp.choices[0].message
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

        return "", {"prompt_tokens": total_prompt, "completion_tokens": total_completion}


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
    nodes_json = json.dumps(
        [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in nodes],
        indent=2
    )
    prompt = REVIEW_NODE_PROMPT.format(nodes=nodes_json)

    def dispatch(name, args):
        if name == "check_node_in_transcript":
            return check_node_in_transcript(args["node_text"], transcript)
        return {"error": f"unknown tool: {name}"}

    content, usage = client.generate_with_tools(prompt, [CHECK_NODE_TOOL], dispatch)

    if content:
        result = extract_json_from_response(content)
        if result and "nodes" in result:
            original_map = {n["id"]: n for n in nodes}
            merged = []
            for n in result["nodes"]:
                orig = original_map.get(n["id"])
                # preserve all original fields; only allow text normalization from LLM
                merged.append({**(orig or n), "text": n["text"]})
            return merged, usage or {}

    return nodes, usage or {}


def extract_edges_with_tools(nodes: list, transcript: str, client: OpenRouterClient) -> tuple:
    """Extract edges using tool-based transcript lookup (no full transcript in prompt)."""
    nodes_summary = json.dumps(
        [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in nodes],
        indent=2
    )
    prompt = EDGE_EXTRACTION_PROMPT.format(nodes=nodes_summary)

    def dispatch(name, args):
        if name == "get_turn":
            return get_turn(args["turn_id"], transcript)
        if name == "search_transcript":
            return search_transcript(args["keyword"], transcript)
        return {"error": f"unknown tool: {name}"}

    content, usage = client.generate_with_tools(
        prompt, [GET_TURN_TOOL, SEARCH_TRANSCRIPT_TOOL], dispatch
    )

    if content:
        result = extract_json_from_response(content)
        if isinstance(result, list):
            result = {"edges": result}
        if result and "edges" in result:
            return result["edges"], usage or {}

    return [], usage or {}


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
    """Three-pass extraction: node agent → review agent (tool) → edge agent (tool)."""
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def add_usage(u):
        if u:
            total_usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += u.get("completion_tokens", 0)

    # Pass 1: node agent
    content, usage = client.generate(NODE_EXTRACTION_PROMPT.format(transcript=transcript))
    add_usage(usage)
    if not content:
        print("(pass1: no content)", end=" ", flush=True)
        return None, total_usage
    node_result = extract_json_from_response(content)
    if isinstance(node_result, list):
        node_result = {"nodes": node_result}
    if not node_result or "nodes" not in node_result:
        print(f"(pass1: bad JSON: {content[:80]})", end=" ", flush=True)
        return None, total_usage
    nodes = node_result["nodes"]

    # Pass 2: review agent — tool-based node verification (no full transcript in prompt)
    reviewed_nodes, usage = review_nodes_with_tool(nodes, transcript, client)
    add_usage(usage)

    # Pass 3: edge agent — tool-based transcript lookup (no full transcript in prompt)
    edges, usage = extract_edges_with_tools(reviewed_nodes, transcript, client)
    add_usage(usage)

    kg = {"nodes": reviewed_nodes, "edges": edges}
    kg = validate_knowledge_graph(kg)

    return kg, total_usage


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
        if method == "node_edge":
            kg, usage = extract_with_node_edge_agents(transcript, client)
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

        print(f"({n}n/{e}e)")
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
                        help="Extraction method: naive, reflect (2-pass), node_edge (3-pass, default)")
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
    