from fastapi import FastAPI, UploadFile, HTTPException
from .parsing import parse_line, save_json
from .embedding import build_template_miner, deduplicate_logs, get_unique_templates
from .vector_store import VectorStore
from .anomaly import is_anomalous
from .rag import load_knowledge_base, build_kb_index, retrieve_context
from .llm_analysis import analyze_log

app = FastAPI(title="LogSage")

template_miner = build_template_miner()
normal_store = VectorStore()
kb_store = VectorStore()
kb_lookup = {}
parsed_logs_db = {}  # trace_id -> parsed log dict

@app.on_event("startup")
def startup():
    global kb_lookup
    kb_entries = load_knowledge_base()
    kb_lookup = build_kb_index(kb_store, kb_entries)


@app.post("/upload/baseline")
async def upload_baseline(file: UploadFile):
    content = (await file.read()).decode("utf-8", errors="ignore")
    lines = content.splitlines()
    parsed_logs = [p for p in (parse_line(l) for l in lines if l.strip()) if p]
    annotated_logs = deduplicate_logs(parsed_logs, template_miner)
    templates = get_unique_templates(template_miner)
    normal_store.build_index(templates)

    return {"baseline_templates": len(templates)}


@app.post("/upload/logs")
async def upload_logs(file: UploadFile):
    content = (await file.read()).decode("utf-8", errors="ignore")
    lines = content.splitlines()
    parsed_logs = [p for p in (parse_line(l) for l in lines if l.strip()) if p]

    for i, entry in enumerate(parsed_logs):
        trace_id = f"{file.filename}-{i}"
        parsed_logs_db[trace_id] = entry

    return {"ingested": len(parsed_logs)}


@app.get("/analyze/{trace_id}")
def analyze(trace_id: str):
    entry = parsed_logs_db.get(trace_id)

    if not entry:
        raise HTTPException(status_code=404, detail="trace_id not found")

    anomalous, nearest = is_anomalous(entry["message"], normal_store, threshold=0.6)

    if not anomalous:
        return {"trace_id": trace_id, "anomalous": False, "nearest_match": nearest}

    context = retrieve_context(entry["message"], kb_store, kb_lookup)
    result = analyze_log(entry["message"], context)

    return {"trace_id": trace_id, "anomalous": True, "analysis": result, "retrieved_context": context}


@app.get("/logs")
def list_logs():
    return list(parsed_logs_db.keys())