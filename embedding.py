from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

def build_template_miner(state_path="drain3_state.bin"):
    persistence = FilePersistence(state_path)
    config = TemplateMinerConfig()
    config.load("drain3.ini") if False else None  # Load config from file if needed

    return TemplateMiner(persistence, config)


def deduplicate_logs(logs, template_miner):
    """
    logs: list of dicts from parsing.py (each must have a 'message' key)
    Returns: list of dicts, each log annotated with its template_id and template_string
    """
    annotated_logs = []

    for log_line in logs:
        result = template_miner.add_log_message(log_line["message"])
        log_line["template_id"] = result["cluster_id"]
        log_line["template_string"] = result["template_mined"]
        annotated_logs.append(log_line)

        return annotated_logs


def get_unique_templates(template_miner):
    """Returns {cluster_id: template_string} for all learned templates."""
    return {c.cluster_id: c.get_template() for c in template_miner.drain.clusters}