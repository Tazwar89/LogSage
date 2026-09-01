from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

config = TemplateMinerConfig()
template_miner = TemplateMiner(config)

def deduplicate_logs(logs):
    """
    Deduplicates log messages using the Drain3 template miner.

    Args:
        logs (list): A list of log messages to be deduplicated.

    Returns:
        None
    """
    print("--- Deduplicated Output Patterns ---")

    for log_line in logs:
        # add_log_message handles matching and learning inline
        result = template_miner.add_log_message(log_line)

        # Check if this log caused a new template or combined into an old one
        if result["change_type"] in ["cluster_created", "cluster_template_changed"]:
            print(f"Template ID {result['cluster_id']}: {result['template_string']}")