"""
Standard Cloud-Native FaaS Async Worker Template (SQS/EventBridge Consumer)
Used for heavy AI/DL algorithm compute, model training/fine-tuning trigger,
batch embedding, and long-running pipeline execution.
"""
import json
import logging
import traceback
from typing import Any, Callable, Dict, List

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def process_sqs_batch(event: Dict[str, Any], record_processor: Callable[[Dict[str, Any]], Any]) -> Dict[str, List[Dict[str, str]]]:
    """
    Process SQS records in batch and return batchItemFailures for partial failure reporting.
    This ensures only failed messages are retried by the SQS queue / dead-letter queue.
    """
    records = event.get("Records", [])
    logger.info("Received %d SQS message record(s)", len(records))
    
    batch_item_failures = []
    for record in records:
        message_id = record.get("messageId", "unknown")
        try:
            body = json.loads(record.get("body", "{}"))
            logger.info("Processing async task payload: %s", message_id)
            record_processor(body)
            logger.info("Task completed successfully: %s", message_id)
        except Exception as e:
            logger.error("Failed to process message %s: %s\n%s", message_id, str(e), traceback.format_exc())
            batch_item_failures.append({"itemIdentifier": message_id})
            
    return {"batchItemFailures": batch_item_failures}


# -------------------------------------------------------------
# Template Entry Point Example
# -------------------------------------------------------------
def execute_heavy_algorithm_task(task_payload: Dict[str, Any]) -> Any:
    """Execute AI/DL compute task (e.g. batch vector embedding, model inference)."""
    task_type = task_payload.get("task_type")
    data = task_payload.get("data")
    logger.info("Running heavy algorithm task: %s", task_type)
    # Perform algorithm execution
    return {"status": "finished"}


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    return process_sqs_batch(event, execute_heavy_algorithm_task)
