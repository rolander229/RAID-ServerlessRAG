"""
Standard Cloud-Native FaaS Function Template (Synchronous API Worker / Orchestrator)
Provides standard request decoding, authorization extraction, error handling,
and structured JSON response wrapping.
"""
import json
import logging
import os
import traceback
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_event_payload(event: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Parse JSON body, query parameters, and path parameters from API Gateway event."""
    body = {}
    if event.get("body"):
        try:
            body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
        except Exception as e:
            return {}, {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": f"Invalid JSON payload: {str(e)}"})
            }
    
    # Merge query string parameters and path parameters
    query_params = event.get("queryStringParameters") or {}
    path_params = event.get("pathParameters") or {}
    
    payload = {
        "body": body,
        "query": query_params,
        "path": path_params,
        "headers": event.get("headers") or {}
    }
    return payload, None


def extract_user_identity(event: Dict[str, Any]) -> Optional[str]:
    """Extract authenticated user identity (Cognito Sub / JWT claim) from API Gateway context."""
    try:
        claims = event.get("requestContext", {}).get("authorizer", {}).get("claims", {})
        return claims.get("sub") or claims.get("username") or claims.get("cognito:username")
    except Exception:
        return None


def format_api_response(status_code: int, data: Any, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Format structured HTTP response conforming to API Gateway specification."""
    default_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type,Authorization"
    }
    if headers:
        default_headers.update(headers)
    
    return {
        "statusCode": status_code,
        "headers": default_headers,
        "body": json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    }


def faas_handler_wrapper(func: Callable):
    """Decorator to apply uniform error handling and logging to FaaS handlers."""
    def wrapper(event: Dict[str, Any], context: Any):
        logger.info("Function Invoked: %s", context.function_name if hasattr(context, 'function_name') else "local")
        try:
            return func(event, context)
        except Exception as e:
            logger.error("Unhandled Exception in FaaS Execution: %s\n%s", str(e), traceback.format_exc())
            return format_api_response(500, {
                "error": "Internal server error during algorithm function execution",
                "detail": str(e)
            })
    return wrapper


# -------------------------------------------------------------
# Template Entry Point Example
# -------------------------------------------------------------
@faas_handler_wrapper
def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    payload, err = parse_event_payload(event)
    if err:
        return err
    
    user_id = extract_user_identity(event)
    logger.info("Executing function for user: %s", user_id)
    
    # Execute Model Algorithm Logic here
    result = {"status": "success", "result": "Algorithm output placeholder"}
    return format_api_response(200, result)
