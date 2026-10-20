# Serverless Architecture Reference for Monolith Migration

This document provides architecture patterns, component rules, and the canonical
blueprint schema for migrating monolithic applications to AWS Serverless.

---

## Architecture Overview

```
Client → API Gateway (REST) → Lambda (API) → DynamoDB / S3
              ↓                    │
    Cognito User Pool          (if cross-Lambda calls exist)
    (if auth in monolith)          │
                              ┌────┴─────────────────────────┐
                              │                              │
                         sync invoke            async publish
                         (RequestResponse)           │
                              │               ┌──────┴──────┐
                              ▼               ▼              ▼
                       Lambda (internal)   SQS Queue    EventBridge
                                              │         │         │
                                              ▼         ▼         ▼
                                        Lambda (sqs)  Lambda   Lambda
                                                     (eb-A)   (eb-B)
```

---

## Required Components

### 1. API Gateway (REST API)
- HTTP entry point replacing the monolith's web server
- One route per API Lambda; non-API Lambdas have NO API Gateway route
- Keep the same HTTP paths and methods from the monolith
- CORS enabled for external clients
- Cognito Authorizer attached only if auth exists in monolith

### 2. Lambda Functions

**ONE endpoint = ONE Lambda** (strict rule):
- Define an endpoint as (HTTP method + path), e.g. `GET /items/{id}`
- Create ONE separate Lambda function for EACH endpoint
- Do NOT group multiple endpoints into one Lambda (no router/switch)
- Organize folders by domain, keep handlers split

**Trigger types** (exactly four):
| trigger_type | Meaning | Key fields |
|---|---|---|
| `api` | HTTP endpoint via API Gateway | `entry_points` |
| `internal` | Called by other Lambdas via synchronous SDK invoke | — |
| `sqs` | Triggered by SQS queue messages | `sqs_source` (no `entry_points`) |
| `eventbridge` | Triggered by EventBridge rule | `event_pattern` (no `entry_points`) |

**Inter-Lambda Communication** — three mechanisms:
- **Synchronous Invoke** (caller NEEDS return value): `boto3 lambda.invoke(InvocationType='RequestResponse')`
- **SQS** (fire-and-forget, single consumer): `sqs.send_message()` → consumer Lambda
- **EventBridge** (fan-out, multiple consumers): `events.put_events()` → consumer Lambdas

Rules:
- Do NOT call another Lambda via HTTP (no `requests.post` / `axios.post` to API Gateway URL)
- Target FunctionName / QueueUrl / EventBus passed via environment variables, set via `!Ref` in SAM
- Invoker Lambda MUST have the corresponding SAM policy

**Handler pattern** (API Lambda):
```python
def lambda_handler(event, context):
    path_params = event.get('pathParameters', {})
    query_params = event.get('queryStringParameters', {})
    body = json.loads(event.get('body', '{}'))

    result = process_business_logic(...)

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(result)
    }
```

### 3. DynamoDB (if database used in monolith)
- Direct table structure migration; keep partition keys, sort keys, indexes
- Use `PAY_PER_REQUEST` billing mode
- **User Table decision**:
  - Only credentials (password/hash/salt/tokens) → **DELETE** entire table (Cognito replaces it)
  - Also has business profile data (avatar, preferences, address) → split:
    credentials → Cognito; business fields → DynamoDB `UserProfiles`
  - Only `name`/`email`/`phone` beyond credentials → use Cognito standard attributes, no `UserProfiles`
- Business tables: change user foreign key from integer to Cognito Sub UUID string

### 4. S3 (if file operations exist in monolith)
- Only add if the monolith has explicit file handling code (uploads, PDF generation, etc.)

### 5. Cognito User Pool (if auth exists in monolith)
- Replaces JWT-based or session-based authentication
- Auth endpoints (`/register`, `/login`, `/logout`, `/refresh-token`, `/me`) are DELETED — clients call Cognito API directly
- Token validation done by API Gateway Cognito Authorizer, not Lambda code
- Lambda receives pre-validated user identity: `event['requestContext']['authorizer']['claims']['sub']`

### 6. CloudWatch (always required)
- One Log Group per Lambda (retention: 7–30 days)
- Alarms: Lambda error rate > 1%, API Gateway 4XX/5XX rate, DynamoDB throttled requests

### 7. SQS Queues (if async point-to-point patterns identified)
- Used when caller does NOT need callee's result and there is a single consumer
- Always configure with a Dead-Letter Queue (DLQ) and `maxReceiveCount`
- `VisibilityTimeout` must be ≥ consumer Lambda timeout

### 8. EventBridge Rules (if fan-out patterns identified)
- Used when one event triggers 2+ independent consumers across different domains
- Semantic: "something happened" (event notification), not "do this task" (command)
- Each consumer Lambda needs `AWS::Lambda::Permission` for EventBridge invocation

---

## Migration Rules

### Rule 1: Preserve All Business Logic
- Copy business logic from monolith exactly
- Only change infrastructure layer (HTTP routing → API Gateway, DB → DynamoDB SDK, auth middleware → Cognito Authorizer)

### Rule 2: Delete Infrastructure Code
- `init_dynamodb.py`, `create_tables.py`, `setup_db.py`, migration scripts, server startup scripts
- Extract table schemas from these files; define tables in SAM template resources

### Rule 3: Delete User Authentication Table
- Do NOT migrate the user credentials table to serverless
- Exception: split business profile data into `UserProfiles` if the backend must own it

### Rule 4: Do Not Add Missing Features
- No auth in monolith → no Cognito
- No database → no DynamoDB
- No file storage → no S3

---

## Blueprint JSON Schema

The Architect writes `blueprint.json` with exactly these sections.
This example shows an application with both sync and async Lambda communication.

Runtime: Choose based on the monolith's source language. Use current runtimes only:
- Python projects → `python3.12`
- Node.js projects → `nodejs22.x`
Do NOT use deprecated runtimes (nodejs18.x was deprecated March 2026, nodejs20.x is
near end-of-life).

```json
{
  "metadata": {
    "application_name": "MyApp",
    "architecture_pattern": "Event-Driven Serverless Architecture",
    "has_authentication": true,
    "has_database": true,
    "has_file_storage": false,
    "has_async_flows": true
  },

  "lambda_functions": [
    {
      "name": "items-get-item",
      "trigger_type": "api",
      "runtime": "python3.12",
      "handler": "handler.lambda_handler",
      "source_files": ["app/routes/items.py", "app/services/item_service.py"],
      "entry_points": ["GET /items/{id}"],
      "environment_variables": {
        "ITEMS_TABLE": "${ItemsTable}"
      }
    },
    {
      "name": "order-checkout",
      "trigger_type": "api",
      "runtime": "python3.12",
      "handler": "handler.lambda_handler",
      "source_files": ["app/routes/orders.py", "app/services/order_service.py"],
      "entry_points": ["POST /orders/checkout"],
      "publishes_to": [
        {"type": "sqs", "target": "BestsellerUpdateQueue"},
        {"type": "eventbridge", "source": "app.orders", "detail_type": "OrderCompleted"}
      ],
      "environment_variables": {
        "ORDERS_TABLE": "${OrdersTable}",
        "BESTSELLER_QUEUE_URL": "${BestsellerUpdateQueue}"
      }
    },
    {
      "name": "update-bestsellers",
      "trigger_type": "sqs",
      "runtime": "python3.12",
      "handler": "handler.lambda_handler",
      "sqs_source": "BestsellerUpdateQueue",
      "source_files": ["utils/bestsellers.py"]
    },
    {
      "name": "process-loyalty",
      "trigger_type": "eventbridge",
      "runtime": "python3.12",
      "handler": "handler.lambda_handler",
      "event_pattern": {"source": ["app.orders"], "detail-type": ["OrderCompleted"]},
      "source_files": ["services/loyalty.py"]
    }
  ],

  "dynamodb_tables": [
    {
      "logical_name": "Orders",
      "purpose": "Store order records",
      "partition_key": {"name": "orderId", "type": "S"},
      "sort_key": {"name": "createdAt", "type": "S"},
      "gsi": [
        {
          "index_name": "userId-index",
          "partition_key": {"name": "userId", "type": "S"}
        }
      ]
    }
  ],

  "s3_buckets": [],

  "cognito": {
    "enabled": true,
    "user_pool_attributes": ["email", "name"],
    "password_policy": {
      "min_length": 8,
      "require_symbols": true
    }
  },

  "api_gateway": {
    "type": "REST",
    "cors": {
      "allow_origins": ["*"],
      "allow_methods": ["GET", "POST", "PUT", "DELETE"]
    },
    "authorizer": "Cognito"
  },

  "lambda_invoke_permissions": [
    {"invoker": "booking-create", "target": "payments-charge"}
  ],

  "sqs_queues": [
    {
      "logical_name": "BestsellerUpdateQueue",
      "purpose": "Async bestseller counter updates after checkout",
      "dlq": true,
      "visibility_timeout": 60,
      "max_receive_count": 3
    }
  ],

  "eventbridge_rules": [
    {
      "name": "OrderCompletedRule",
      "source": "app.orders",
      "detail_type": "OrderCompleted",
      "targets": ["process-loyalty", "send-notification"],
      "description": "Fan-out: order completion triggers loyalty and notification"
    }
  ],

  "dropped_functions": [
    {"item": "POST /register", "type": "endpoint", "reason": "Replaced by Cognito User Pool registration"},
    {"item": "POST /login", "type": "endpoint", "reason": "Replaced by Cognito User Pool authentication"},
    {"item": "POST /logout", "type": "endpoint", "reason": "Replaced by Cognito User Pool sign-out"},
    {"item": "verifyToken middleware", "type": "code", "reason": "Replaced by API Gateway Cognito Authorizer"},
    {"item": "init_dynamodb.py", "type": "script", "reason": "Table creation handled by SAM template"},
    {"item": "Users table", "type": "table", "reason": "User authentication moved to Cognito"}
  ]
}
```

### Blueprint field notes

- `metadata.architecture_pattern`: `"Basic Serverless Architecture"` if no async flows;
  `"Event-Driven Serverless Architecture"` if `sqs_queues` or `eventbridge_rules` are non-empty
- `metadata.has_async_flows`: `true` if `sqs_queues` or `eventbridge_rules` are non-empty
- `dynamodb_tables`: Include EXACT schema (partition_key, sort_key, gsi) extracted from source files.
  Attribute types: `"S"` (String), `"N"` (Number), `"B"` (Binary). Omit sort_key/gsi if not present.
- `lambda_invoke_permissions`: Only for synchronous Lambda-to-Lambda calls (mechanism A).
  Set to `[]` if no synchronous cross-Lambda calls.
- `sqs_queues` / `eventbridge_rules`: Set to `[]` if no async patterns identified.
- `dropped_functions`: Only if auth or infra code exists; items have `item`, `type`, `reason` fields.
- Consumer Lambdas (`trigger_type: "sqs"` / `"eventbridge"`) do NOT have `entry_points`.
- API Lambdas that publish async messages list targets in `publishes_to`.
- `source_files` for each Lambda: derived from `entry_point_dependencies` in analysis_report.json
  (handler file + all callee_module file paths).

---

## Common Mistakes

### Mistake 1: Keeping User Credentials Table
Delete the entire users table if it only stores auth artifacts. Cognito replaces it.

### Mistake 2: Creating Auth Lambda Functions
No Lambda for login/register/logout. Clients call Cognito API directly.

### Mistake 3: Creating Infrastructure Lambda
No Lambda for `init_dynamodb` or table setup. SAM template handles infrastructure.

### Mistake 4: Adding Features Not in Monolith
If monolith has no auth → no Cognito. If no database → no DynamoDB. If no files → no S3.

### Mistake 5: Calling Lambda via HTTP
Do NOT use `requests.post(api_gateway_url)` to invoke another Lambda. Use SDK invoke, SQS, or EventBridge.

---

## Summary

**Key Rules**:
1. One (HTTP method + path) = one Lambda function
2. Delete user credentials table → Cognito
3. Delete infrastructure scripts → SAM template
4. Delete auth endpoints → Cognito handles them
5. Preserve all business logic exactly
6. Do not add features that don't exist in monolith
7. Cross-Lambda communication: sync invoke OR SQS OR EventBridge (decided per-relationship via call graph analysis)

**Success Criteria**:
- All monolith API endpoints work in serverless
- Data is correctly stored and retrieved
- Authentication works (if it existed in monolith)
- Business logic produces same results as monolith
- Async workflows correctly decouple fire-and-forget operations
