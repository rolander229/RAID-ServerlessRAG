# AWS Lambda Coding Reference for Serverless Migration

This document contains authoritative patterns and rules for writing AWS Lambda handlers
when migrating monolithic applications to serverless. All patterns are sourced from
official AWS documentation.

---

## 1. Handler Signature

### Python (python3.12)
```python
def lambda_handler(event, context):
    ...
```
- File name convention: `handler.py`
- SAM Handler property: `handler.lambda_handler`
- Do NOT use `async def` — Python Lambda handlers must be synchronous.
- The handler must accept exactly two arguments (`event`, `context`). A single-argument
  handler will raise a runtime error.

### Node.js (nodejs22.x)
```javascript
exports.handler = async function(event, context) { ... };
```
- File name convention: `handler.js` (CommonJS — do NOT use `"type": "module"` in package.json)
- SAM Handler property: `handler.handler`
- Prefer `async/await` over callbacks.
- **MUST use CommonJS syntax** (`require` / `exports`) — NOT ESM (`import` / `export`).
  Reason: Lambda layers are mounted at `/opt/nodejs/`. ESM module resolution does NOT
  search `NODE_PATH`, so shared layer code and the runtime's built-in `@aws-sdk/*`
  packages cannot be resolved from ESM imports within layer code. CJS `require()` uses
  `NODE_PATH` which includes `/opt/nodejs/` and the runtime's SDK path, ensuring all
  dependencies resolve correctly.

---

## 2. API Gateway Proxy Integration — Event Structure

When a Lambda is triggered by API Gateway (REST API, proxy integration), the `event`
object has the following structure:

```json
{
  "httpMethod": "GET",
  "path": "/items/123",
  "pathParameters": { "id": "123" },
  "queryStringParameters": { "filter": "active" },
  "multiValueQueryStringParameters": { "tag": ["a", "b"] },
  "headers": { "Authorization": "Bearer eyJ..." },
  "body": "{\"name\": \"example\"}",
  "isBase64Encoded": false,
  "requestContext": {
    "authorizer": {
      "claims": {
        "sub": "uuid-cognito-user-id",
        "email": "user@example.com"
      }
    }
  }
}
```

### Extracting parameters (Python)
```python
def lambda_handler(event, context):
    # Path parameters: /items/{id} -> event['pathParameters']['id']
    item_id = (event.get('pathParameters') or {}).get('id')

    # Query string: ?filter=active
    query_params = event.get('queryStringParameters') or {}
    filter_val = query_params.get('filter')

    # Request body (always a JSON string from API Gateway)
    import json
    body = json.loads(event.get('body') or '{}')

    # Cognito user identity (pre-validated by API Gateway Cognito Authorizer)
    user_id = event['requestContext']['authorizer']['claims']['sub']
```

### Extracting parameters (Node.js)
```javascript
exports.handler = async (event) => {
    const itemId = (event.pathParameters || {}).id;
    const queryParams = event.queryStringParameters || {};
    const body = JSON.parse(event.body || '{}');
    const userId = event.requestContext.authorizer.claims.sub;
};
```

---

## 3. Response Format

API Gateway requires a specific response format. Returning anything else causes a
`502 Bad Gateway` error.

### Python
```python
import json

def make_response(status_code, body, headers=None):
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            **(headers or {})
        },
        'body': json.dumps(body)
    }

# Usage
return make_response(200, {'message': 'ok', 'data': result})
return make_response(400, {'error': 'Missing required field: name'})
return make_response(404, {'error': 'Item not found'})
return make_response(500, {'error': 'Internal server error'})
```

### Node.js (CommonJS)
```javascript
function makeResponse(statusCode, body, headers = {}) {
    return {
        statusCode,
        headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            ...headers
        },
        body: JSON.stringify(body)
    };
}

// Usage
return makeResponse(200, { message: 'ok', data: result });
return makeResponse(404, { error: 'Not found' });
```

---

## 4. Global Initialization (Critical for Performance)

Initialize SDK clients and loggers **outside** the handler function. Lambda reuses the
execution environment across invocations — code outside the handler runs only once per
cold start, not per invocation.

### Python
```python
import boto3
import logging
import os

# These run ONCE per cold start — reused across invocations
logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['TABLE_NAME'])

lambda_client = boto3.client('lambda')   # if this function invokes others

def lambda_handler(event, context):
    # Use the pre-initialized clients here
    ...
```

### Node.js (CommonJS — AWS SDK v3)
```javascript
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient } = require('@aws-sdk/lib-dynamodb');

// These run ONCE per cold start — reused across invocations
const ddbClient = new DynamoDBClient({});
const dynamoDB = DynamoDBDocumentClient.from(ddbClient, {
    marshallOptions: { removeUndefinedValues: true }
});
const tableName = process.env.TABLE_NAME;

exports.handler = async (event) => {
    // Use pre-initialized dynamoDB client
};
```

> **Rule**: Never initialize SDK clients (e.g. `new DynamoDBClient()`)
> inside the handler. Doing so creates a new connection on every invocation, wasting
> time and causing cold-start amplification.

---

## 5. Environment Variables

Never hardcode resource names (table names, bucket names, function names). Always use
environment variables, which are injected by the SAM template via `!Ref` / `!Sub`.

### Python
```python
import os

TABLE_NAME = os.environ['ITEMS_TABLE']                    # required — raises KeyError if missing
REGION = os.environ.get('AWS_REGION', 'us-east-1')        # optional with default
TARGET_FUNCTION = os.environ['NOTIFY_FUNCTION_NAME']      # for Lambda-to-Lambda invocation
```

### Node.js (CommonJS)
```javascript
const TABLE_NAME = process.env.ITEMS_TABLE;
if (!TABLE_NAME) throw new Error('ITEMS_TABLE environment variable is not set');
const TARGET_FUNCTION = process.env.NOTIFY_FUNCTION_NAME;
```

---

## 6. DynamoDB Access Patterns (boto3 / aws-sdk v3)

### Python — boto3 DynamoDB resource

```python
import boto3, os, json
from boto3.dynamodb.conditions import Key, Attr

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['TABLE_NAME'])

# GetItem — single item by primary key
response = table.get_item(Key={'id': item_id})
item = response.get('Item')
if not item:
    return make_response(404, {'error': 'Not found'})

# PutItem — create / overwrite
table.put_item(Item={
    'id': item_id,
    'userId': user_id,
    'name': name,
    'createdAt': datetime.utcnow().isoformat()
})

# UpdateItem — partial update
table.update_item(
    Key={'id': item_id},
    UpdateExpression='SET #n = :name, updatedAt = :ts',
    ExpressionAttributeNames={'#n': 'name'},   # 'name' is a reserved word
    ExpressionAttributeValues={':name': new_name, ':ts': datetime.utcnow().isoformat()}
)

# DeleteItem
table.delete_item(Key={'id': item_id})

# Query — items by partition key (+ optional sort key / filter)
response = table.query(
    KeyConditionExpression=Key('userId').eq(user_id)
)
items = response.get('Items', [])

# Query GSI
response = table.query(
    IndexName='userId-index',
    KeyConditionExpression=Key('userId').eq(user_id)
)

# Scan (avoid in production; use Query + GSI instead)
response = table.scan(FilterExpression=Attr('status').eq('active'))
```

### Node.js — AWS SDK v3 DynamoDBDocumentClient (CommonJS)

```javascript
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, GetCommand, PutCommand, UpdateCommand, DeleteCommand, QueryCommand } = require('@aws-sdk/lib-dynamodb');

const ddbClient = new DynamoDBClient({});
const dynamoDB = DynamoDBDocumentClient.from(ddbClient, {
    marshallOptions: { removeUndefinedValues: true }
});
const TABLE_NAME = process.env.TABLE_NAME;

// GetItem
const { Item } = await dynamoDB.send(new GetCommand({
    TableName: TABLE_NAME,
    Key: { id: itemId }
}));

// PutItem
await dynamoDB.send(new PutCommand({
    TableName: TABLE_NAME,
    Item: { id: itemId, userId, name, createdAt: new Date().toISOString() }
}));

// UpdateItem
await dynamoDB.send(new UpdateCommand({
    TableName: TABLE_NAME,
    Key: { id: itemId },
    UpdateExpression: 'SET #n = :name, updatedAt = :ts',
    ExpressionAttributeNames: { '#n': 'name' },
    ExpressionAttributeValues: { ':name': newName, ':ts': new Date().toISOString() }
}));

// DeleteItem
await dynamoDB.send(new DeleteCommand({
    TableName: TABLE_NAME,
    Key: { id: itemId }
}));

// Query (with GSI)
const { Items } = await dynamoDB.send(new QueryCommand({
    TableName: TABLE_NAME,
    IndexName: 'userId-index',
    KeyConditionExpression: 'userId = :uid',
    ExpressionAttributeValues: { ':uid': userId }
}));
```

> **Node.js SDK Rule**: Always use AWS SDK **v3** (`@aws-sdk/*` modular packages).
> Do NOT use v2 (`aws-sdk`) — v2 reached end-of-support on September 8, 2025.
> Node.js 18+ Lambda runtimes include v3 by default; bundling v2 adds unnecessary
> dependency weight and increases cold-start time.

---

## 7. S3 Access Patterns

### Python
```python
import boto3, os

s3 = boto3.client('s3')
BUCKET = os.environ['FILE_BUCKET']

# Upload
s3.put_object(Bucket=BUCKET, Key=f'uploads/{file_key}', Body=file_content)

# Download
response = s3.get_object(Bucket=BUCKET, Key=f'uploads/{file_key}')
content = response['Body'].read()

# Generate pre-signed URL (for client-side upload/download)
url = s3.generate_presigned_url(
    'put_object',
    Params={'Bucket': BUCKET, 'Key': f'uploads/{file_key}'},
    ExpiresIn=3600
)

# Delete
s3.delete_object(Bucket=BUCKET, Key=f'uploads/{file_key}')
```

### Node.js (AWS SDK v3, CommonJS)
```javascript
const { S3Client, PutObjectCommand, GetObjectCommand, DeleteObjectCommand } = require('@aws-sdk/client-s3');
const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');

const s3 = new S3Client({});
const BUCKET = process.env.FILE_BUCKET;

// Upload
await s3.send(new PutObjectCommand({
    Bucket: BUCKET, Key: `uploads/${fileKey}`, Body: fileContent
}));

// Download
const response = await s3.send(new GetObjectCommand({
    Bucket: BUCKET, Key: `uploads/${fileKey}`
}));
const content = await response.Body.transformToString();

// Generate pre-signed URL (for client-side upload/download)
const url = await getSignedUrl(s3, new PutObjectCommand({
    Bucket: BUCKET, Key: `uploads/${fileKey}`
}), { expiresIn: 3600 });

// Delete
await s3.send(new DeleteObjectCommand({
    Bucket: BUCKET, Key: `uploads/${fileKey}`
}));
```

---

## 8. Cross-Lambda Communication Patterns

Three mechanisms for cross-Lambda communication. Choose based on the blueprint:

### 8a. Synchronous Invoke (lambda_invoke_permissions)

Caller NEEDS the callee's return value for its own HTTP response.

```python
import boto3, json, os

lambda_client = boto3.client('lambda')

response = lambda_client.invoke(
    FunctionName=os.environ['PROCESS_PAYMENT_FUNCTION_NAME'],
    InvocationType='RequestResponse',
    Payload=json.dumps({'amount': amount, 'userId': user_id})
)
result = json.loads(response['Payload'].read())
if result.get('statusCode') != 200:
    raise Exception(f"Payment failed: {result.get('body')}")
```

```javascript
const { LambdaClient, InvokeCommand } = require('@aws-sdk/client-lambda');

const lambdaClient = new LambdaClient({});
const response = await lambdaClient.send(new InvokeCommand({
    FunctionName: process.env.PROCESS_PAYMENT_FUNCTION_NAME,
    InvocationType: 'RequestResponse',
    Payload: JSON.stringify({ amount, userId })
}));
const result = JSON.parse(new TextDecoder().decode(response.Payload));
```

### 8b. SQS Publish (publishes_to type="sqs")

Fire-and-forget: caller does NOT need result. Single consumer.

```python
import boto3, json, os

sqs = boto3.client('sqs')

sqs.send_message(
    QueueUrl=os.environ['BESTSELLER_QUEUE_URL'],
    MessageBody=json.dumps({
        'orderId': order_id,
        'books': purchased_books
    })
)
```

```javascript
const { SQSClient, SendMessageCommand } = require('@aws-sdk/client-sqs');

const sqsClient = new SQSClient({});
// ⚠️ MUST await — Lambda freezes execution after handler returns;
// non-awaited sends may be silently dropped.
await sqsClient.send(new SendMessageCommand({
    QueueUrl: process.env.BESTSELLER_QUEUE_URL,
    MessageBody: JSON.stringify({ orderId, books: purchasedBooks })
}));
```

### 8c. EventBridge Publish (publishes_to type="eventbridge")

Event notification: one or more consumers react independently.

```python
import boto3, json

events = boto3.client('events')

events.put_events(Entries=[{
    'Source': 'app.orders',
    'DetailType': 'OrderCompleted',
    'Detail': json.dumps({
        'orderId': order_id,
        'userId': user_id,
        'totalAmount': total
    })
}])
```

```javascript
const { EventBridgeClient, PutEventsCommand } = require('@aws-sdk/client-eventbridge');

const ebClient = new EventBridgeClient({});
// ⚠️ MUST await — same reason as SQS above.
await ebClient.send(new PutEventsCommand({
    Entries: [{
        Source: 'app.orders',
        DetailType: 'OrderCompleted',
        Detail: JSON.stringify({ orderId, userId, totalAmount: total })
    }]
}));
```

> **Rules**:
> - Do NOT use `lambda.invoke(InvocationType='Event')` for async — use SQS or EventBridge.
> - Do NOT use HTTP fetch/requests/axios to call another Lambda via API Gateway URL.
> - Target FunctionName / QueueUrl / EventBus read from environment variables.

---

## 8d. SQS Consumer Handler Pattern (trigger_type="sqs")

SQS-triggered Lambdas receive `event['Records']`. No API Gateway response format needed.

```python
import json, logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    for record in event['Records']:
        payload = json.loads(record['body'])
        process_item(payload)
```

```javascript
exports.handler = async (event) => {
    for (const record of event.Records) {
        const payload = JSON.parse(record.body);
        await processItem(payload);
    }
};
```

## 8e. EventBridge Consumer Handler Pattern (trigger_type="eventbridge")

EventBridge-triggered Lambdas receive event detail directly. No API Gateway response format needed.

```python
import json, logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    detail = event['detail']
    user_id = detail['userId']
    order_id = detail['orderId']
    add_loyalty_points(user_id, order_id)
```

```javascript
exports.handler = async (event) => {
    const { userId, orderId } = event.detail;
    await addLoyaltyPoints(userId, orderId);
};
```

> **Consumer rules**:
> - SQS consumers iterate `event['Records']` and parse `record['body']`.
> - EventBridge consumers access `event['detail']` directly.
> - Neither type returns `{ statusCode, headers, body }` — no API Gateway wrapper.
> - Consumer Lambdas should be **idempotent** (safe to process the same message twice).

---

## 9. Cognito User Identity

When API Gateway is configured with a Cognito Authorizer, the JWT token is validated
**before** Lambda is invoked. Lambda receives the pre-validated user claims in
`event.requestContext.authorizer.claims`.

```python
def lambda_handler(event, context):
    # Cognito Sub (UUID string) — use as the userId foreign key in DynamoDB
    user_id = event['requestContext']['authorizer']['claims']['sub']
    email   = event['requestContext']['authorizer']['claims'].get('email')
```

```javascript
// Inside exports.handler = async (event) => { ... }
const userId = event.requestContext.authorizer.claims.sub;
const email  = event.requestContext.authorizer.claims.email;
```

> **Rule**: Do NOT write JWT validation code in Lambda. Do NOT call Cognito APIs to
> verify tokens. API Gateway has already validated the token. Lambda only reads claims.

---

## 10. Error Handling Pattern

### Python — Standard pattern
```python
import json, logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    try:
        # Validate required inputs
        body = json.loads(event.get('body') or '{}')
        name = body.get('name')
        if not name:
            return {'statusCode': 400, 'body': json.dumps({'error': 'name is required'})}

        # Business logic
        result = do_work(name)

        logger.info(f"Success: {result}")
        return {'statusCode': 200, 'body': json.dumps({'data': result})}

    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}", exc_info=True)
        return {'statusCode': 500, 'body': json.dumps({'error': 'Internal server error'})}
```

### Node.js — Standard pattern (CommonJS)
```javascript
exports.handler = async (event) => {
    try {
        const body = JSON.parse(event.body || '{}');
        if (!body.name) {
            return { statusCode: 400, body: JSON.stringify({ error: 'name is required' }) };
        }
        const result = await doWork(body.name);
        return { statusCode: 200, body: JSON.stringify({ data: result }) };
    } catch (err) {
        console.error('Unhandled error:', err);
        return { statusCode: 500, body: JSON.stringify({ error: 'Internal server error' }) };
    }
};
```

> **Rule**: Always catch exceptions and return a proper API Gateway response object.
> Never let the Lambda runtime throw an uncaught exception for client-visible errors,
> as this causes API Gateway to return a 502.

---

## 11. Dependency Files

### Python — requirements.txt
```
# boto3 is pre-installed in Lambda Python runtimes.
# Include it only if you need to pin a specific version for reproducible builds.
# boto3>=1.34.0
```
- `boto3` is pre-installed in all Python Lambda runtimes. Only include it in
  `requirements.txt` if you need to pin a specific version for reproducible builds.
- Include only 3rd-party libraries used by this specific Lambda.
- Do NOT include `aws-lambda-powertools` unless explicitly used by the monolith.
- Do NOT include libraries already provided by a shared Lambda Layer.

### Node.js — package.json
```json
{
  "name": "my-lambda",
  "version": "1.0.0",
  "main": "handler.js",
  "engines": { "node": ">=22.0.0" },
  "dependencies": {
    "uuid": "^10.0.0"
  }
}
```
- **Do NOT** add `"type": "module"` — all handler.js files use CommonJS syntax.
- Always specify `engines.node` to match the runtime (`nodejs22.x` → `>=22.0.0`).
- Do NOT assume all `@aws-sdk/*` packages are built into the nodejs22.x Lambda runtime.
  Critical exception: `@aws-sdk/lib-dynamodb` is **not** runtime built-in and must be
  added to `package.json` or a shared layer package when used.
- `@aws-sdk/client-*` service packages are often available in the runtime, but helper
  libraries such as `@aws-sdk/lib-dynamodb` are not. When in doubt, prefer packaging
  the exact SDK helper library your code imports.
- Only include **non-runtime** dependencies actually used by the Lambda. AWS SDK helper
  packages like `@aws-sdk/lib-dynamodb` count as real dependencies.
- If the Lambda has NO external dependencies beyond built-in modules and runtime-bundled
  SDK clients, **do NOT create a package.json at all**. SAM will package the handler code as-is.
- Only create package.json when the Lambda truly needs a third-party npm package.

---

## 12. File & Directory Structure

```
output/
  lambdas/
    {domain}/
      {lambda-name}/
        handler.py          # (Python) or handler.js (Node.js, CommonJS)
        requirements.txt    # (Python only, ONLY if non-runtime deps exist)
        package.json        # (Node.js only, ONLY if non-AWS third-party deps exist)
  layers/
    shared/                 # Only if 3+ functions share utilities
      python/
        shared_utils.py
      requirements.txt      # (ONLY if non-runtime deps exist)
      nodejs/               # (Node.js layer structure)
        shared.js           # CommonJS: module.exports = { ... }
        package.json        # (ONLY if non-AWS third-party deps exist; NO "type": "module")
```

- One directory per Lambda function.
- `{domain}` is inferred from the monolith's route/module organization.
- The SAM `CodeUri` for each function must exactly match the directory path.

---

## 13. Things to NEVER do in Lambda Handlers

| Anti-Pattern | Correct Alternative |
|---|---|
| `boto3.client()` / `new DynamoDBClient()` inside the handler | Initialize SDK clients outside the handler |
| Hardcode `TableName = 'my-table'` | Use `os.environ['TABLE_NAME']` |
| Use HTTP requests to call another Lambda | Use SDK invoke, SQS, or EventBridge |
| Use `lambda.invoke(InvocationType='Event')` for async | Use SQS `send_message` or EventBridge `put_events` |
| Write JWT validation / token decode logic | Read `event['requestContext']['authorizer']['claims']` |
| Create a Users table for authentication | Use Cognito — no Lambda auth code needed |
| Return bare Python dict without `statusCode` (API Lambda) | Always return `{'statusCode': ..., 'body': ...}` |
| Return `{statusCode, body}` from SQS/EventBridge consumer | Consumer Lambdas have no API Gateway response format |
| `import *` from heavy frameworks (Flask, Django) | Import only what is needed |
| Use ESM `import/export` in Node.js Lambda handler.js | Use CJS `require()` / `exports.handler` (layer compat) |
| Add `"type": "module"` to package.json | Omit it — all .js files must be CJS |
| Omit `@aws-sdk/lib-dynamodb` from dependencies | Package it explicitly in the function or shared layer |
| Fire-and-forget SQS/EventBridge send (no `await`) | Always `await` the send call before returning |
| Store user data in global state between invocations | Keep state in DynamoDB / S3 |
