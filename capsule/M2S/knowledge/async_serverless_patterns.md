# Async & Event-Driven Serverless Patterns

This document covers asynchronous communication patterns for serverless migrations.
Use these patterns when the architecture blueprint identifies async candidates
via cross-Lambda call graph analysis.

---

## Decision Flowchart: Sync vs Async

For each cross-Lambda call relationship identified in `entry_point_dependencies`:

```
Q1: Does the caller NEED the callee's return value for its HTTP response?
  │
  ├─ YES → SYNCHRONOUS Lambda Invoke (RequestResponse)
  │
  └─ NO → Q2: Does this action trigger operations in 2+ different service domains?
           │
           ├─ YES → EventBridge (fan-out / event-driven)
           │
           └─ NO → SQS (point-to-point async)
```

---

## Pattern 1: SQS — Point-to-Point Async

**When to use**:
- One producer, one consumer
- The caller does NOT need the callee's return value
- The callee performs a side-effect that can tolerate seconds/minutes of delay
- Examples: counter updates, email/notification sending, log enrichment, image processing

**Key configuration**:
- `VisibilityTimeout`: must be ≥ consumer Lambda timeout (SQS default: 30s; recommend ≥ 60s)
- Dead-Letter Queue (DLQ): always configure with `maxReceiveCount: 3`
- `BatchSize`: 1–10,000 messages per invocation for Standard queues (1–10 for FIFO); default: 10

### Producer Code (in API Lambda)

```python
import boto3, json, os

sqs = boto3.client('sqs')

def lambda_handler(event, context):
    # ... process main request ...
    order_id = create_order(...)

    # Send message to SQS instead of calling function directly
    sqs.send_message(
        QueueUrl=os.environ['BESTSELLER_QUEUE_URL'],
        MessageBody=json.dumps({
            'orderId': order_id,
            'books': purchased_books
        })
    )

    # Return HTTP response after SQS message is confirmed sent
    return {
        'statusCode': 200,
        'body': json.dumps({'orderId': order_id})
    }
```

```javascript
// Node.js (CommonJS) — ⚠️ MUST await the send call
const { SQSClient, SendMessageCommand } = require('@aws-sdk/client-sqs');
const sqsClient = new SQSClient({});

exports.handler = async (event) => {
    const orderId = await createOrder(/* ... */);

    // ⚠️ MUST await — Lambda freezes execution after handler returns.
    // Non-awaited async sends (.then().catch() without await, or fire-and-forget
    // promises) may be silently dropped, causing message loss.
    await sqsClient.send(new SendMessageCommand({
        QueueUrl: process.env.BESTSELLER_QUEUE_URL,
        MessageBody: JSON.stringify({ orderId, books: purchasedBooks })
    }));

    return { statusCode: 200, body: JSON.stringify({ orderId }) };
};
```

### Consumer Lambda (SQS-triggered)

```python
import json

def lambda_handler(event, context):
    for record in event['Records']:
        payload = json.loads(record['body'])
        process_bestseller_update(payload['books'])

    # No HTTP response format needed (no statusCode/headers/body)
    # Return value is ignored by SQS trigger
```

### SAM Template

```yaml
# Queue + DLQ
BestsellerUpdateQueue:
  Type: AWS::SQS::Queue
  Properties:
    QueueName: !Sub '${AWS::StackName}-bestseller-updates'
    VisibilityTimeout: 60
    RedrivePolicy:
      deadLetterTargetArn: !GetAtt BestsellerUpdateDLQ.Arn
      maxReceiveCount: 3

BestsellerUpdateDLQ:
  Type: AWS::SQS::Queue
  Properties:
    QueueName: !Sub '${AWS::StackName}-bestseller-updates-dlq'

# Consumer Lambda (SQS-triggered)
UpdateBestsellersFunction:
  Type: AWS::Serverless::Function
  Properties:
    CodeUri: lambdas/orders/update-bestsellers/
    Handler: handler.lambda_handler
    Events:
      SQSEvent:
        Type: SQS
        Properties:
          Queue: !GetAtt BestsellerUpdateQueue.Arn
          BatchSize: 10
    Policies:
      - SQSPollerPolicy:
          QueueName: !GetAtt BestsellerUpdateQueue.QueueName

# Producer Lambda needs SQSSendMessagePolicy + env var
OrderCheckoutFunction:
  Type: AWS::Serverless::Function
  Properties:
    # ... standard Api Event ...
    Policies:
      - SQSSendMessagePolicy:
          QueueName: !GetAtt BestsellerUpdateQueue.QueueName
    Environment:
      Variables:
        BESTSELLER_QUEUE_URL: !Ref BestsellerUpdateQueue
```

---

## Pattern 2: EventBridge — Fan-Out / Event-Driven

**When to use**:
- One event triggers multiple independent consumers across different service domains
- Semantic: "something happened" (event notification), not "do this task" (command)
- Cross-domain decoupling: producer doesn't know/care about consumers
- Event chains: each step emits a "completed" event triggering the next step
- Examples: order completed → loyalty points + notification + analytics

**Key configuration**:
- Use the default event bus (no custom bus needed for most cases)
- Event pattern matching on `source` + `detail-type`
- Each consumer needs `AWS::Lambda::Permission` for EventBridge invocation

### Producer Code (in API Lambda)

```python
import boto3, json, os

events = boto3.client('events')

def lambda_handler(event, context):
    # ... process main request ...
    order = complete_order(...)

    # Publish event — multiple consumers will react independently
    events.put_events(Entries=[{
        'Source': 'app.orders',
        'DetailType': 'OrderCompleted',
        'Detail': json.dumps({
            'orderId': order['id'],
            'userId': order['userId'],
            'totalAmount': order['total']
        })
    }])

    return {
        'statusCode': 200,
        'body': json.dumps({'orderId': order['id'], 'status': 'completed'})
    }
```

```javascript
// Node.js (CommonJS) — ⚠️ MUST await the put_events call
const { EventBridgeClient, PutEventsCommand } = require('@aws-sdk/client-eventbridge');
const ebClient = new EventBridgeClient({});

exports.handler = async (event) => {
    const order = await completeOrder(/* ... */);

    await ebClient.send(new PutEventsCommand({
        Entries: [{
            Source: 'app.orders',
            DetailType: 'OrderCompleted',
            Detail: JSON.stringify({
                orderId: order.id,
                userId: order.userId,
                totalAmount: order.total
            })
        }]
    }));

    return { statusCode: 200, body: JSON.stringify({ orderId: order.id }) };
};
```

### Consumer Lambda (EventBridge-triggered)

```python
import json

def lambda_handler(event, context):
    detail = event['detail']
    user_id = detail['userId']
    order_id = detail['orderId']

    add_loyalty_points(user_id, order_id)

    # No HTTP response format needed
```

### SAM Template

```yaml
# EventBridge Rule
OrderCompletedRule:
  Type: AWS::Events::Rule
  Properties:
    EventPattern:
      source:
        - "app.orders"
      detail-type:
        - "OrderCompleted"
    Targets:
      - Arn: !GetAtt ProcessLoyaltyFunction.Arn
        Id: "ProcessLoyalty"
      - Arn: !GetAtt SendNotificationFunction.Arn
        Id: "SendNotification"

# Permission for EventBridge to invoke each target Lambda
ProcessLoyaltyEventPermission:
  Type: AWS::Lambda::Permission
  Properties:
    FunctionName: !Ref ProcessLoyaltyFunction
    Action: lambda:InvokeFunction
    Principal: events.amazonaws.com
    SourceArn: !GetAtt OrderCompletedRule.Arn

SendNotificationEventPermission:
  Type: AWS::Lambda::Permission
  Properties:
    FunctionName: !Ref SendNotificationFunction
    Action: lambda:InvokeFunction
    Principal: events.amazonaws.com
    SourceArn: !GetAtt OrderCompletedRule.Arn

# Producer Lambda needs EventBridgePutEventsPolicy
OrderCheckoutFunction:
  Type: AWS::Serverless::Function
  Properties:
    # ... standard Api Event ...
    Policies:
      - EventBridgePutEventsPolicy:
          EventBusName: default
```

---

## Pattern 3: Synchronous Lambda Invoke (Keep When Needed)

**When to use**:
- The caller MUST wait for the callee's result to construct its own HTTP response
- Data consistency requires atomic/synchronous execution
- Examples: payment charge → order confirmation, inventory check → availability

```python
import boto3, json, os

lambda_client = boto3.client('lambda')

response = lambda_client.invoke(
    FunctionName=os.environ['PROCESS_PAYMENT_FUNCTION_NAME'],
    InvocationType='RequestResponse',
    Payload=json.dumps({'chargeId': charge_id, 'amount': total})
)
result = json.loads(response['Payload'].read())
```

---

## Consumer Lambda Design Notes

- SQS consumers receive `event['Records']` — iterate and parse `record['body']`
- EventBridge consumers receive `event['detail']` directly
- Neither type returns API Gateway response format (no `statusCode`/`headers`/`body`)
- Consumer Lambdas have `trigger_type` of `"sqs"` or `"eventbridge"` in the blueprint
- Consumer Lambdas do NOT have `entry_points` (they are not API-triggered)
- Consumer Lambda `source_files` come from the callee module in the original monolith

## Error Handling Best Practices

- **SQS**: Failed messages retry automatically (up to `maxReceiveCount`), then go to DLQ
- **EventBridge**: Failed invocations can be configured with retry policy and DLQ on the rule target
- **Both**: Consumer Lambdas should be idempotent (safe to process the same message twice)
