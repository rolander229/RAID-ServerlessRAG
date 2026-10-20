# AWS SAM Reference & Patterns (Simplified)

This knowledge base provides essential AWS SAM templates, resource definitions, and policy patterns for the Serverless Migration project. This is a simplified version focusing on the most commonly used patterns.

## 1. Standard SAM Template Structure

### Header & Globals
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: Serverless application infrastructure

Globals:
  Function:
    # Basic Configuration (Required)
    Runtime: python3.12  # Adjust based on actual code
    Handler: index.handler
    Timeout: 30  # seconds
    MemorySize: 512  # MB
    
    # Environment Variables (Common)
    Environment:
      Variables:
        ENVIRONMENT: !Ref Environment
        LOG_LEVEL: INFO
    
    # Monitoring & Tracing (Recommended)
    Tracing: Active  # Enable X-Ray tracing
    
    # Shared Layers (Optional)
    Layers:
      - !Ref CommonLayer
    
    # Tags (Recommended)
    Tags:
      Project: MyProject
      Environment: !Ref Environment
```

### Parameters
```yaml
Parameters:
  Environment:
    Type: String
    Default: dev
    AllowedValues: [dev, staging, prod]
```

## 2. Resource Patterns

### Lambda Function Pattern
```yaml
{LambdaName}Function:
  Type: AWS::Serverless::Function
  Properties:
    # ========== Required Configuration ==========
    CodeUri: backend/lambdas/{lambda_name}/
    Handler: handler.lambda_handler
    Runtime: python3.12  # python3.12, nodejs22.x, etc.
    
    # ========== Common Configuration ==========
    MemorySize: 512  # 128-10240 MB (affects CPU and cost)
    Timeout: 30  # seconds (max 900)
    
    Description: "Function description"  # Optional but recommended
    
    # Environment Variables
    Environment:
      Variables:
        TABLE_NAME: !Ref TableName
        BUCKET_NAME: !Ref BucketName
    
    # Shared Code Layers
    Layers:
      - !Ref CommonLayer
    
    # Permission Policies (choose one approach)
    Policies:
      # Approach 1: SAM Policy Templates (Recommended)
      - DynamoDBCrudPolicy:
          TableName: !Ref TableName
      
      # Approach 2: Inline Policy (Fine-grained control)
      - Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Action:
              - s3:GetObject
              - s3:PutObject
            Resource: !Sub arn:aws:s3:::${BucketName}/*
          
          # AI/ML Services
          - Effect: Allow
            Action:
              - comprehend:DetectSentiment
              - rekognition:DetectLabels
            Resource: '*'
    
    # ========== Optional Configuration (Specific scenarios) ==========
    
    # Concurrency Limits (Prevent cost runaway)
    ReservedConcurrentExecutions: 100
    
    # Version Management (For blue-green deployments)
    AutoPublishAlias: live
    
    # Async Invocation Config (Error handling)
    EventInvokeConfig:
      MaximumRetryAttempts: 1
      DestinationConfig:
        OnFailure:
          Type: SQS
          Destination: !GetAtt DLQ.Arn
    
    # Event Triggers
    Events:
      # API Trigger
      ApiEvent:
        Type: Api
        Properties:
          Path: /api/resource
          Method: GET
          RestApiId: !Ref MyApi
          # Auth is inherited from API's DefaultAuthorizer — do NOT repeat it here.
          # Only add Auth to OVERRIDE (e.g., Auth: { Authorizer: NONE } for public routes).
```

### API Gateway Pattern (REST API)

**Recommended: Explicitly define API resource, let SAM auto-generate paths**

```yaml
MyApi:
  Type: AWS::Serverless::Api
  Properties:
    # Required
    StageName: !Ref Environment
    
    # CORS Configuration (Required for frontend apps)
    Cors:
      AllowMethods: "'*'"
      AllowHeaders: "'*'"
      AllowOrigin: "'*'"
    
    # Authentication & Authorization (Optional)
    Auth:
      DefaultAuthorizer: CognitoAuthorizer
      Authorizers:
        CognitoAuthorizer:
          UserPoolArn: !GetAtt UserPool.Arn
    
    # ⚠️ IMPORTANT — Per-Event Auth Rules:
    # When DefaultAuthorizer is set on the API, ALL events attached to this API
    # automatically inherit it. You do NOT need to add Auth on each event.
    #
    # CRITICAL: For PUBLIC endpoints (e.g., product catalog, search, public listings),
    # you MUST add Auth: Authorizer: NONE to override the DefaultAuthorizer.
    # Without this override, public endpoints will return 401 Unauthorized.
    #
    # Example — public endpoint (no auth required):
    #   Events:
    #     PublicEndpoint:
    #       Type: Api
    #       Properties:
    #         Path: /books
    #         Method: GET
    #         RestApiId: !Ref MyApi
    #         Auth:
    #           Authorizer: NONE          # ← Override: no auth for this route
    #
    # Example — authenticated endpoint (inherits DefaultAuthorizer, NO Auth block needed):
    #   Events:
    #     ProtectedEndpoint:
    #       Type: Api
    #       Properties:
    #         Path: /cart
    #         Method: POST
    #         RestApiId: !Ref MyApi
    #         # Auth is inherited — do NOT add Auth block here
    #
    # If you DO need to explicitly set an authorizer on an event, Auth MUST be a map:
    #   ✅ Correct:
    #     Auth:
    #       Authorizer: CognitoAuthorizer
    #   ❌ Wrong (string, will fail validation):
    #     Auth: CognitoAuthorizer

    # Access Logs (Recommended for production)
    AccessLogSetting:
      DestinationArn: !GetAtt ApiLogGroup.Arn
      Format: '$context.requestId $context.error.message'
    
    # X-Ray Tracing (Recommended)
    TracingEnabled: true
```

### CORS Configuration Best Practice
⚠️ **IMPORTANT**: Define CORS properties as simple strings.
❌ **DO NOT** use `!Sub` with a list or object structure for the Cors property.
❌ **DO NOT** try to parameterize lists of headers inside the Cors object.

✅ **Correct**:
```yaml
Cors:
  AllowOrigin: "'*'"
  AllowHeaders: "'Content-Type,Authorization'"
  AllowMethods: "'GET,POST,OPTIONS'"
```

### API Log Group (Companion Resource for API Gateway)
```yaml
ApiLogGroup:
  Type: AWS::Logs::LogGroup
  Properties:
    LogGroupName: !Sub /aws/apigateway/${MyApi}
    RetentionInDays: 30
```

### DynamoDB Table Pattern

```yaml
{TableName}:
  Type: AWS::DynamoDB::Table
  # ⚠️ Data Protection Policy (Strongly Recommended)
  # Prevents data loss when stack is deleted
  DeletionPolicy: Retain
  UpdateReplacePolicy: Retain
  Properties:
    # Required
    TableName: !Sub ${Environment}-{table-name}
    
    # Key Definitions (Required)
    # ⚠️ CRITICAL: List EVERY attribute used in KeySchema OR any GSI KeySchema here.
    # Attributes NOT used as keys (e.g., name, email, status) must NOT be listed here.
    AttributeDefinitions:
      - AttributeName: userId
        AttributeType: S
      - AttributeName: createdAt
        AttributeType: S
    
    KeySchema:
      - AttributeName: userId
        KeyType: HASH
      - AttributeName: createdAt
        KeyType: RANGE
    
    # Billing Mode (Pay-per-request recommended)
    BillingMode: PAY_PER_REQUEST
    
    # Tags
    Tags:
      - Key: Environment
        Value: !Ref Environment
```

### DynamoDB Table Pattern — With GSI (Global Secondary Index)

```yaml
# ⚠️ GSI Rules:
# 1. Every attribute used in any GSI KeySchema MUST be declared in the top-level AttributeDefinitions.
# 2. GSIs do not require a RANGE key — a HASH-only GSI is valid.
# 3. Do NOT declare non-key attributes (e.g., title, status) in AttributeDefinitions.

{TableName}:
  Type: AWS::DynamoDB::Table
  DeletionPolicy: Retain
  UpdateReplacePolicy: Retain
  Properties:
    TableName: !Sub ${Environment}-{table-name}
    AttributeDefinitions:
      - AttributeName: pk          # Table partition key
        AttributeType: S
      - AttributeName: sk          # Table sort key
        AttributeType: S
      - AttributeName: userId      # GSI partition key — MUST be listed here
        AttributeType: S
    KeySchema:
      - AttributeName: pk
        KeyType: HASH
      - AttributeName: sk
        KeyType: RANGE
    GlobalSecondaryIndexes:
      - IndexName: userId-index
        KeySchema:
          - AttributeName: userId  # Must match an entry in AttributeDefinitions above
            KeyType: HASH
        Projection:
          ProjectionType: ALL      # ALL | KEYS_ONLY | INCLUDE
    BillingMode: PAY_PER_REQUEST
    Tags:
      - Key: Environment
        Value: !Ref Environment
```

### S3 Bucket Pattern
```yaml
{BucketName}:
  Type: AWS::S3::Bucket
  # ⚠️ Data Protection Policy (Strongly Recommended)
  # Prevents data loss when stack is deleted
  DeletionPolicy: Retain
  UpdateReplacePolicy: Retain
  Properties:
    BucketName: !Sub ${Environment}-{bucket-name}
    
    # Security Configuration (Recommended)
    PublicAccessBlockConfiguration:
      BlockPublicAcls: true
      BlockPublicPolicy: true
      IgnorePublicAcls: true
      RestrictPublicBuckets: true
    
    # CORS Configuration (Required for frontend uploads)
    CorsConfiguration:
      CorsRules:
        - AllowedOrigins: ["https://example.com"]
          AllowedMethods: [GET, PUT, POST]
          AllowedHeaders: ["*"]
```

### SQS Queue Pattern (with Dead-Letter Queue)

```yaml
<QueueLogicalName>:
  Type: AWS::SQS::Queue
  Properties:
    QueueName: !Sub '${AWS::StackName}-<kebab-name>'
    VisibilityTimeout: 60
    RedrivePolicy:
      deadLetterTargetArn: !GetAtt <QueueLogicalName>DLQ.Arn
      maxReceiveCount: 3

<QueueLogicalName>DLQ:
  Type: AWS::SQS::Queue
  Properties:
    QueueName: !Sub '${AWS::StackName}-<kebab-name>-dlq'
```

**Key rules**:
- `VisibilityTimeout` must be ≥ consumer Lambda Timeout (default: 60s)
- Always define a DLQ with `maxReceiveCount` to prevent poison messages
- Consumer Lambda uses SQS Event source; producer Lambda uses `SQSSendMessagePolicy`

### SQS-Triggered Lambda Pattern

```yaml
<ConsumerFunction>:
  Type: AWS::Serverless::Function
  Properties:
    CodeUri: lambdas/<domain>/<function-dir>/
    Handler: handler.lambda_handler
    Policies:
      - SQSPollerPolicy:
          QueueName: !GetAtt <QueueLogicalName>.QueueName
    Events:
      SQSEvent:
        Type: SQS
        Properties:
          Queue: !GetAtt <QueueLogicalName>.Arn
          BatchSize: 10
```

### EventBridge Rule Pattern

```yaml
<RuleName>:
  Type: AWS::Events::Rule
  Properties:
    EventPattern:
      source:
        - "<event-source>"
      detail-type:
        - "<detail-type>"
    Targets:
      - Arn: !GetAtt <ConsumerFunction>.Arn
        Id: "<ConsumerName>"

<ConsumerName>EventPermission:
  Type: AWS::Lambda::Permission
  Properties:
    FunctionName: !Ref <ConsumerFunction>
    Action: lambda:InvokeFunction
    Principal: events.amazonaws.com
    SourceArn: !GetAtt <RuleName>.Arn
```

**Key rules**:
- Each consumer Lambda in a rule's Targets needs its own `AWS::Lambda::Permission`
- EventBridge-triggered Lambdas have NO `Events` block — the Rule Target handles invocation
- Producer Lambda needs `EventBridgePutEventsPolicy` and reads event bus from env var

## 3. Monitoring Patterns

### CloudWatch Alarms
```yaml
# Lambda Error Alarm
LambdaErrorAlarm:
  Type: AWS::CloudWatch::Alarm
  Properties:
    AlarmName: !Sub ${FunctionName}-errors
    AlarmDescription: Lambda function errors
    ComparisonOperator: GreaterThanThreshold
    EvaluationPeriods: 1
    MetricName: Errors
    Namespace: AWS/Lambda
    Period: 60
    Statistic: Sum
    Threshold: 0
    Dimensions:
      - Name: FunctionName
        Value: !Ref FunctionName
    AlarmActions:
      - !Ref AlarmTopic

# API Gateway Error Alarm
ApiErrorAlarm:
  Type: AWS::CloudWatch::Alarm
  Properties:
    AlarmName: !Sub ${ApiName}-5xx-errors
    ComparisonOperator: GreaterThanThreshold
    EvaluationPeriods: 2
    MetricName: 5XXError
    Namespace: AWS/ApiGateway
    Period: 60
    Statistic: Sum
    Threshold: 5
    Dimensions:
      - Name: ApiName
        Value: !Ref MyApi
    AlarmActions:
      - !Ref AlarmTopic
```

### CloudWatch Log Groups
```yaml
FunctionLogGroup:
  Type: AWS::Logs::LogGroup
  Properties:
    LogGroupName: !Sub /aws/lambda/${FunctionName}
    RetentionInDays: 30  # 7, 14, 30, 60, 90, 120, 180, 365, ...

ApiLogGroup:
  Type: AWS::Logs::LogGroup
  Properties:
    LogGroupName: !Sub /aws/apigateway/${ApiName}
    RetentionInDays: 30
```

### SNS Alarm Topic
```yaml
AlarmTopic:
  Type: AWS::SNS::Topic
  Properties:
    TopicName: !Sub ${Environment}-alarms
    Subscription:
      - Protocol: email
        Endpoint: alerts@example.com
```

## 4. Cognito Pattern

```yaml
UserPool:
  Type: AWS::Cognito::UserPool
  # ⚠️ Data Protection Policy (Strongly Recommended)
  # Prevents data loss when stack is deleted
  DeletionPolicy: Retain
  UpdateReplacePolicy: Retain
  Properties:
    UserPoolName: !Sub ${Environment}-users
    
    # Username Configuration (email as username)
    UsernameAttributes:
      - email
    
    # Auto Verification
    AutoVerifiedAttributes:
      - email
    
    # ⚠️ Schema is ONLY for custom attributes
    # Standard attributes (email, phone_number, name, etc.) should NOT be defined here
    # Custom attributes example (all custom attributes are optional by default):
    # Schema:
    #   - Name: department
    #     AttributeDataType: String
    #     Mutable: true
    #   - Name: employee_id
    #     AttributeDataType: String
    #     Mutable: false
    #   - Name: role
    #     AttributeDataType: String
    #     Mutable: true
    #
    # ⚠️ CRITICAL naming rule for Cognito custom attributes:
    # In UserPool.Schema, use the bare custom attribute name only
    # (for example, Name: role).
    # Do NOT include the custom: prefix in the schema (for example, do NOT write
    # Name: custom:role).
    # Cognito automatically exposes that attribute to AdminCreateUser and JWT claims
    # as custom:role.
    # Therefore, if Lambda code or test/bootstrap logic reads claims['custom:role']
    # or sets custom:role, the schema MUST declare Name: role so the schema name,
    # Cognito attribute name, and handler claim access all stay aligned.
    
    # Password Policy
    Policies:
      PasswordPolicy:
        MinimumLength: 8
        RequireUppercase: true
        RequireLowercase: true
        RequireNumbers: true
        RequireSymbols: false

UserPoolClient:
  Type: AWS::Cognito::UserPoolClient
  Properties:
    UserPoolId: !Ref UserPool
    ClientName: !Sub ${Environment}-client
    GenerateSecret: false
    ExplicitAuthFlows:
      - ALLOW_USER_PASSWORD_AUTH
      - ALLOW_REFRESH_TOKEN_AUTH
```

⚠️ **UserPoolClient: DO NOT add OAuth settings unless a Hosted UI domain is also provisioned**

The following properties require `AWS::Cognito::UserPoolDomain` to exist first.
Without the domain resource, CloudFormation `EarlyValidation::ResourceExistenceCheck` will **fail the deployment immediately**.

```yaml
# ❌ DO NOT add these unless you also create AWS::Cognito::UserPoolDomain:
# AllowedOAuthFlows: [code]
# AllowedOAuthFlowsUserPoolClient: true
# AllowedOAuthScopes: [email, openid, profile]
# CallbackURLs: [...]
# LogoutURLs: [...]
# SupportedIdentityProviders: [COGNITO]
```

For standard API authentication (Amplify Auth.signIn → API Gateway Cognito Authorizer), the minimal `ExplicitAuthFlows` config above is sufficient. No OAuth or Hosted UI is required.

### Cognito Triggers (Avoid Circular Dependencies)

**Recommended (SAM-native) pattern**: bind Cognito triggers from the Lambda side using `AWS::Serverless::Function` events.

Why:
- Avoids the circular dependency where API Gateway authorizers need the UserPool ARN, while the UserPool wants a trigger Lambda ARN.
- SAM automatically generates `AWS::Lambda::Permission` so Cognito can invoke the function (no missing invoke permission).

⚠️ Important:
- ❌ Do NOT set `UserPool.Properties.LambdaConfig` to reference a Lambda ARN when using a single template.
- ✅ Use SAM `Events: Cognito` instead.

```yaml
UserPool:
  Type: AWS::Cognito::UserPool
  Properties:
    UserPoolName: !Sub ${Environment}-users
    UsernameAttributes: [email]
    AutoVerifiedAttributes: [email]
    # ❌ Avoid LambdaConfig here to prevent circular dependency

PostConfirmationFunction:
  Type: AWS::Serverless::Function
  Properties:
    CodeUri: lambdas/PostConfirmationFunction/
    Handler: handler.handler
    Runtime: nodejs22.x
    Events:
      PostConfirmationTrigger:
        Type: Cognito
        Properties:
          UserPool: !Ref UserPool
          Trigger: PostConfirmation
```

## 5. SAM Build Behavior: Lambda Functions vs Layers

SAM build treats Lambda functions and Layers **differently** by default.
Understanding this distinction is critical for correct dependency packaging.

### Lambda Functions — Automatic Dependency Installation
SAM sees a `requirements.txt` (Python) or `package.json` (Node.js) in the
function's `CodeUri` directory and **automatically runs pip/npm install**,
bundling dependencies alongside source code into the deployment package.
No extra configuration is needed — this is the default behavior.

### Layers — No Automatic Build Unless `Metadata: BuildMethod` Is Present
By default, SAM copies the `ContentUri` directory **as-is** into the deployment
artifact. It does **not** run pip/npm install. To have SAM build a layer,
declare `Metadata: BuildMethod` on the layer resource:

```yaml
SharedUtilsLayer:
  Type: AWS::Serverless::LayerVersion
  Properties:
    ContentUri: layers/shared/
    CompatibleRuntimes:
      - python3.12
  Metadata:
    BuildMethod: python3.12
```

Without this `Metadata` section, `sam build` will just copy the files from
`ContentUri`, and any expected build/install step for the layer will be skipped.

### Critical Rule — `ContentUri` Must Point To The Runtime-Specific Layer Root
When a layer exists, `ContentUri` must point to the directory that is the
runtime-specific root of the layer:

- Python layer: `layers/shared/`
- Node.js layer: `layers/shared/nodejs/`

This rule is especially important for Node.js. If you incorrectly set
`ContentUri: layers/shared/` for a Node.js layer whose source already lives
under `layers/shared/nodejs/`, SAM can produce a broken double nesting such as
`nodejs/nodejs/...` in the build artifact. At runtime, Lambda mounts layers
under `/opt`, so Node.js code expects modules under `/opt/nodejs/...`. A double
nested artifact breaks module resolution and causes imports like
`require('/opt/nodejs/shared')` or layer-provided packages to fail.

### Correct Layer Patterns

#### Python Layer
Use the layer root as `ContentUri`, with code placed under `python/`:

```yaml
SharedUtilsLayer:
  Type: AWS::Serverless::LayerVersion
  Properties:
    ContentUri: layers/shared/
    CompatibleRuntimes:
      - python3.12
  Metadata:
    BuildMethod: python3.12
```

Directory layout:

```text
layers/shared/
├── python/
│   └── shared_utils.py
└── requirements.txt
```

#### Node.js Layer
Use the `nodejs/` directory itself as `ContentUri`:

```yaml
SharedUtilsLayer:
  Type: AWS::Serverless::LayerVersion
  Properties:
    ContentUri: layers/shared/nodejs/
    CompatibleRuntimes:
      - nodejs22.x
  Metadata:
    BuildMethod: nodejs22.x
```

Directory layout:

```text
layers/shared/
└── nodejs/
    ├── shared.js
    ├── package.json
    └── node_modules/
```

⚠️ For Node.js, do **not** point `ContentUri` at `layers/shared/` when the
actual layer content already lives in `layers/shared/nodejs/`. That is the
exact mistake that leads to `nodejs/nodejs/...` build artifacts.

### Decision Guide for Layers

| Scenario | BuildMethod | ContentUri | Directory layout |
|---|---|---|---|
| Python shared layer | `Metadata: BuildMethod: python3.12` | `layers/shared/` | code in `layers/shared/python/`; deps at `layers/shared/requirements.txt` |
| Node.js shared layer | `Metadata: BuildMethod: nodejs22.x` | `layers/shared/nodejs/` | code and `package.json` in `layers/shared/nodejs/` |
| Custom build logic needed | `Metadata: BuildMethod: makefile` | Layer root directory | Makefile copies artifacts into runtime-required subdir (`python/` or `nodejs/`) |

⚠️ **Key rule**: Lambda runtimes look for layer code in `/opt/python/` (Python)
or `/opt/nodejs/` (Node.js). Your build output must match this structure, or
imports will fail at runtime.

### Official Runtime Layout Reminder
AWS Lambda's official runtime expectations are:

- Python layer code under `python/`
- Node.js layer packages under `nodejs/node_modules/` (or versioned
  `nodeXX/node_modules/`)

For Node.js layers in this project, prefer the simpler shared-layer convention:
set `ContentUri` to the on-disk `nodejs/` directory so the built artifact maps
cleanly to `/opt/nodejs/...` without accidental extra nesting.

## 6. Output Definitions

```yaml
Outputs:
  # API URL
  ApiUrl:
    Description: API Gateway endpoint URL
    Value: !Sub "https://${MyApi}.execute-api.${AWS::Region}.amazonaws.com/${Environment}"
    Export:
      Name: !Sub ${AWS::StackName}-ApiUrl
  
  # Table Name
  TableName:
    Description: DynamoDB table name
    Value: !Ref MyTable
    Export:
      Name: !Sub ${AWS::StackName}-TableName
  
  # S3 Bucket Name
  BucketName:
    Description: S3 bucket name
    Value: !Ref MyBucket
    Export:
      Name: !Sub ${AWS::StackName}-BucketName
  
  # UserPool ID
  UserPoolId:
    Description: Cognito User Pool ID
    Value: !Ref UserPool
    Export:
      Name: !Sub ${AWS::StackName}-UserPoolId
```

## 7. SAM Policy Templates Reference

### Most Commonly Used Policy Templates

#### Database
- `DynamoDBCrudPolicy` - Full DynamoDB operations (GetItem, PutItem, UpdateItem, DeleteItem, Query, Scan)
  ```yaml
  - DynamoDBCrudPolicy:
      TableName: !Ref MyTable
  ```
- `DynamoDBReadPolicy` - DynamoDB read-only operations (GetItem, Query, Scan)
  ```yaml
  - DynamoDBReadPolicy:
      TableName: !Ref MyTable
  ```

#### Storage
- `S3CrudPolicy` - Full S3 operations (GetObject, PutObject, DeleteObject, ListBucket)
  ```yaml
  - S3CrudPolicy:
      BucketName: !Ref MyBucket
  ```
- `S3ReadPolicy` - S3 read-only operations (GetObject, ListBucket)
  ```yaml
  - S3ReadPolicy:
      BucketName: !Ref MyBucket
  ```
- `S3WritePolicy` - S3 write operations (PutObject, DeleteObject)
  ```yaml
  - S3WritePolicy:
      BucketName: !Ref MyBucket
  ```

#### Compute
- `LambdaInvokePolicy` - Invoke other Lambda functions (synchronous)
  ```yaml
  - LambdaInvokePolicy:
      FunctionName: !Ref AnotherFunction
  ```

#### Messaging (SQS / EventBridge)
- `SQSSendMessagePolicy` - Send messages to an SQS queue (for producer Lambdas)
  ```yaml
  - SQSSendMessagePolicy:
      QueueName: !GetAtt MyQueue.QueueName
  ```
- `SQSPollerPolicy` - Poll and consume messages from an SQS queue (for consumer Lambdas)
  ```yaml
  - SQSPollerPolicy:
      QueueName: !GetAtt MyQueue.QueueName
  ```
- `EventBridgePutEventsPolicy` - Publish events to EventBridge (for producer Lambdas)
  ```yaml
  - EventBridgePutEventsPolicy:
      EventBusName: default
  ```

#### Security
- `AWSSecretsManagerGetSecretValuePolicy` - Get Secrets Manager secrets
- `SSMParameterReadPolicy` - Read SSM parameters

#### AI/ML (Use inline policies)
```yaml
Policies:
  - Version: '2012-10-17'
    Statement:
      # Comprehend
      - Effect: Allow
        Action:
          - comprehend:DetectSentiment
          - comprehend:DetectEntities
        Resource: '*'
      
      # Rekognition
      - Effect: Allow
        Action:
          - rekognition:DetectLabels
          - rekognition:DetectText
        Resource: '*'
      
      # Textract
      - Effect: Allow
        Action:
          - textract:AnalyzeDocument
          - textract:DetectDocumentText
        Resource: '*'
```

## 8. Complete Example

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: Serverless REST API with monitoring

Parameters:
  Environment:
    Type: String
    Default: dev
    AllowedValues: [dev, staging, prod]

Globals:
  Function:
    Runtime: python3.12
    Handler: index.handler
    Timeout: 30
    MemorySize: 512
    Tracing: Active
    Environment:
      Variables:
        ENVIRONMENT: !Ref Environment
        LOG_LEVEL: INFO
    Tags:
      Environment: !Ref Environment

Resources:
  # API
  MyApi:
    Type: AWS::Serverless::Api
    Properties:
      StageName: !Ref Environment
      TracingEnabled: true
      Cors:
        AllowOrigin: "'*'"
        AllowHeaders: "'*'"
        AllowMethods: "'*'"
      AccessLogSetting:
        DestinationArn: !GetAtt ApiLogGroup.Arn
        Format: '$context.requestId'
  
  ApiLogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub /aws/apigateway/${MyApi}
      RetentionInDays: 30
  
  # Lambda Function
  ApiFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: backend/api/
      Handler: app.handler
      Environment:
        Variables:
          TABLE_NAME: !Ref DataTable
      Policies:
        - DynamoDBCrudPolicy:
            TableName: !Ref DataTable
      Events:
        GetApi:
          Type: Api
          Properties:
            Path: /items
            Method: GET
            RestApiId: !Ref MyApi
  
  # DynamoDB Table
  DataTable:
    Type: AWS::DynamoDB::Table
    # ⚠️ Data Protection: Prevents data loss on stack deletion
    DeletionPolicy: Retain
    UpdateReplacePolicy: Retain
    Properties:
      TableName: !Sub ${Environment}-data
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: id
          AttributeType: S
      KeySchema:
        - AttributeName: id
          KeyType: HASH
      SSESpecification:
        SSEEnabled: true
      PointInTimeRecoverySpecification:
        PointInTimeRecoveryEnabled: true
  
  # CloudWatch Alarm
  ApiErrorAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: !Sub ${Environment}-api-errors
      ComparisonOperator: GreaterThanThreshold
      EvaluationPeriods: 1
      MetricName: 5XXError
      Namespace: AWS/ApiGateway
      Period: 60
      Statistic: Sum
      Threshold: 5
      Dimensions:
        - Name: ApiName
          Value: !Ref MyApi
      AlarmActions:
        - !Ref AlarmTopic
  
  AlarmTopic:
    Type: AWS::SNS::Topic
    Properties:
      Subscription:
        - Protocol: email
          Endpoint: alerts@example.com

Outputs:
  ApiUrl:
    Value: !Sub "https://${MyApi}.execute-api.${AWS::Region}.amazonaws.com/${Environment}"
  TableName:
    Value: !Ref DataTable
```

### Most Commonly Used Policy Templates (By actual usage frequency)

```yaml
Policies:
  # DynamoDB Access (Most Common)
  - DynamoDBCrudPolicy:
      TableName: !Ref MyTable
  
  # S3 Access
  - S3CrudPolicy:
      BucketName: !Ref MyBucket
  
  # Lambda Invocation (synchronous)
  - LambdaInvokePolicy:
      FunctionName: !Ref AnotherFunction
  
  # SQS — producer sends messages
  - SQSSendMessagePolicy:
      QueueName: !GetAtt MyQueue.QueueName
  
  # SQS — consumer polls messages
  - SQSPollerPolicy:
      QueueName: !GetAtt MyQueue.QueueName
  
  # EventBridge — producer publishes events
  - EventBridgePutEventsPolicy:
      EventBusName: default
```

### Policy Decision Tree

```
Need to access AWS resources?
├─ Yes → Use SAM Policy Template (Recommended)
│   ├─ DynamoDB → DynamoDBCrudPolicy
│   ├─ S3 → S3CrudPolicy
│   ├─ Lambda invoke → LambdaInvokePolicy
│   ├─ SQS send → SQSSendMessagePolicy
│   ├─ SQS consume → SQSPollerPolicy
│   ├─ EventBridge publish → EventBridgePutEventsPolicy
│   └─ Others → Check list above
│
├─ Need custom permissions? → Inline IAM Policy
│   ```yaml
│   Policies:
│     - Version: '2012-10-17'
│       Statement:
│         - Effect: Allow
│           Action: ['dynamodb:GetItem']
│           Resource: !GetAtt Table.Arn
│   ```
│
└─ Need to reuse across multiple Lambdas? → Create separate IAM Role
    ```yaml
    SharedRole:
      Type: AWS::IAM::Role
      Properties:
        AssumeRolePolicyDocument: ...
        ManagedPolicyArns:
          - arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess
    
    MyFunction:
      Type: AWS::Serverless::Function
      Properties:
        Role: !GetAtt SharedRole.Arn
    ```
```

---
