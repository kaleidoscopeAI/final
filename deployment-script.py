#!/bin/bash

# Kaleidoscope AI Deployment Script
# This script automates the deployment of the entire system to AWS

set -e

# Configuration
STACK_NAME="kaleidoscope-ai"
REGION="us-east-1"
S3_BUCKET="${STACK_NAME}-artifacts-$(date +%s)"
DYNAMODB_TABLE="${STACK_NAME}-metrics"
LOG_GROUP="/kaleidoscope/system"

echo "Starting Kaleidoscope AI deployment..."

# Create S3 bucket for artifacts
aws s3api create-bucket \
    --bucket "$S3_BUCKET" \
    --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION"

# Enable versioning and encryption
aws s3api put-bucket-versioning \
    --bucket "$S3_BUCKET" \
    --versioning-configuration Status=Enabled

aws s3api put-bucket-encryption \
    --bucket "$S3_BUCKET" \
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "AES256"
            }
        }]
    }'

echo "Created S3 bucket: $S3_BUCKET"

# Create DynamoDB table
aws dynamodb create-table \
    --table-name "$DYNAMODB_TABLE" \
    --attribute-definitions AttributeName=node_id,AttributeType=S \
    --key-schema AttributeName=node_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --stream-specification StreamEnabled=true,StreamViewType=NEW_AND_OLD_IMAGES \
    --region "$REGION"

echo "Created DynamoDB table: $DYNAMODB_TABLE"

# Create CloudWatch Log Group
aws logs create-log-group --log-group-name "$LOG_GROUP"

echo "Created CloudWatch Log Group: $LOG_GROUP"

# Package and deploy Lambda functions
LAMBDA_BUCKET="${S3_BUCKET}-lambda"

# Create bucket for Lambda code
aws s3api create-bucket \
    --bucket "$LAMBDA_BUCKET" \
    --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION"

# Package Lambda functions
cd lambda
zip -r ../recovery.zip recovery_function.py
cd ..

# Upload Lambda package
aws s3 cp recovery.zip "s3://${LAMBDA_BUCKET}/recovery.zip"

# Create Lambda function
aws lambda create-function \
    --function-name "${STACK_NAME}-recovery" \
    --runtime python3.9 \
    --handler recovery_function.lambda_handler \
    --role "arn:aws:iam::${AWS_ACCOUNT_ID}:role/KaleidoscopeExecutionRole" \
    --code "S3Bucket=${LAMBDA_BUCKET},S3Key=recovery.zip" \
    --timeout 300 \
    --memory-size 1024 \
    --environment "Variables={
        DYNAMODB_TABLE=${DYNAMODB_TABLE},
        S3_BUCKET=${S3_BUCKET},
        LOG_GROUP=${LOG_GROUP}
    }"

echo "Deployed Lambda functions"

# Deploy CloudFormation stack
aws cloudformation deploy \
    --template-file infrastructure/main.yaml \
    --stack-name "$STACK_NAME" \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameter-overrides \
        S3BucketName="$S3_BUCKET" \
        DynamoDBTableName="$DYNAMODB_TABLE" \
        LogGroupName="$LOG_GROUP"

echo "Deployed CloudFormation stack"

# Upload core system code
aws s3 cp kaleidoscope_core.py "s3://${S3_BUCKET}/core/kaleidoscope_core.py"

echo "Uploaded system code"

# Create CloudWatch Dashboard
aws cloudwatch put-dashboard \
    --dashboard-name "${STACK_NAME}-dashboard" \
    --dashboard-body file://monitoring/dashboard.json

echo "Created CloudWatch Dashboard"

# Final verification
echo "Verifying deployment..."

# Check S3 bucket
aws s3 ls "s3://${S3_BUCKET}/core/kaleidoscope_core.py"

# Check DynamoDB table
aws dynamodb describe-table --table-name "$DYNAMODB_TABLE"

# Check Lambda function
aws lambda get-function --function-name "${STACK_NAME}-recovery"

echo "Deployment complete! System is ready."
echo "Dashboard URL: https://${REGION}.console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards:name=${STACK_NAME}-dashboard"
