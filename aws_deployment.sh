
#!/bin/bash
# Variables (customize these)
CLUSTER_NAME="kaleidoscope-cluster"
SERVICE_NAME="kaleidoscope-backend-service"
TASK_DEFINITION_FILE="task-definition.json"
REGION="us-east-1"

# Create ECS cluster
aws ecs create-cluster --cluster-name $CLUSTER_NAME --region $REGION

# Register task definition
aws ecs register-task-definition --cli-input-json file://$TASK_DEFINITION_FILE --region $REGION

# Create service with desired count
aws ecs create-service \
  --cluster $CLUSTER_NAME \
  --service-name $SERVICE_NAME \
  --desired-count 2 \
  --task-definition kaleidoscope-backend \
  --region $REGION
