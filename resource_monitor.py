"""
Kaleidoscope AI - Resource Monitor and Cleanup
--------------------------------------------
Handles resource monitoring, health checks, and infrastructure cleanup.
"""

import boto3
import json
import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
from dataclasses import dataclass

@dataclass
class ResourceMetrics:
    """Container for resource metrics data."""
    cpu_utilization: float
    memory_utilization: float
    disk_usage: float
    network_in: float
    network_out: float
    timestamp: datetime

class ResourceMonitor:
    """Monitors and reports on Kaleidoscope AI infrastructure health."""
    
    def __init__(self, region: str, config: Dict, logger: logging.Logger):
        self.region = region
        self.config = config
        self.logger = logger
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.ec2 = boto3.client('ec2', region_name=region)

    def get_resource_metrics(self, instance_id: str, start_time: datetime, end_time: datetime) -> Dict[str, List[ResourceMetrics]]:
        """Get comprehensive metrics for specified instance."""
        try:
            metrics = {}
            period = 300  # 5-minute intervals

            # Get CPU utilization
            cpu_stats = self._get_metric_statistics(
                namespace="AWS/EC2",
                metric_name="CPUUtilization",
                dimension_name="InstanceId",
                dimension_value=instance_id,
                start_time=start_time,
                end_time=end_time,
                period=period
            )

            # Get memory utilization
            memory_stats = self._get_metric_statistics(
                namespace=self.config['cloudwatch']['metrics_namespace'],
                metric_name="mem_used_percent",
                dimension_name="InstanceId",
                dimension_value=instance_id,
                start_time=start_time,
                end_time=end_time,
                period=period
            )

            # Get disk usage
            disk_stats = self._get_metric_statistics(
                namespace=self.config['cloudwatch']['metrics_namespace'],
                metric_name="disk_used_percent",
                dimension_name="InstanceId",
                dimension_value=instance_id,
                start_time=start_time,
                end_time=end_time,
                period=period
            )

            # Get network metrics
            network_in = self._get_metric_statistics(
                namespace="AWS/EC2",
                metric_name="NetworkIn",
                dimension_name="InstanceId",
                dimension_value=instance_id,
                start_time=start_time,
                end_time=end_time,
                period=period
            )

            network_out = self._get_metric_statistics(
                namespace="AWS/EC2",
                metric_name="NetworkOut",
                dimension_name="InstanceId",
                dimension_value=instance_id,
                start_time=start_time,
                end_time=end_time,
                period=period
            )

            # Combine metrics into time series
            timestamps = sorted(set(
                point['Timestamp'] for metric in [cpu_stats, memory_stats, disk_stats]
                for point in metric
            ))

            metrics_series = []
            for timestamp in timestamps:
                cpu_point = next((p for p in cpu_stats if p['Timestamp'] == timestamp), {'Average': 0})
                memory_point = next((p for p in memory_stats if p['Timestamp'] == timestamp), {'Average': 0})
                disk_point = next((p for p in disk_stats if p['Timestamp'] == timestamp), {'Average': 0})
                network_in_point = next((p for p in network_in if p['Timestamp'] == timestamp), {'Average': 0})
                network_out_point = next((p for p in network_out if p['Timestamp'] == timestamp), {'Average': 0})

                metrics_series.append(ResourceMetrics(
                    cpu_utilization=cpu_point['Average'],
                    memory_utilization=memory_point['Average'],
                    disk_usage=disk_point['Average'],
                    network_in=network_in_point['Average'],
                    network_out=network_out_point['Average'],
                    timestamp=timestamp
                ))

            return metrics_series

        except Exception as e:
            self.logger.error(f"Failed to get resource metrics: {e}")
            raise

    def _get_metric_statistics(self, namespace: str, metric_name: str,
                             dimension_name: str, dimension_value: str,
                             start_time: datetime, end_time: datetime,
                             period: int) -> List[Dict]:
        """Helper method to get metric statistics from CloudWatch."""
        try:
            response = self.cloudwatch.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric_name,
                Dimensions=[{'Name': dimension_name, 'Value': dimension_value}],
                StartTime=start_time,
                EndTime=end_time,
                Period=period,
                Statistics=['Average']
            )
            return response['Datapoints']
        except Exception as e:
            self.logger.error(f"Failed to get metric statistics for {metric_name}: {e}")
            raise

    def check_instance_health(self, instance_id: str) -> Tuple[bool, Dict]:
        """Check instance health and return status with details."""
        try:
            # Get instance status
            response = self.ec2.describe_instance_status(
                InstanceIds=[instance_id],
                IncludeAllInstances=True
            )

            if not response['InstanceStatuses']:
                return False, {"error": "Instance not found"}

            status = response['InstanceStatuses'][0]
            
            # Get recent metrics
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=15)
            recent_metrics = self.get_resource_metrics(instance_id, start_time, end_time)
            
            if not recent_metrics:
                return False, {"error": "No recent metrics available"}

            latest_metrics = recent_metrics[-1]

            # Check against thresholds
            cpu_threshold = self.config['cloudwatch']['alarms']['cpu_utilization']['threshold']
            memory_threshold = self.config['cloudwatch']['alarms']['memory_utilization']['threshold']
            disk_threshold = self.config['cloudwatch']['alarms']['disk_usage']['threshold']

            health_status = {
                "instance_status": status['InstanceStatus']['Status'],
                "system_status": status['SystemStatus']['Status'],
                "metrics": {
                    "cpu_utilization": latest_metrics.cpu_utilization,
                    "memory_utilization": latest_metrics.memory_utilization,
                    "disk_usage": latest_metrics.disk_usage,
                    "network_in": latest_metrics.network_in,
                    "network_out": latest_metrics.network_out
                },
                "thresholds_exceeded": {
                    "cpu": latest_metrics.cpu_utilization > cpu_threshold,
                    "memory": latest_metrics.memory_utilization > memory_threshold,
                    "disk": latest_metrics.disk_usage > disk_threshold
                }
            }

            # Determine overall health
            is_healthy = (
                status['InstanceStatus']['Status'] == 'ok' and
                status['SystemStatus']['Status'] == 'ok' and
                not any(health_status['thresholds_exceeded'].values())
            )

            return is_healthy, health_status

        except Exception as e:
            self.logger.error(f"Failed to check instance health: {e}")
            raise

class ResourceCleanup:
    """Manages cleanup of Kaleidoscope AI infrastructure resources."""
    
    def __init__(self, region: str, config: Dict, logger: logging.Logger):
        self.region = region
        self.config = config
        self.logger = logger
        self.ec2 = boto3.client('ec2', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.backup = boto3.client('backup', region_name=region)

    def cleanup_instance(self, instance_id: str, force: bool = False) -> None:
        """Clean up EC2 instance and related resources."""
        try:
            # Get instance details
            response = self.ec2.describe_instances(InstanceIds=[instance_id])
            if not response['Reservations']:
                self.logger.warning(f"Instance {instance_id} not found")
                return

            instance = response['Reservations'][0]['Instances'][0]
            
            # Check if instance can be terminated
            if not force:
                tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                if tags.get('Protected') == 'true':
                    raise ValueError(f"Instance {instance_id} is protected from termination")

            # Delete CloudWatch alarms
            self._delete_instance_alarms(instance_id)

            # Terminate instance
            self.ec2.terminate_instances(InstanceIds=[instance_id])
            
            # Wait for termination
            waiter = self.ec2.get_waiter('instance_terminated')
            waiter.wait(
                InstanceIds=[instance_id],
                WaiterConfig={'Delay': 15, 'MaxAttempts': 40}
            )

            # Cleanup security groups
            for sg in instance['SecurityGroups']:
                try:
                    self.ec2.delete_security_group(GroupId=sg['GroupId'])
                    self.logger.info(f"Deleted security group {sg['GroupId']}")
                except ClientError as e:
                    if e.response['Error']['Code'] != 'DependencyViolation':
                        raise

            self.logger.info(f"Successfully cleaned up instance {instance_id}")

        except Exception as e:
            self.logger.error(f"Failed to cleanup instance {instance_id}: {e}")
            raise

    def cleanup_bucket(self, bucket_name: str, force: bool = False) -> None:
        """Clean up S3 bucket and its contents."""
        try:
            # Check if bucket exists
            try:
                self.s3.head_bucket(Bucket=bucket_name)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    self.logger.warning(f"Bucket {bucket_name} not found")
                    return
                raise

            if not force:
                # Check bucket versioning
                versioning = self.s3.get_bucket_versioning(Bucket=bucket_name)
                if versioning.get('Status') == 'Enabled':
                    self.logger.warning(f"Bucket {bucket_name} has versioning enabled. Use force=True to delete.")
                    return

            # Delete all objects, including versions
            paginator = self.s3.get_paginator('list_object_versions')
            for page in paginator.paginate(Bucket=bucket_name):
                objects = []
                
                # Handle current versions
                for version in page.get('Versions', []):
                    objects.append({
                        'Key': version['Key'],
                        'VersionId': version['VersionId']
                    })
                
                # Handle delete markers
                for marker in page.get('DeleteMarkers', []):
                    objects.append({
                        'Key': marker['Key'],
                        'VersionId': marker['VersionId']
                    })
                
                if objects:
                    self.s3.delete_objects(
                        Bucket=bucket_name,
                        Delete={'Objects': objects}
                    )

            # Delete bucket
            self.s3.delete_bucket(Bucket=bucket_name)
            self.logger.info(f"Successfully cleaned up bucket {bucket_name}")

        except Exception as e:
            self.logger.error(f"Failed to cleanup bucket {bucket_name}: {e}")
            raise

    def _delete_instance_alarms(self, instance_id: str) -> None:
        """Delete CloudWatch alarms associated with an instance."""
        try:
            # List alarms for instance
            paginator = self.cloudwatch.get_paginator('describe_alarms')
            for page in paginator.paginate():
                for alarm in page['MetricAlarms']:
                    for dimension in alarm['Dimensions']:
                        if dimension['Name'] == 'InstanceId' and dimension['Value'] == instance_id:
                            self.cloudwatch.delete_alarms(AlarmNames=[alarm['AlarmName']])
                            self.logger.info(f"Deleted alarm {alarm['AlarmName']}")

        except Exception as e:
            self.logger.error(f"Failed to delete instance alarms: {e}")
            raise

    def cleanup_backup_resources(self, force: bool = False) -> None:
        """Clean up AWS Backup resources."""
        try:
            vault_name = f"KaleidoscopeAI-{self.config['environment']}"

            # Delete backup selection and plan
            paginator = self.backup.get_paginator('list_backup_plans')
            for page in paginator.paginate():
                for plan in page['BackupPlansList']:
                    if 'KaleidoscopeAI' in plan['BackupPlanName']:
                        # Delete selections
                        selections = self.backup.list_backup_selections(
                            BackupPlanId=plan['BackupPlanId']
                        )
                        for selection in selections['BackupSelectionsList']:
                            self.backup.delete_backup_selection(
                                BackupPlanId=plan['BackupPlanId'],
                                SelectionId=selection['SelectionId']
                            )

                        # Delete plan
                        self.backup.delete_backup_plan(
                            BackupPlanId=plan['BackupPlanId']
                        )

            # Delete backup vault (only if force=True)
            if force:
                try:
                    self.backup.delete_backup_vault(
                        BackupVaultName=vault_name
                    )
                except ClientError as e:
                    if e.response['Error']['Code'] != 'ResourceNotFoundException':
                        raise

            self.logger.info("Successfully cleaned up backup resources")

        except Exception as e:
            self.logger.error(f"Failed to cleanup backup resources: {e}")
            raise

def create_resource_report(metrics: List[ResourceMetrics], health_status: Dict) -> Dict:
    """Create a comprehensive resource utilization report."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "health_status": health_status,
        "metrics_summary": {
            "cpu_utilization": {
                "average": sum(m.cpu_utilization for m in metrics) / len(metrics),
                "max": max(m.cpu_utilization for m in metrics),
                "min": min(m.cpu_utilization for m in metrics)
            },
            "memory_utilization": {
                "average": sum(m.memory_utilization for m in metrics) / len(metrics),
                "max": max(m.memory_utilization for m in metrics),
                "min": min(m.memory_utilization for m in metrics)
            },
            "disk_usage": {
                "average": sum(m.disk_usage for m in metrics) / len(metrics),
                "max": max(m.disk_usage for m in metrics),
                "min": min(m.disk_usage for m in metrics)
            },
            "network":
