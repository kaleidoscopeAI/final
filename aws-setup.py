import boto3
import logging
import argparse
import time
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, WaiterError
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def configure_logging(verbosity: bool) -> None:
    """Configure logging level based on verbosity."""
    if verbosity:
        logging.getLogger().setLevel(logging.DEBUG)
    # Add logging handler for file output
    file_handler = logging.FileHandler('kaleidoscope_setup.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description="AWS Environment Setup for Kaleidoscope AI")
    parser.add_argument("--bucket-name", type=str, required=True,
                      help="Name of the S3 bucket (must be globally unique)")
    parser.add_argument("--region", type=str, default="us-east-1",
                      help="AWS region to deploy resources")
    parser.add_argument("--instance-type", type=str, default="t3.medium",
                      help="Type of EC2 instance")
    parser.add_argument("--ami-id", type=str, required=True,
                      help="AMI ID for launching EC2 instance")
    parser.add_argument("--key-pair", type=str, required=True,
                      help="Name of the key pair for SSH access")
    parser.add_argument("--vpc-id", type=str, required=True,
                      help="VPC ID for the EC2 instance")
    parser.add_argument("--subnet-id", type=str, required=True,
                      help="Subnet ID for the EC2 instance")
    parser.add_argument("--tags", type=str, default="{}",
                      help="JSON string of tags to apply to resources")
    parser.add_argument("--verbosity", action="store_true",
                      help="Enable verbose output")
    return parser.parse_args()

def get_aws_client(service: str, region: str):
    """Create an AWS service client with proper error handling."""
    try:
        return boto3.client(service, region_name=region)
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error("AWS credentials not found or incomplete: %s", e)
        raise
    except Exception as e:
        logging.error("Error creating %s client: %s", service, e)
        raise

def create_security_group(ec2, vpc_id: str, instance_name: str) -> str:
    """Create a security group with minimal required access."""
    try:
        response = ec2.create_security_group(
            GroupName=f"{instance_name}-sg",
            Description="Security group for Kaleidoscope AI instance",
            VpcId=vpc_id
        )
        security_group_id = response['GroupId']
        
        # Configure security group rules
        ec2.authorize_security_group_ingress(
            GroupId=security_group_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'SSH access'}]
                },
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 443,
                    'ToPort': 443,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'HTTPS access'}]
                }
            ]
        )
        
        logging.info("Created security group: %s", security_group_id)
        return security_group_id
    except Exception as e:
        logging.error("Error creating security group: %s", e)
        raise

def configure_s3_bucket(s3, bucket_name: str, region: str) -> None:
    """Configure S3 bucket with security settings and encryption."""
    try:
        # Enable versioning
        s3.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={'Status': 'Enabled'}
        )
        
        # Enable encryption
        s3.put_bucket_encryption(
            Bucket=bucket_name,
            ServerSideEncryptionConfiguration={
                'Rules': [
                    {
                        'ApplyServerSideEncryptionByDefault': {
                            'SSEAlgorithm': 'AES256'
                        }
                    }
                ]
            }
        )
        
        # Block public access
        s3.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                'BlockPublicAcls': True,
                'IgnorePublicAcls': True,
                'BlockPublicPolicy': True,
                'RestrictPublicBuckets': True
            }
        )
        
        logging.info("Configured security settings for bucket '%s'", bucket_name)
    except Exception as e:
        logging.error("Error configuring bucket '%s': %s", bucket_name, e)
        raise

def wait_for_instance(ec2, instance_id: str) -> None:
    """Wait for EC2 instance to be running and pass status checks."""
    try:
        waiter = ec2.get_waiter('instance_running')
        waiter.wait(
            InstanceIds=[instance_id],
            WaiterConfig={'Delay': 10, 'MaxAttempts': 30}
        )
        
        waiter = ec2.get_waiter('instance_status_ok')
        waiter.wait(
            InstanceIds=[instance_id],
            WaiterConfig={'Delay': 10, 'MaxAttempts': 30}
        )
        
        logging.info("Instance %s is running and passed status checks", instance_id)
    except WaiterError as e:
        logging.error("Timeout waiting for instance %s: %s", instance_id, e)
        raise
    except Exception as e:
        logging.error("Error waiting for instance %s: %s", instance_id, e)
        raise

def create_bucket_if_needed(s3, bucket_name: str, region: str) -> None:
    """Create and configure S3 bucket with proper error handling."""
    try:
        s3.head_bucket(Bucket=bucket_name)
        logging.info("Bucket '%s' already exists", bucket_name)
    except s3.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        if error_code == '404':
            try:
                if region == 'us-east-1':
                    s3.create_bucket(Bucket=bucket_name)
                else:
                    s3.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                logging.info("Created bucket '%s' in region '%s'", bucket_name, region)
                
                # Configure bucket security and encryption
                configure_s3_bucket(s3, bucket_name, region)
            except Exception as create_error:
                logging.error("Error creating bucket '%s': %s", bucket_name, create_error)
                raise
        else:
            logging.error("Error checking bucket '%s': %s", bucket_name, e)
            raise

def launch_instance_if_needed(
    ec2,
    instance_type: str,
    ami_id: str,
    key_pair: str,
    instance_name: str,
    subnet_id: str,
    security_group_id: str,
    tags: dict
) -> Optional[str]:
    """Launch EC2 instance with security group and proper monitoring."""
    try:
        instances = ec2.describe_instances(
            Filters=[
                {'Name': 'tag:Name', 'Values': [instance_name]},
                {'Name': 'instance-state-name', 'Values': ['pending', 'running']}
            ]
        )
        
        if any(reservation['Instances'] for reservation in instances.get('Reservations', [])):
            logging.info("Instance '%s' already exists", instance_name)
            return None
        
        # Prepare tags
        tag_specifications = [
            {
                'ResourceType': 'instance',
                'Tags': [{'Key': k, 'Value': v} for k, v in tags.items()] + [
                    {'Key': 'Name', 'Value': instance_name},
                    {'Key': 'Environment', 'Value': 'Production'},
                    {'Key': 'Application', 'Value': 'KaleidoscopeAI'}
                ]
            }
        ]
        
        # Launch instance with detailed monitoring
        response = ec2.run_instances(
            ImageId=ami_id,
            InstanceType=instance_type,
            KeyName=key_pair,
            SubnetId=subnet_id,
            SecurityGroupIds=[security_group_id],
            MinCount=1,
            MaxCount=1,
            Monitoring={'Enabled': True},
            TagSpecifications=tag_specifications,
            MetadataOptions={
                'HttpTokens': 'required',  # Require IMDSv2
                'HttpPutResponseHopLimit': 1
            }
        )
        
        instance_id = response['Instances'][0]['InstanceId']
        logging.info("Launched instance '%s' with ID: %s", instance_name, instance_id)
        
        # Wait for instance to be ready
        wait_for_instance(ec2, instance_id)
        return instance_id
    
    except Exception as e:
        logging.error("Error launching instance '%s': %s", instance_name, e)
        raise

def main():
    """Main execution function with proper error handling and cleanup."""
    args = parse_arguments()
    configure_logging(args.verbosity)
    
    try:
        # Create AWS clients
        s3 = get_aws_client('s3', args.region)
        ec2 = get_aws_client('ec2', args.region)
        
        # Set up S3 bucket
        create_bucket_if_needed(s3, args.bucket_name, args.region)
        
        # Create security group
        instance_name = "KaleidoscopeAIInstance"
        security_group_id = create_security_group(ec2, args.vpc_id, instance_name)
        
        # Launch EC2 instance
        instance_id = launch_instance_if_needed(
            ec2,
            args.instance_type,
            args.ami_id,
            args.key_pair,
            instance_name,
            args.subnet_id,
            security_group_id,
            eval(args.tags)
        )
        
        if instance_id:
            logging.info("Successfully set up Kaleidoscope AI environment")
            
    except Exception as e:
        logging.error("Failed to set up environment: %s", e)
        raise

if __name__ == "__main__":
    main()
