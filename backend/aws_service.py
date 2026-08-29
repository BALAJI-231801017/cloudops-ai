"""AWS integration that supports IAM roles and local credential profiles."""

from datetime import datetime, timedelta, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError, ProfileNotFound

from backend.config import Settings


class AwsServiceError(RuntimeError):
    """Raised when AWS cannot service a request."""


class AwsService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _session(self):
        session_args = {"region_name": self.settings.aws_region}
        if self.settings.aws_profile:
            session_args["profile_name"] = self.settings.aws_profile
        return boto3.Session(**session_args)

    def _client(self, service_name: str):
        try:
            return self._session().client(service_name)
        except ProfileNotFound as error:
            raise AwsServiceError(
                "Configured AWS profile was not found. Set CLOUDOPS_AWS_PROFILE "
                "to an available local profile, or unset it to use IAM role/default credentials."
            ) from error
        except BotoCoreError as error:
            raise AwsServiceError(f"Unable to create AWS {service_name} client: {error}") from error

    def list_instances(self):
        try:
            response = self._client("ec2").describe_instances()
        except (BotoCoreError, ClientError) as error:
            raise AwsServiceError(f"Unable to discover EC2 instances: {error}") from error

        instances = []
        for reservation in response.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                tags = {tag.get("Key"): tag.get("Value") for tag in instance.get("Tags", [])}
                instances.append(
                    {
                        "id": instance.get("InstanceId"),
                        "name": tags.get("Name", "Unnamed"),
                        "state": instance.get("State", {}).get("Name", "unknown"),
                        "type": instance.get("InstanceType"),
                        "private_ip": instance.get("PrivateIpAddress"),
                        "public_ip": instance.get("PublicIpAddress"),
                        "availability_zone": instance.get("Placement", {}).get("AvailabilityZone"),
                    }
                )
        return instances

    def get_instance_metrics(self, instance_id: str, hours: int = 1):
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        cloudwatch = self._client("cloudwatch")
        metric_names = ("CPUUtilization", "NetworkIn", "NetworkOut")
        metrics = {}

        try:
            for metric_name in metric_names:
                response = cloudwatch.get_metric_statistics(
                    Namespace="AWS/EC2",
                    MetricName=metric_name,
                    Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=300,
                    Statistics=["Average"],
                )
                datapoints = sorted(response.get("Datapoints", []), key=lambda point: point["Timestamp"])
                metrics[metric_name] = [
                    {
                        "timestamp": point["Timestamp"].astimezone(timezone.utc).isoformat(),
                        "average": round(point["Average"], 2),
                        "unit": point.get("Unit"),
                    }
                    for point in datapoints
                ]
        except (BotoCoreError, ClientError) as error:
            raise AwsServiceError(f"Unable to retrieve CloudWatch metrics: {error}") from error

        return {
            "instance_id": instance_id,
            "region": self.settings.aws_region,
            "window_hours": hours,
            "metrics": metrics,
        }
