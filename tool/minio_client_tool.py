import json

from minio import Minio

from config.config import MinIoConfig
from tool.logger import logger

minio_client = None


def get_minio_client():
    global minio_client
    if not minio_client:
        try:
            # 创建客户端
            client = Minio(
                endpoint=MinIoConfig.minio_endpoint,
                access_key=MinIoConfig.minio_access_key,
                secret_key=MinIoConfig.minio_secret_key,
                secure=False,
            )

            # 创建桶
            bucket_name = MinIoConfig.minio_bucket_name
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)

            # 设置权限
            policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"AWS": "*"},
                        "Action": ["s3:GetBucketLocation", "s3:ListBucket"],
                        "Resource": f"arn:aws:s3:::{bucket_name}",
                    },
                    {
                        "Effect": "Allow",
                        "Principal": {"AWS": "*"},
                        "Action": "s3:GetObject",
                        "Resource": f"arn:aws:s3:::{bucket_name}/*",
                    },
                ],
            }
            client.set_bucket_policy(bucket_name, json.dumps(policy))
            minio_client = client
        except:
            logger.error(f"创建MinIO客户端失败")
            raise
    return minio_client
