"""
Lambda handler for CAMRIE MRI simulation.

Accepts simulation configuration via event payload,
runs simulation, and returns results to S3.
"""

import json
import os
import tempfile
from pathlib import Path

import boto3


def handler(event, context):
    """
    Lambda handler for CAMRIE simulation.
    
    Event format:
    {
        "config": {
            "rho_s3": "s3://bucket/rho.nii.gz",
            "t1_s3": "s3://bucket/t1.nii.gz",
            "t2_s3": "s3://bucket/t2.nii.gz",
            "sequence_s3": "s3://bucket/sequence.seq",
            "slice_normal": [0, 0, 1],
            "num_slices": 5,
            "output_s3": "s3://bucket/output/"
        }
    }
    """
    try:
        config = event.get("config", {})
        
        # Create temp directories
        temp_dir = tempfile.mkdtemp()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Download input files from S3
        s3 = boto3.client("s3")
        
        def download_s3(s3_uri: str, local_path: Path) -> str:
            """Download file from S3."""
            bucket, key = s3_uri.replace("s3://", "").split("/", 1)
            local_file = local_path / Path(key).name
            s3.download_file(bucket, key, str(local_file))
            return str(local_file)
        
        rho_path = download_s3(config["rho_s3"], input_dir)
        t1_path = download_s3(config["t1_s3"], input_dir)
        t2_path = download_s3(config.get("t2_s3", config["t1_s3"]), input_dir)
        
        seq_path = None
        if "sequence_s3" in config:
            seq_path = download_s3(config["sequence_s3"], input_dir)
        
        # Import CAMRIE
        import sys
        sys.path.insert(0, os.environ.get("LAMBDA_TASK_ROOT", "/var/task"))
        
        from camrie import quick_sim
        
        # Run simulation
        result = quick_sim(
            rho_path=rho_path,
            t1_path=t1_path,
            t2_path=t2_path,
            sequence_path=seq_path,
            slice_normal=config.get("slice_normal", [0, 0, 1]),
            num_slices=config.get("num_slices", 1),
            slice_thickness_mm=config.get("slice_thickness_mm", 5.0),
            spin_factor=config.get("spin_factor", 1),
            output_dir=str(output_dir),
        )
        
        # Upload results to S3
        output_s3 = config.get("output_s3", "s3://default-bucket/camrie-output/")
        bucket, prefix = output_s3.replace("s3://", "").split("/", 1)
        
        uploaded_files = []
        for file in output_dir.rglob("*"):
            if file.is_file():
                key = prefix.rstrip("/") + "/" + str(file.relative_to(output_dir))
                s3.upload_file(str(file), bucket, key)
                uploaded_files.append(f"s3://{bucket}/{key}")
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Simulation complete",
                "num_slices": len(result["images"]),
                "output_files": uploaded_files,
            })
        }
        
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "type": type(e).__name__,
            })
        }
