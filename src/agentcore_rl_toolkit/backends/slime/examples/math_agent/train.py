"""Train the strands math agent on GSM8K via slime — Python entry point.

Prerequisites (see ``SETUP.md`` for full instructions):
    - Inside a working slime environment (``slimerl/slime:latest`` container or equivalent).
    - Agent deployed to ACR; runtime ARN + S3 bucket handy.
    - Model checkpoint and training JSONL downloaded locally.

Run:
    python train.py
"""

from agentcore_rl_toolkit.backends.slime import SlimeRunner

if __name__ == "__main__":
    SlimeRunner(
        exp_id="gsm8k-3b-smoke",
        agent_runtime_arn="arn:aws:bedrock-agentcore:<region>:<account>:runtime/<runtime-id>",
        s3_bucket="your-bucket-name",
        model_dir="/workspace/slime_workdir/models/Qwen2.5-3B-Instruct",
        data_path="/workspace/slime_workdir/data/gsm8k_tiny.jsonl",
        model_type="qwen2.5-3B",
    ).train(num_rollout=1)
