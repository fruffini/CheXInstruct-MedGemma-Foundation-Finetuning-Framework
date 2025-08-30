from accelerate import Accelerator

from .helper import *

def initialize_accelerator_safely(training_args):
    """
    Initialize accelerator with proper error handling for SLURM + TorchRun.
    """
    import os
    import time
    import random

    if "LOCAL_RANK" in os.environ and "RANK" in os.environ:
        print("🔄 Detected TorchRun environment, using simple accelerator initialization...")

        # Simple initialization that works with torchrun
        accelerator = Accelerator(
                gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                log_with=training_args.report_to if training_args.with_tracking else None,
                project_dir=training_args.output_dir,
                mixed_precision="bf16" if training_args.bf16 else "fp16" if training_args.fp16 else "fp32",
                split_batches=False,

        )

        print("✅ TorchRun accelerator initialized successfully")
        return accelerator


    # Fallback for accelerate launch (with retry logic)
    max_retries = 5
    base_delay = 5

    for attempt in range(max_retries):
        try:
            print(f"🔄 Initializing Accelerator (attempt {attempt + 1}/{max_retries})...")

            # Add a small random delay to prevent simultaneous initialization
            if attempt > 0:
                delay = base_delay + random.uniform(0, 5)
                print(f"⏳ Waiting {delay:.1f}s before retry...")
                time.sleep(delay)

            accelerator = Accelerator(
                    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                    log_with=training_args.report_to if training_args.with_tracking else None,
                    project_dir=training_args.output_dir,
                    mixed_precision="bf16" if training_args.bf16 else "fp16",
                    split_batches=False,

            )

            print("✅ Accelerator initialized successfully")
            return accelerator

        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")

            if "EADDRINUSE" in str(e) or "address already in use" in str(e):
                print("🔍 Port conflict detected, will retry with delay...")
                if attempt == max_retries - 1:
                    print("💡 Try running: pkill -f 'python.*finetune' to clean up any hanging processes")
                    raise RuntimeError(
                            "Failed to initialize accelerator after multiple attempts. "
                            "This is likely due to port conflicts from previous runs. "
                            "Please check for hanging processes and try again."
                    )
            else:
                # For non-port related errors, fail immediately
                raise

    raise RuntimeError("Failed to initialize accelerator after all retries")
