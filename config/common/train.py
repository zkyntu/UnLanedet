train = dict(
    output_dir="./output",
    init_checkpoint="",
    max_iter=90000,
    amp=dict(enabled=False),  # options for Automatic Mixed Precision
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    checkpointer=dict(period=500, max_to_keep=3),  # options for PeriodicCheckpointer
    eval_period=500,
    log_period=20,
    device="cuda",
    # ...
)