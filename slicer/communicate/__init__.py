"""Communication Optimization Module (Sec. II-D): autoencoder, shared memory, message channels."""

from .channel import (  # noqa: F401
    Arena,
    Channel,
    PipeEndpoint,
    ShmEndpoint,
    open_pipe,
    open_shm,
    pipe_chunk,
    tmpfs_root,
)
