# Config file for parsl
from parsl.config import Config
from parsl.channels import LocalChannel
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor
from parsl.executors import ThreadPoolExecutor
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_interface

localConfig = Config(
    app_cache=True, 
    checkpoint_files=None, 
    checkpoint_mode=None, 
    checkpoint_period=None, 
    executors=(ThreadPoolExecutor(
        label='threads', 
        max_threads=1, 
        storage_access=None, 
        thread_name_prefix='', 
        working_dir=None
    ),), 
    garbage_collect=True, 
    initialize_logging=True, 
    internal_tasks_max_threads=1, 
    max_idletime=120.0, 
    monitoring=None, 
    retries=0, 
    retry_handler=None, 
    run_dir='runinfo', 
    strategy='simple', 
    strategy_period=5, 
    usage_tracking=False
)