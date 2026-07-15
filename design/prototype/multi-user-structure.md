Large architectural change based around the following.

* We want to remove the dependecy on fragile ssh to keep vllm alive, and allow muiltiple users within a project to access a single running vllm, by each creating ssh tunnel to single node runnning vllm model.
* local client should be able to detach and reattach.

To make this happen
* all control of vllm processes on isambard is managed within the process itself.
* all communication with local users is via lockfile and logs stored in a project accessible location

* vllm startup is triggered by a remote user based on previosly uploaded config and script or new config and script
* vllm state is persisted in a lockfile in the $PROJECTDIR on isambard, which monitors jobs that have are pending -> initializing -> running, (request) cancel, failed or stopped, and provides enough details for user to connect to or manage running jobs.
* job lifecycle is largely automatic after being started:
    * jobs are `pending` until allocated resources
    * when slurm script starts they are `initializing`
    * jobs monitor themselves for startup failure -> `failed` status
    * jobs intercept SLURM timeouts -> `stopped`
    * jobs can time out due to inactivity by monitoring their own log files -> `stopped`
    * jobs can be cancelled remotely by requesting `cancel` through the status flag in lockfile.
    * jobs can be forced to shutdown remotely by `scancel` the jobid and remotely updateing the lockfile status to `stopped`

* localhost clients monitor lockfile status flag.
* clients can tail log files for more information.
* when a lockfile status is `running` client ssh connections can be made to running instances by tunnelling (any project member).

* vllm instances on isambard run as sbatch processes (in the interactive allocation)
* vllm instances monitor their own usage and will shut themselves down if no activity for a  configurable time out by monitoring logfiles for specific api requests.

## design choices

* move away from templated bash scripts as far as possible to plain bash functions and
slurm scripts.
* avoid heavy use of global bash environment variables. Stick to functions and local variables and reduced complexity in reasoning.
* use node libraries for configuration flags
* future roadmap includes one user connecting simultanaously to multiple running models and locally routing base on model name.

## fixed project structure

* $PROJECTDIR/hf/ - model download directories
* $PROJECTDIR/engine/jobs/
    * utils.sh - shared shell scipt utilities (see prototype.sh) - lockfile management, caching, diagnostics, etc.
    * hf.sh - model download slurm script (using interactive reservation).
    * preamble.sh - isambard specific shared environment variables
    * <jobname>/
        * status.json - the main lock file
        * jit-cache.tar.gz - the complilation cache
        * vllm.0.log - the logs of node 0 (vllm.1.log, ... etc for multi-node)
        * vllm.yaml - the configuration for this job
        * slurm.sh - slurm script to launch the job. will be multinode or single node and reuse the utils.sh functions - see test-vllm.sh for skeleton.
* $PROJECTDIR/engine/vllm/
    * setup.sh - slurm script to install a specific vllm version
    * vllm_logs.yaml - log configuration to ensure correctly timestamped vllm output.
    * <version>/ - vllm version installation directory
* $PROJECTDIR/engine/diagnostics
    * <jobname>/
        * <date>/
            * copy of vllm.*.log, vllm.yaml, slurm.sh for failed jobs

## lockfile format (status.json)

* status: pending | initializing | running | failed | stopped | cancel
* jobName: e.g. qwen36
* model: e.g. Qwen/Qwen3.6-35B-A3B-FP8
* serverPort: randomly generated high port
* requestedTime: time job first submitted
* idleTimeout: the length of time to wait in minutes
* slurmJobId?: the slurm job identifier
* computeHostname?: the hostname of the head compute node
* startTime?: the start time of the slurm job
* stopTime?: the end time of the slurm job (future) or the time the job stopped
* vllmPid?: the process id of the vllm
* reason?: the reason for stopping or failure
* exitCode?: the vllm exit code in case of failure

## user commands

ivllm list
- lists all lockfiles in $PROJECTDIR/engine/jobs
- presents list of jobs, current status, time left (if running), reasons for stopping

ivllm connect <job>
- starts up job in `stopped`, or `failed` state by running slurm.sh in interactive reservation
- tails remote log files while job in `pending` or `initializing` state
- connects to `running` job by building ssh tunnel

ivllm connect <job> --config <local vllm.yaml>
- parses local vllm.yaml file and constructs slurm.sh script.
- creates new job directory if non existent
- warns if existing config is working (job not in `failed` state in lockfile)
- uploads vllm.yaml and slurm.sh (overwriting existing)
- triggers hf download if required.
- starts job as per above.

ivllm cancel <job>
- updates status of <job> lockfile to `cancel`
- tail logs until remote job has shutdown (status `stopped`)

ivllm cancel <job> --force
- scancel based on slurmJobId
- update lockfile to status `stopped`

ivllm setup <version>
- trigger vllm setup script remotely.

## log files

with `vllm_logs.json` referenced by `VLLM_LOGGING_CONFIG_PATH` on startup logs show lines like:

```
(APIServer pid=34633) [2026-07-14 22:37:50,765] INFO:     10.242.0.28:38194 - "GET /health HTTP/1.1" 200 OK
(APIServer pid=34633) [2026-07-14 22:37:50,935] INFO:     10.242.0.28:45178 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=34633) [2026-07-14 22:38:02,993] INFO:     10.242.0.28:45178 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=34633) [2026-07-14 22:38:03,533] INFO:     10.242.0.28:34266 - "GET /health HTTP/1.1" 200 OK
```

which should work with design approach but time formats will need tweaking
