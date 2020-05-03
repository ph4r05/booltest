## Job client-server


### Job server
- Start job server `server.sh`, can load big all-in-one jobs configurations as jsons, continuously. 
- Has to have public IP and unfirewalled port so clients can reach it.

### Job client

```bash
for i in {1..500}; do qsub -l select=1:ncpus=1:mem=4gb:scratch_local=500mb -l walltime=2:00:00 runner2.sh ; done
for i in {1..500}; do qsub -l select=1:ncpus=1:mem=6gb:scratch_local=500mb -l walltime=4:00:00 runner4.sh ; done
for i in {1..500}; do qsub -l select=1:ncpus=2:mem=6gb:scratch_local=500mb -l walltime=4:00:00 runner4-2.sh ; done
for i in {1..500}; do qsub -l select=1:ncpus=4:mem=8gb:scratch_local=500mb -l walltime=2:00:00 runner2-4.sh ; done
```
