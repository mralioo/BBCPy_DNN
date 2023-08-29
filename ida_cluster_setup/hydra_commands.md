# Hydra Network

Main documentation: https://git.tu-berlin.de/ml-group/hydra/documentation


### Specs 

| Number of heads | CPU | RAM | GPU | [Slurm gpu type](#GPUTYPE) | Fast local storage | Name |
|--|--|--|--| -- | -- | -- | 
| 20 | 2x16 | 512 GB | - | - | 1.7 TB | head001-020 |
| 4 | 2x16 | 1024 GB | 8x A100 80GB | 80gb | 7.3 TB | head021-024 |
| 8 | 2x16 | 340 GB | 4x P100 12 GB | p100 | 700 GB | head026-033 | 
| 2| 2x16 | 750 GB | 2x Quadro RTX 6000 24 GB | 6000 | 3,5 TB | head034-035 |
| 12 | 2x16 | 775 GB | 4x A100 40GB | 40gb | 3.2 TB | head040-051 |
| 4 | 2x10 | 180 GB | 2x RTX 3090 24GB | 3090 | 1.3 TB | head055-058| 



# Steps 


## log to Hydra head node
```shell
srun --partition=cpu-2h --pty bash
```

## Generate enviorment container using apptainer

```shell
apptainer build python_container.sif python_container.def
```


## Compress datasets

```
squash-dataset /home/space/datasets/squashfs_example /home/space/datasets-sqfs/squashfs_example.sqfs
```




Available partitions:

| Name | Kind | Runtime |
| -- | -- | -- |
| cpu-9m | CPU | 9m |
| cpu-2h | CPU | 2h |
| cpu-5h | CPU | 5h |
| cpu-2d | CPU | 2d |
| cpu-7d | CPU | 7d |
| gpu-9m | GPU | 9m |
| gpu-2h | GPU | 2h |
| gpu-5h | GPU | 5h |
| gpu-2d | GPU | 2d |
| gpu-7d | GPU | 7d |
| cpu-test | CPU | 15m |
| gpu-test | GPU | 15m|
