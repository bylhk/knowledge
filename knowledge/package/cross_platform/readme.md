# Cross platform
DevOps is always multi-platform: local machines, VMs and various clouds. It can save our life if the modules support multi-platform.

## io Example
Build up a custom module to handle multi-platform automatically.
```python
io = Custom_io_module()
io.load(path)
# path = /local_path/example.csv
# path = gs://bucket/example.csv
# path = s3://bucket/example.csv
```

## Runner Example
Build up a runner module to handle the certs and variables for the pipelines.
```bash
python -m module.runner
python -m module.runner --local
* python -m module.runner --docker
```

### * Docker container
Cloud service usually run pipelines with docker images. Local runner can test the pipeline quickly, but it can't test the docker image. Building a docker wrapper can run the pipeline with container, it can speed up the troubleshooting.
