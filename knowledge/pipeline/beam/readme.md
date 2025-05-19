# Beam
Beam is a data tool which allow complex the row level data processing. But it is less efficient if any group level aggregation.

## Batch processing
Some data processing is more efficient with vectorisation (eg: numpy, tensor). Therefore, the batching tool can speed up the beam pipeline by thousands of times.

## Dynamic path
Some native IO writers do not support dynamic paths with the beam template. Therefore, we need to apply the basic IO `WriteToFiles` with the `file_naming` function. *** We need to develop the custom IO sinks for the WriteToFiles if not text-based files (eg: parquet, etc). ***<br>
https://beam.apache.org/releases/pydoc/2.36.0/apache_beam.io.fileio.html#dynamic-destinations

## Cross platform runner
Please read `package/cross_platform/readme.md`.<br>
DockerRunner can be developed with docker wrapper.