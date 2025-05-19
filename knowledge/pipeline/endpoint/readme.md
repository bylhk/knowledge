Endpoint

## Functional test
Pytest test is recommended, because it can run a local endpoint for further testing.

## Load test
K6, locust tests are recommended.

## ipv6
IPv6 may cause an inefficient process due to the timeout trial, it can be solved by VPC blocker.

## Request generator
If we don't have sufficient test requests, the request generator will be a good solution. It can generate dynamic dummy requests in different scenarios.

## Logging

### Unique log key for each step

### Always with request id

### One log for One response with unique response log key
Whatever valid responses or error messages, the response log must get the same response log key without duplicates.

### Predefined status code

## Cross platform runner
Please read `package/cross_platform/readme.md`.<br>
DockerRunner can be developed with docker wrapper.