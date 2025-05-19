# Version
Create version tags to keep the static version.

# Merge request
Parallel development is common, please split development into small goals with branches.<br>
The merge request flow is [branch > dev > prod > version tag], and the merge request should contain change logs. 

# CHANGELOG
Change log should follow this format
```
## [version] - yyyy-mm-dd
Desc

### Added
- items
 
### Changed
 
### Fixed
```

# Documentation
Docstring is recommended; it can be converted to documentation pages easily.

# Unit tests
Pytest is recommended for unit tests and integration tests.

# CICD
The CICD can handle the version tag for the prod deployment process.
