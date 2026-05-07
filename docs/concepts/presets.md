# Presets

Presets are named configurations for common use cases.

```python
from sumospace import SumoSettings

# Local-only, maximum safety
settings = SumoSettings.from_preset("local_safe")

# Speed-optimised, committee disabled
settings = SumoSettings.from_preset("fast")
```
