# Configuration

SumoSpace is configured via `SumoSettings` or a `.env` file.

```python
from sumospace import SumoSettings

settings = SumoSettings(
    provider="ollama",
    model="phi3:mini",
    workspace=".",
    committee_enabled=True,
)
```

See the [SumoSettings reference](../reference/settings.md) for all options.
