import json
from sumospace.schemas import ResolverOutput
print(json.dumps(ResolverOutput.model_json_schema(), indent=2))
