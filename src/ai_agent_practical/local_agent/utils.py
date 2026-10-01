import json

def fallback(o):
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)

def printf_custom(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=fallback))
