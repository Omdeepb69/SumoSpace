# sumospace/benchmarks/fixtures/sample_project/utils.py
# BENCHMARK FIXTURE — do not modify
# These functions intentionally have no docstrings

def calculate_discount(price, discount_percent):
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount must be between 0 and 100")
    return price * (1 - discount_percent / 100)

def format_currency(amount, currency="USD"):
    symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
    symbol = symbols.get(currency, currency)
    return f"{symbol}{amount:.2f}"

def validate_email(email):
    return "@" in email and "." in email.split("@")[-1]

def paginate(items, page, per_page=10):
    start = (page - 1) * per_page
    end = start + per_page
    return items[start:end]

def merge_dicts(*dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result

def truncate_string(text, max_length, suffix="..."):
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def flatten_list(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
