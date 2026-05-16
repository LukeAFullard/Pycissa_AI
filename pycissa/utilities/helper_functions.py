import inspect
def get_keyword_args(func):
    # Get the function signature
    sig = inspect.signature(func)

    # Extract parameters
    params = sig.parameters

    # Get list of keyword arguments (parameters with default values)
    keyword_args = [name for name, param in params.items() if param.default != inspect.Parameter.empty]

    return keyword_args
