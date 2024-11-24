import json
import os
import warnings
from typing import Optional

from paths import get_finetune_data_path


def save_finetune_data(data, platform_key: str, finetune_key: str, strict=True) -> str:
    path = get_finetune_data_path(platform_key, finetune_key)
    print("Saving finetune data")
    print("-" * 80)
    print("Path:       {}".format(path))
    print("Samples:    {}".format(len(data["input"])))
    print("-" * 80)

    if os.path.exists(path):
        with open(path) as f:
            data_string = json.dumps(data, indent=4)
            existing_data_string = f.read()
            if data_string != existing_data_string:
                message = "Finetune data file already exists but is different at: {}".format(path)
                if strict:
                    raise Exception(message)
                else:
                    warnings.warn(message)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    return path


def load_finetune_data(platform_key: str, finetune_key: str) -> Optional[dict]:
    path = get_finetune_data_path(platform_key, finetune_key)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data
    else:
        return None


def list_of_dicts_to_dict_of_lists(list_of_dict):
    dict_of_lists = {}
    for key in list_of_dict[0].keys():
        dict_of_lists[key] = [d[key] for d in list_of_dict]
    return dict_of_lists


from functools import partial, wraps
from beartype import beartype
from beartype.typing import Callable, Optional, Union, List, Tuple
# from toolformer_pytorch import invoke_tools
import re
def add(a, b):
    "Same as a + b."
    return a + b

def sub(a, b):
    "Same as a - b."
    return a - b

def mul(a, b):
    "Same as a * b."
    return a * b

def truediv(a, b):
    "Same as a / b."
    return a / b

def is_number(s):
    "Check if the string represents a number (int or float)."
    try:
        float(s)
        return True
    except ValueError:
        return False

def Calculator(input_query: str):
    operators = {
        '+': add,
        '-': sub,
        '*': mul,
        '/': truediv
    }
    print('input_query', input_query)
    input_query = input_query.replace(",", "")
    # Handle numbers directly (int or float)
    input_query = input_query.strip()
    if is_number(input_query):
        result = float(input_query)
        return int(result) if result.is_integer() else result

    # Parse input with multiple operators
    for op in operators.keys():
        if op in input_query:
            parts = input_query.split(op, 1)  # Split on the first occurrence of the operator
            left = Calculator(parts[0].strip())
            right = Calculator(parts[1].strip())
            print('operators', operators)
            result = operators[op](left, right)
            print('result', result)
            return int(result) if float(result).is_integer() else round(result, 2)
    
    raise ValueError("Invalid input query")
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def identity(t):
    return t

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def try_except(fn, callback = identity):
    @wraps(fn)
    def inner(args):
        print('args', args)
        try:
            return fn(args)
        except Exception as e:
            print(e)
            return callback(e)
    return inner


def is_valid_string(s):
    return exists(re.fullmatch(r"'[^']*'|\"[^\"]*\"", s))

def is_valid_integer(s):
    return exists(re.fullmatch(r"[+-]?\d+", s))

def is_valid_float(s):
    return exists(re.fullmatch(r"[+-]?\d+(\.\d+)?", s))

def parse_param(s: str) -> Optional[Union[int, float, str]]:
    if is_valid_string(s):
        return str(s)
    elif is_valid_integer(s):
        return int(s)
    elif is_valid_float(s):
        return float(s)

    return None


@beartype
def replace_fn(
    registry: dict[str, Callable],
    matches,
    delimiter = '→'
):
    orig_text = matches.group(0)

    text_without_end_api_token = matches.group(1)
    end_api_token = matches.group(4)
    function_name = matches.group(2)

    # unable to find function in registry
    if function_name not in registry:
        
        return orig_text

    fn = registry[function_name]
    params = matches.group(3).split(',')
    params = list(map(lambda s: s.strip(), params))
    params = list(filter(len, params))
    params = list(map(parse_param, params))
    params = matches.group(3)
    # print('params', params)
    # if any of the parameters are not parseable, return

    if any([(not exists(p)) for p in params]):
        return orig_text
    
    # just return original text if there is some error with the function

    out = try_except(fn, always(None))(params)
    print('out', out)
    # the api calling function can also arrest the process, by returning None

    if not exists(out):
        return orig_text

    # return original text with the output delimiter and the stringified output

    return f'{text_without_end_api_token}{end_api_token}{delimiter}{str(out)}'


def create_function_regex(
    api_start = '[',
    api_stop = ']'
):
    api_start_regex, api_stop_regex = map(re.escape, (api_start, api_stop))
    return rf'({api_start_regex}(\w+)\(([^)]*)\))({api_stop_regex})'

def invoke_tools(
    registry: dict[str, Callable] = {'Calculator': Calculator},
    text: str = '',
    delimiter: str = ' ',
    api_start = '[',
    api_stop = ']'
) -> str:
    regex = create_function_regex(api_start, api_stop)
    replace_ = partial(replace_fn, registry, delimiter = delimiter)
    return re.sub(regex, replace_, text)
