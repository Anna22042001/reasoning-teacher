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
    input_query = input_query.replace(",", "").strip()

    # Handle numbers directly
    if is_number(input_query):
        result = float(input_query)
        return int(result) if result.is_integer() else result

    # Handle parentheses
    while '(' in input_query or ')' in input_query:
        # Find innermost parentheses
        inner_expr = re.search(r'\([^()]*\)', input_query)
        if inner_expr:
            sub_expr = inner_expr.group(0)  # Get the matched parentheses
            sub_result = Calculator(sub_expr[1:-1])  # Remove parentheses and evaluate
            input_query = input_query.replace(sub_expr, str(sub_result), 1)
        else:
            raise ValueError("Mismatched parentheses in input")

    # Parse input with operators
    for op in operators.keys():
        if op in input_query:
            parts = input_query.split(op, 1)  # Split on the first occurrence of the operator
            left = Calculator(parts[0].strip())
            right = Calculator(parts[1].strip())
            result = operators[op](left, right)
            print(result)
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
