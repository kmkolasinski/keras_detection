import collections
from collections import OrderedDict
from typing import Any, Dict, Union, Callable

NestedDict = Union[Dict[str, "NestedDict"], Any]


def flatten_dict(
    dictionary: Dict[str, Any], sep: str = "/", parent_key: str = ""
) -> OrderedDict:

    """Flatten nested dictionary.
    For example if
         dictionary = {
            'a': {
                'x': 1,
                'y': 2}
            'b': {
                'x': 3,
                'y': 4}}
    and sep = '/' the resulting dictionary will have form:
        {
         'a/x': 1,
         'a/y': 2,
         'b/x': 3,
         'b/y': 4,
        }
    Args:
        dictionary: a dictionary which may contain another dicts
        sep: flattening separator.
        parent_key: name of the parent key, used internally.

    Returns:
        flattened dictionary
    """
    items = []
    for k, v in dictionary.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, sep=sep, parent_key=new_key).items())
        else:
            items.append((new_key, v))
    return OrderedDict(items)


def unflatten_dict(
    dictionary: Dict[str, Any], sep: str = "/", dict_type: type = OrderedDict
) -> Dict:

    """Unflatten dictionary it is a reverse operator to flatten_dict
    function i.e following equation is true

        dict == unflatten_dict(flatten_dict(dict))

    if flattening separator allows for unique keys separation.

    Args:
        dictionary: a flat dictionary with string keys e.g. "a/b/c": value
        sep: key flattening separator e.g. "/".
        dict_type: a type of the output nested structure

    Returns:
        nested_dict: nested dictionary structure of type `dict_type`.
    """
    unflattened_dict = dict_type()
    for k, v in dictionary.items():
        context = unflattened_dict
        for sub_key in k.split(sep)[:-1]:
            if sub_key not in context:
                context[sub_key] = dict_type()
            context = context[sub_key]
        context[k.split(sep)[-1]] = v
    return unflattened_dict


def map_nested_dict(
    dictionary: NestedDict,
    map_fn: Callable[[Any], Any],
    sep: str = "/",
    output_dict_type: type = dict,
) -> NestedDict:
    """Apply map_fn function on every element of nested dictionary.
    Example:
        dictionary = {
            "a0": {
                "a1": 5,
                "a2": -1
            },
            "b0": 3
        }
        map_fn = lambda x: x**2

        d = map_nested_dict(dictionary, map_fn)

    Expected result:
        dictionary = {
            "a0": {
                "a1": 25,
                "a2": 1
            },
            "b0": 9
        }

    Args:
        dictionary: a nested dictionary
        map_fn: a function which is applied on every element of nested dict:
        sep: flattening separator used by flatten_dict function

    Returns:

    """
    return unflatten_dict(
        {k: map_fn(v) for k, v in flatten_dict(dictionary, sep=sep).items()},
        dict_type=output_dict_type,
    )
