from typing import Any


def from_dict_to_type(
    dictionary: dict[str, Any],
) -> Any:

    return type("Class", (), dictionary)


def from_type_to_dict(
    instance: Any,
) -> dict[str, Any]:

    new_dictionary: dict[str, Any] = {}
    dictionary: dict[str, Any] = dict(instance.__dict__)

    for key in dictionary.keys():
        if not key.startswith("__"):
            new_dictionary[key] = dictionary[key]

    return new_dictionary
