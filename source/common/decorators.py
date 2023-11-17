from functools import wraps

def singleton(original_class):
    create = original_class.__new__
    instance = None

    @wraps(original_class.__new__)
    def __new__(cls, *args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = create(cls, *args, **kwargs)
        return instance
    original_class.__new__ = __new__
    return original_class