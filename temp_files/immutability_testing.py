from dataclasses import dataclass, is_dataclass
import warnings


def immutable_old(immutable_field):
    """[a decorator to make specified fields 'immutable'. This prevents
        the specified fields from being altered unless <class>.immutable=False]

    Args:
        immutable_field ([list OR str]): [field name(s) to make immutable]

    Raises:
        TypeError: [if immutable_field is not str or list]

    Warns:
        UserWarning: [if attempting to mutate immutable variable]

    Returns:
        [Object]: [returns inherited class of decorated type]
    """
    if not isinstance(immutable_field, list):
        if isinstance(immutable_field, str):
            immutable_field = [immutable_field]
        else:
            raise TypeError('Input must be a field name or list of names')

    def decorator(OldClass):
        class NewClass(OldClass):
            """[A new class with immutable fields]

            Args:
                OldClass ([object]): [decorated type class]
            """
            def __init__(self, *args, **kwargs):
                if not is_dataclass(OldClass):
                    # If type is not dataclass, it must be initialized
                    super().__init__(*args, **kwargs)

                for field in immutable_field:
                    # Generate private fields for each immutable type
                    setattr(self, '_' + field, getattr(self, field))

                if not hasattr(self, 'immutable'):
                    self.immutable = True

            def __setattr__(self, key, value):
                # set field if not immutable type, otherwise _set_attr
                if key in immutable_field and hasattr(self, key):
                    self._setter(key, value)
                else:
                    object.__setattr__(self, key, value)

            def __getattribute__(self, key):
                # return hidden value for immutable types, otherwise standard
                hidden_key = '_' + key
                if key in immutable_field and hasattr(self, hidden_key):
                    return getattr(self, hidden_key)
                return object.__getattribute__(self, key)

            def _setter(self, key, value):
                # Initialize variable if not present, otherwise immutability rules
                hidden_key = '_' + key
                if not hasattr(self, hidden_key):
                    setattr(self, hidden_key, getattr(self, key))
                elif not self.immutable or key == 'immutable':
                    setattr(self, hidden_key, value)
                else:
                    message = '<class>.immutable=True, change to mutate'
                    warnings.warn(message, UserWarning)
        return NewClass
    return decorator


def immutable(imm_fields):
    """[a decorator to make specified fields 'immutable'. This prevents
        the specified fields from being altered unless <class>.immutable=False]

    Args:
        imm_field ([list OR str]): [field name(s) to make immutable]

    Raises:
        TypeError: [if immutable_field is not str or list]

    Warns:
        UserWarning: [if attempting to mutate immutable variable]

    Returns:
        [Object]: [returns inherited class of decorated type]
    """
    if not isinstance(imm_fields, list):
        if isinstance(imm_fields, str):
            imm_fields = [imm_fields]
        else:
            raise TypeError('Input must be a field name or list of names')
    assert 'immutable' not in imm_fields, 'attribute immutable is protected'

    def decorator(OldClass):
        class NewClass(OldClass):
            """[A new class with immutable fields]

            Args:
                OldClass ([object]): [decorated type class]
            """

            def __init__(self, *args, **kwargs):
                if not is_dataclass(OldClass):
                    # If type is not dataclass, it must be initialized
                    super().__init__(*args, **kwargs)

                if not hasattr(self, 'immutable'):
                    self.immutable = True

            def __setattr__(self, key, value):
                # set field if not immutable type, otherwise _set_attr
                if key in imm_fields and hasattr(self, key) and self.immutable:
                    message = '<class>.immutable=True, change to mutate'
                    warnings.warn(message, UserWarning)
                else:
                    object.__setattr__(self, key, value)

        return NewClass
    return decorator


def immutable_test():
    @immutable(['v1', 'v2'])
    @dataclass
    class DClass:
        v1: str = 'value1'
        v2: str = 'value2'
        v3: str = 'value3'

        def four_times_v1(self):
            return 4 * self.v1

    @immutable(['v1', 'v2'])
    class Normal:
        def __init__(self, v1='value1', v2='value2', v3='value3'):
            self.v1 = v1
            self.v2 = v2
            self.v3 = v3

        def four_times_v1(self):
            return 4 * self.v1

    def all_tests(obj):
        tests = []  # Stores all tests
        # Test Defaults
        tests.append(obj.v1 == 'value1')
        tests.append(obj.v2 == 'value2')
        tests.append(obj.v3 == 'value3')

        # Test for change failure
        tests.append(obj.immutable is True)
        obj.v1 = 4
        obj.v2 = 5
        obj.v3 = 6
        tests.append(obj.v1 == 'value1')
        tests.append(obj.v2 == 'value2')
        tests.append(obj.v3 == 6)

        # Test for mutability
        obj.immutable = False
        tests.append(obj.immutable is False)
        obj.v1 = 7
        obj.v2 = 8
        obj.v3 = 9
        tests.append(obj.v1 == 7)
        tests.append(obj.v2 == 8)
        tests.append(obj.v3 == 9)
        return all(tests)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        dc = DClass()
        assert all_tests(dc), 'Failed Dataclass'

        nm = Normal()
        assert all_tests(nm), 'Failed Normal'

    print('Passed Assertion Test')


if __name__ == '__main__':
    immutable_test()
