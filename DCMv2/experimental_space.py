from pathlib import Path
from dataclasses import dataclass, fields, field
import numpy as np
import warnings


class GroupingErrors(Exception):
    def UnequalMerge(self):
        # Merge is for equal level groups
        return None
    def EqualAppend(self):
        # Append is for unequal level groups
        return None


class abilities:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        name = self.__class__.__name__

        if type(self.value[0]) is not self.child:
            return name + ': ' + str(self.value)

        self.tot_string += str(name) + '\n'

        for index, item in enumerate(self.value):
            pad = '│ ' * max(0, self.hierarchy - 1)
            if index < (len(self.value) - 1):
                pad += '├─'
            else:
                pad += '└─'
            self.tot_string += pad + item.__str__()

        end_cap = '└─' + '┴──'*self.hierarchy + '─'*9*self.hierarchy + '\n'
        if self.__class__ == Cohort:
            return self.tot_string + end_cap
        return self.tot_string

    def __iter__(self):
        return iter(self.value)

    def print_child(self):
        _ = self.child(self.value).print_child()

    def split(self):
        length = len(self.value)
        if length > 1:
            half = length // 2
            group1 = self.child(self.value[:half]).split()
            group2 = self.child(self.value[half:]).split()
            self.value = [group1, group2]
        else:
            self.value = [self.child(self.value)]
        return self


@dataclass(repr=False)
class Series(abilities):
    value: list = None
    child: object = None
    tot_string: str = ''
    hierarchy: int = 4

    def __str__(self):
        return 'Final:' + str(self.value) + '\n'

    def print_child(self):
        print('in lowest level')

    def split(self):
        return self


@dataclass(repr=False)
class Study(abilities):
    value: list = None
    child: object = Series
    tot_string: str = ''
    hierarchy: int = 3


@dataclass(repr=False)
class Patient(abilities):
    value: list = None
    child: object = Study
    tot_string: str = ''
    hierarchy: int = 2


@dataclass(repr=False)
class Cohort(abilities):
    value: int
    test: int
    _test: int = field(init=False, repr=False)
    child: object = Patient
    tot_string: str = ''
    hierarchy: int = 1
    immutable: bool = True

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, test):
        if not hasattr(self, '_test'):
            self._test = test
        elif not self.immutable:
            self._test = test
        else:
            message = 'class.immutable=True by default'
            warnings.warn(message, UserWarning)



l3 = Cohort(4, np.arange(7))
print(l3.test)
l3.test = 5
print(l3.test)
l3.immutable = False
l3.test = 6
print(l3.test)
l3._test = 5
print(l3.test)
print(l3)
repr(l3)
new = l3.split()
print(new)
repr(new)
