
class DisplayablePath(object):
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(list(path
                               for path in root.iterdir()
                               if criteria(path)),
                          key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))


class DisplayGroupings:
    child_prefix_middle = '├──'
    child_prefix_last = '└──'
    parent_prefix_middle = '    '
    parent_prefix_last = '│   '

    def __init__(self, data, parent, is_last):
        self.data = data
        self.parent = parent
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        return self.__class__.__name__  # + ': ' + self.group_id

    @classmethod
    def make_tree(cls, data, parent=None, is_last=False, criteria=None):
        criteria = criteria or cls._defautl_criteria

        displayable_root = cls(data, parent, is_last)
        yield displayable_root

        count = 1
        for child in data:
            is_last = count == len(data)
            if child.__class__ is not Series:
                yield from cls.make_tree(child.data, parent=displayable_root,
                                         is_last=is_last, criteria=criteria)
            else:
                yield cls(child, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    def displayable(self):
        if self.parent is None:
            return self.displayname

        if self.is_last:
            _group_prefix = self.child_prefix_last
        else:
            _group_prefix = self.child_prefix_middle

        parts = [f'{_group_prefix} {self.displayname}']

        parent = self.parent
        while parent and parent.parent is not None:
            if parent.is_last:
                parts.append(self.parent_prefix_middle)
            else:
                parts.append(self.parent_prefix_last)
        return ''.join(reversed(parts))
