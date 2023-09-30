def continuous_feature(name):
    return ContinuousFeature(name)


class ContinuousFeature:
    def __init__(self, name):
        self.name = name


def global_category(name, size):
    return GlobalCategory(name, size)


class GlobalCategory:
    def __init__(self, name, size):
        self.name = name
        self.size = size


def local_category(name):
    return LocalCategory(name)


class LocalCategory:
    def __init__(self, name):
        assert name.startswith('task_'), \
            "only task feature supported: {}".format(name)
        self.name = name


def local_feature(name):
    return LocalFeature(name)


class LocalFeature:
    def __init__(self, name):
        assert name.startswith('task_'), \
            "only task feature supported: {}".format(name)
        self.name = name


def sparse_local_feature(index, value):
    return SparseLocalFeature(index, value)


class SparseLocalFeature:
    def __init__(self, index, value):
        assert index.startswith('task_'), \
            "only task feature supported for index: {}".format(index)
        assert value.startswith('task_'), \
            "only task feature supported for value: {}".format(value)

        self.index = index
        self.value = value


def variable_feature(name):
    return VariableFeature(name)


class VariableFeature:
    def __init__(self, name):
        self.name = name
