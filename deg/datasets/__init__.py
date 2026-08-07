import importlib

__attributes = {
    'TextConditionedGSOctreeLatent': 'latent_seq',
    'ImageConditionedGSOctreeLatent': 'latent_seq',
    'SmartImageConditionedGSOctreeLatent': 'latent_seq',

    'PcdFeat2Render': 'pcdfeat2render',
    'OctreePcdFeat2Render': 'pcdfeat2render',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]
