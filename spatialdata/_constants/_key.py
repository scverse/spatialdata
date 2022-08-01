from typing import Any, Callable


class cprop:
    def __init__(self, f: Callable[..., str]):
        self.f = f

    def __get__(self, obj: Any, owner: Any) -> str:
        return self.f(owner)


class Key:
    """Class which manages keys in :class:`anndata.AnnData`."""

    class obs:
        pass

    class obsm:
        @cprop
        def spatial(cls) -> str:
            return "spatial"

    class uns:
        @cprop
        def spatial(cls) -> str:
            return Key.obsm.spatial

        @classmethod
        def nhood_enrichment(cls, cluster: str) -> str:
            return f"{cluster}_nhood_enrichment"
