from typing import Any, ClassVar, Dict, Type

from ..decorators import pretty_repr
from .torch import (
    ContinuousMeanVFunction,
    ContinuousVFunction,
    Encoder,
)


@pretty_repr
class VFunctionFactory:
    TYPE: ClassVar[str] = "none"

    _bootstrap: bool
    _share_encoder: bool

    def __init__(self, bootstrap: bool, share_encoder: bool):
        self._bootstrap = bootstrap
        self._share_encoder = share_encoder
    #
    # def create_discrete(
    #     self, encoder: Encoder, action_size: int
    # ) -> DiscreteQFunction:
    #     """Returns PyTorch's Q function module.
    #
    #     Args:
    #         encoder: an encoder module that processes the observation to
    #             obtain feature representations.
    #         action_size: dimension of discrete action-space.
    #
    #     Returns:
    #         discrete Q function object.
    #
    #     """
    #     raise NotImplementedError

    def create_continuous(
        self, encoder: Encoder
    ) -> ContinuousVFunction:
        """Returns PyTorch's Q function module.

        Args:
            encoder: an encoder module that processes the observation and
                action to obtain feature representations.

        Returns:
            continuous Q function object.

        """
        raise NotImplementedError

    def get_type(self) -> str:
        """Returns Q function type.

        Returns:
            Q function type.

        """
        return self.TYPE

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns Q function parameters.

        Returns:
            Q function parameters.

        """
        raise NotImplementedError

    @property
    def bootstrap(self) -> bool:
        return self._bootstrap

    @property
    def share_encoder(self) -> bool:
        return self._share_encoder


class MeanVFunctionFactory(VFunctionFactory):
    """Standard Q function factory class.

    This is the standard Q function factory class.

    References:
        * `Mnih et al., Human-level control through deep reinforcement
          learning. <https://www.nature.com/articles/nature14236>`_
        * `Lillicrap et al., Continuous control with deep reinforcement
          learning. <https://arxiv.org/abs/1509.02971>`_

    Args:
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder over multiple Q functions.

    """

    TYPE: ClassVar[str] = "mean"

    def __init__(self, bootstrap: bool = False, share_encoder: bool = False):
        super().__init__(bootstrap, share_encoder)

    # def create_discrete(
    #     self,
    #     encoder: Encoder,
    #     action_size: int,
    # ) -> DiscreteMeanQFunction:
    #     return DiscreteMeanQFunction(encoder, action_size)

    def create_continuous(
        self,
        encoder: Encoder,
    ) -> ContinuousMeanVFunction:
        return ContinuousMeanVFunction(encoder)

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "bootstrap": self._bootstrap,
            "share_encoder": self._share_encoder,
        }









V_FUNC_LIST: Dict[str, Type[VFunctionFactory]] = {}


def register_v_func_factory(cls: Type[VFunctionFactory]) -> None:
    """Registers Q function factory class.

    Args:
        cls: Q function factory class inheriting ``QFunctionFactory``.

    """
    # print("register v")
    is_registered = cls.TYPE in V_FUNC_LIST
    assert not is_registered, "%s seems to be already registered" % cls.TYPE
    V_FUNC_LIST[cls.TYPE] = cls


def create_v_func_factory(name: str, **kwargs: Any) -> VFunctionFactory:
    """Returns registered Q function factory object.

    Args:
        name: registered Q function factory type name.
        kwargs: Q function arguments.

    Returns:
        Q function factory object.

    """
    assert name in V_FUNC_LIST, "%s seems not to be registered." % name
    factory = V_FUNC_LIST[name](**kwargs)
    assert isinstance(factory, VFunctionFactory)
    return factory


register_v_func_factory(MeanVFunctionFactory)
