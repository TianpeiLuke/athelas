"""
Factory for creating model instances.

Provides centralized model creation with type safety and validation.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..hyperparams.hyperparameters_lightgbmmt import LightGBMMtModelHyperparameters
    from ..loss.base_loss_function import BaseLossFunction
    from ..base.training_state import TrainingState

from ..base.base_model import BaseMultiTaskModel
from ..implementations.mtgbm_model import MtgbmModel


class ModelFactory:
    """
    Factory for creating model instances.

    Provides centralized model creation with type safety and
    configuration validation.
    """

    _registry = {
        "mtgbm": MtgbmModel,
    }

    @classmethod
    def create(
        cls,
        model_type: str,
        loss_function: "BaseLossFunction",
        training_state: "TrainingState",
        hyperparams: "LightGBMMtModelHyperparameters",
    ) -> BaseMultiTaskModel:
        """
        Create a model instance.

        Parameters
        ----------
        model_type : str
            Type of model ('mtgbm')
        loss_function : BaseLossFunction
            Loss function instance
        training_state : TrainingState
            Training state for tracking progress
        hyperparams : LightGBMMtModelHyperparameters
            Model hyperparameters

        Returns
        -------
        model : BaseMultiTaskModel
            Instantiated model

        Raises
        ------
        ValueError
            If model_type is not registered
        """
        if model_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown model_type: '{model_type}'. Available types: {available}"
            )

        model_class = cls._registry[model_type]

        return model_class(
            loss_function=loss_function,
            training_state=training_state,
            hyperparams=hyperparams,
        )

    @classmethod
    def register(cls, name: str, model_class: type) -> None:
        """
        Register a new model type.

        Parameters
        ----------
        name : str
            Name to register the model under
        model_class : type
            Model class to register

        Raises
        ------
        TypeError
            If model_class doesn't inherit from BaseMultiTaskModel
        """
        if not issubclass(model_class, BaseMultiTaskModel):
            raise TypeError(f"{model_class} must inherit from BaseMultiTaskModel")
        cls._registry[name] = model_class

    @classmethod
    def get_available_models(cls) -> list:
        """
        Get list of available model types.

        Returns
        -------
        model_types : list
            List of registered model type names
        """
        return list(cls._registry.keys())
