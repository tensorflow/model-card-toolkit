"""
Removal notice for the ModelCardGenerator TFX component.
"""

raise ImportError(
    """
    The ModelCardGenerator component and its dependencies have been moved to
    the TFX Addons project (https://github.com/tensorflow/tfx-addons).

    Install the tfx-addons package with

    ```
    $ pip install tfx-addons[model_card_generator]
    ```

    You can then import the component via
    ```
    from tfx_addons.model_card_generator.component import ModelCardGenerator
    ```

    and use it as usual.
    """
)
