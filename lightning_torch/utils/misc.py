"""
  Taken from this repository:
  https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/template_utils.py
"""

import warnings

from omegaconf import DictConfig

from lightning_torch.utils.logging import log, print_config


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.

    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        log.info("Printing config tree with Rich! <config.print_config=True>")
        print_config(config, resolve=True)
