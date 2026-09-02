"""Top-level training controller lifecycle."""

from ..logger import logger
from ..schema import ConfigSchema
from .artifacts import finalize_run_artifacts


def execute_training(
    controller,
    config: ConfigSchema,
    *,
    configuration_provenance: dict | None = None,
) -> None:
    """Run one controller lifecycle and always release manager resources."""
    setup_succeeded = False
    try:
        controller.setup()
        setup_succeeded = True
        controller.start_training()
    except KeyboardInterrupt:
        logger.info("Interrupted by user — shutting down")
    except Exception:
        logger.exception("Training failed with unhandled exception")
        raise
    finally:
        try:
            if setup_succeeded:
                finalize_run_artifacts(
                    controller,
                    config,
                    configuration_provenance=configuration_provenance,
                )
        finally:
            controller.teardown()
