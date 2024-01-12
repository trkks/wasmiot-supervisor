"""
Itegrate python logging into flask application.
"""

import logging
from flask.logging import default_handler as flask_handler
from flask import Flask


def init_app(app: Flask, logger: logging.Logger = None):
    """
    Integrate our own logging interface into application.

    To bind logger into your application instance use::
        >>> init_logging(app)

    :param app: :class:`~Flask` instance to use as logging basis.
    """

    # If flask is running in debug mode, set our own handler to log also debug
    # messages.
    if app.debug:
        logger.setLevel(level=logging.DEBUG)

    logger.addHandler(flask_handler)

    if app.config.get("SENTRY_DSN"):
        from .sentry import init_app as init_sentry_logging  # pylint: disable=import-outside-toplevel
        init_sentry_logging(app)

    try:
        from flask_rich import RichApplication
        RichApplication(app)
    except ImportError:
        pass
