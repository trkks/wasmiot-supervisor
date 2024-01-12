"""
Sentry integration for 
"""

from os import environ
from typing import Any

import sentry_sdk
from flask import Flask
from sentry_sdk.integrations.flask import FlaskIntegration


def init_app(app: Flask):
    """
    Integrate sentry logging into application.
    """

    # Try to get enviroment name from different sources
    if enviroment := environ.get('CI_ENVIRONMENT_NAME'):
        enviroment = enviroment.lower()
    elif app.testing:
        enviroment = "testing"
    elif app.debug:
        enviroment = "development"

    # Populate config with environment variables for sentry logging
    app.config.setdefault('SENTRY_DSN', environ.get('SENTRY_DSN'))
    app.config.setdefault('SENTRY_ENVIRONMENT', enviroment)
    app.config.setdefault('CI_COMMIT_SHA', environ.get('CI_COMMIT_SHA'))

    # Setup sentry logging
    sentry_dsn = app.config.get("SENTRY_DSN")
    enviroment = app.config.get("SENTRY_ENVIRONMENT", "production")

    integrations = [
        # Flask integration
        FlaskIntegration()
    ]

    if sentry_dsn:
        sentry = sentry_sdk.init(
            dsn=sentry_dsn,
            integrations=integrations,

            traces_sample_rate=1.0,

            environment=enviroment,
        )

        app.config.setdefault("SENTRY_RELEASE", sentry._client.options["release"])  # pylint: disable=protected-access
        app.logger.info("Sentry logging enabled.", extra={"SENTRY_DSN": sentry_dsn})

    else:
        app.logger.warning("Sentry DSN not found. Sentry logging disabled.")
