"""
Utilities for Flask-endpoint operations etc.
"""

from flask import jsonify


def endpoint_failed(request, msg="", status_code=500):
    """
    Helper to construct JSON response for communicating that request to endpoint failed (and log this).

    :request The request that failed
    :msg The message to use in the `status` field of response
    :status_code The status code to use in response
    """
    # Log the message TODO Use an actual logger.
    print(f"{request.method} on '{request.path}': {msg}")

    resp = jsonify({ "status": msg })
    resp.status_code = status_code
    return resp
