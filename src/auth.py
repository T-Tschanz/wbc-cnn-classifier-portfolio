# auth.py
# auth.py
from __future__ import annotations
import streamlit as st
import streamlit_authenticator as stauth

def _build_config_from_secrets():
    """Read auth config from st.secrets with sane defaults."""
    s = st.secrets.get("auth", {})
    cookie = s.get("cookie", {})
    creds = s.get("credentials", {})
    return {
        "credentials": creds,
        "cookie": {
            "name": cookie.get("name", "wbc_auth"),
            # accept several possible keys, fall back to a placeholder
            "key": cookie.get("key") or cookie.get("cookie_key") or cookie.get("signature_key") or "CHANGE_ME",
            "expiry_days": int(cookie.get("expiry_days", 7)),
        },
        "preauthorized": s.get("preauthorized", []),
    }

def _make_authenticator(cfg):
    """
    Create stauth.Authenticate in a version-agnostic way.
    Tries positional signature first, then keyword variants.
    """
    # Most robust: use positional arguments (works across many versions)
    args = (
        cfg["credentials"],
        cfg["cookie"]["name"],
        cfg["cookie"]["key"],
        cfg["cookie"]["expiry_days"],
        cfg.get("preauthorized", []),
    )
    try:
        return stauth.Authenticate(*args)
    except TypeError:
        # Try keyword with cookie_key (newer versions)
        try:
            return stauth.Authenticate(
                credentials=cfg["credentials"],
                cookie_name=cfg["cookie"]["name"],
                cookie_key=cfg["cookie"]["key"],
                cookie_expiry_days=cfg["cookie"]["expiry_days"],
                preauthorized=cfg.get("preauthorized", []),
            )
        except TypeError:
            # Try keyword with signature_key (older versions)
            return stauth.Authenticate(
                credentials=cfg["credentials"],
                cookie_name=cfg["cookie"]["name"],
                signature_key=cfg["cookie"]["key"],
                cookie_expiry_days=cfg["cookie"]["expiry_days"],
                preauthorized=cfg.get("preauthorized", []),
            )

def login(header: str = "Secure Access"):
    """
    Render the login widget and return (authenticator, name, auth_status, username).
    Handles API differences for the login() call as well.
    """
    cfg = _build_config_from_secrets()
    authenticator = _make_authenticator(cfg)
    try:
        name, auth_status, username = authenticator.login(header, location="main")
    except TypeError:
        # Older versions don't accept location kwarg
        name, auth_status, username = authenticator.login(header)
    return authenticator, name, auth_status, username

def logout_button(authenticator, label: str = "Logout"):
    """Place a logout button in the sidebar."""
    try:
        authenticator.logout(label, "sidebar")
    except TypeError:
        # Some versions only take the label
        authenticator.logout(label)

