# auth_simple.py
import binascii
import hashlib
import hmac
import streamlit as st

DEFAULT_ITERS = 200_000
ALGO = "sha256"

def _pbkdf2_hex(password: str, salt_hex: str, iterations: int) -> str:
    salt = binascii.unhexlify(salt_hex.encode("ascii"))
    dk = hashlib.pbkdf2_hmac(ALGO, password.encode("utf-8"), salt, iterations)
    return binascii.hexlify(dk).decode("ascii")

def require_login(header: str = "Secure Access") -> str:
    """
    Renders a sidebar login form and blocks the app until the user authenticates.
    Returns the username once authenticated.
    """
    if st.session_state.get("auth_ok"):
        return st.session_state.get("username", "")

    cfg = st.secrets.get("auth", {})
    expected_user = cfg.get("username", "")
    salt_hex = cfg.get("salt_hex", "")
    expected_hash = cfg.get("password_hash", "")
    iterations = int(cfg.get("iterations", DEFAULT_ITERS))

    with st.sidebar.form("login_form", clear_on_submit=False):
        st.subheader(header)
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign in")

    if not submit:
        st.stop()

    if u != expected_user:
        st.error("Invalid credentials.")
        st.stop()

    computed = _pbkdf2_hex(p, salt_hex, iterations)

    if hmac.compare_digest(computed, expected_hash):
        st.session_state["auth_ok"] = True
        st.session_state["username"] = u
        st.rerun()
    else:
        st.error("Invalid credentials.")
        st.stop()

def logout_button(label: str = "Log out"):
    if st.sidebar.button(label):
        for k in ("auth_ok", "username"):
            st.session_state.pop(k, None)
        st.rerun()
