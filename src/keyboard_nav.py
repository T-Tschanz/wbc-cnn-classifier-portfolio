# keyboard_nav.py
import streamlit as st

def _prev(max_idx):
    st.session_state.current_idx = max(0, st.session_state.current_idx - 1)

def _next(max_idx):
    st.session_state.current_idx = min(max_idx, st.session_state.current_idx + 1)

def arrow_key_nav(max_idx: int) -> int:
    """Returns the current index after listening for ←/→ clicks or arrow-key presses."""
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0

    # Inject JS that forwards arrow-key presses to our hidden buttons
    st.markdown("""
    <script>
      window.addEventListener('keydown', function(e) {
        const prev = document.getElementById('prev-btn');
        const nxt  = document.getElementById('next-btn');
        if (!prev || !nxt) return;
        if (e.key === 'ArrowLeft')  prev.click();
        if (e.key === 'ArrowRight') nxt.click();
      });
    </script>
    """, unsafe_allow_html=True)

    # Render two invisible buttons to drive our callbacks
    st.button("← Prev", key="prev-btn", on_click=_prev, args=(max_idx,))
    st.button("Next →", key="next-btn", on_click=_next, args=(max_idx,))

    return st.session_state.current_idx


