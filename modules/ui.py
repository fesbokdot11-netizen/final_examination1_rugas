"""Shared presentation helpers for Streamlit pages."""
from html import escape

import streamlit as st


def render_page_header(eyebrow: str, title: str, description: str):
    """Render a consistent title block while allowing each page to keep its own identity."""
    st.markdown(
        f"""
        <section class="page-hero">
            <div class="page-eyebrow">{escape(eyebrow)}</div>
            <h1>{escape(title)}</h1>
            <p>{escape(description)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, description: str = ""):
    """Render a compact, consistent section heading."""
    st.markdown(
        f"""
        <div class="page-section-heading">
            <h2>{escape(title)}</h2>
            {f"<p>{escape(description)}</p>" if description else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )
