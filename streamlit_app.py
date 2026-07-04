"""
Streamlit frontend for the LangGraph blog-writer pipeline
(Router -> Research? -> Orchestrator -> Workers -> Reducer/Images).

Run from the SAME directory as blog_writer.py:
    streamlit run streamlit_app.py

Required env vars (in a .env file next to this app, or exported):
    GOOGLE_API_KEY   - for the Gemini planning/writing LLM
    HF_TOKEN         - for Hugging Face image generation
    TAVILY_API_KEY   - optional, only needed for hybrid/open_book research
"""

from __future__ import annotations

import os
import re
import time
from datetime import date
from pathlib import Path

import streamlit as st

import blog_writer as bw

st.set_page_config(page_title="AI Blog Writer", page_icon="📝", layout="wide")

NODE_LABELS = {
    "router": "🧭 Deciding whether research is needed",
    "research": "🔎 Researching the web for evidence",
    "orchestrator": "🗂️ Planning the blog outline",
    "worker": "✍️ Writing a section",
    "merge_content": "🧩 Merging sections into one document",
    "decide_images": "🖼️ Deciding which images to generate",
    "generate_and_place_images": "🎨 Generating images (Hugging Face) & placing them",
}

MODE_LABELS = {
    "auto": "Auto (let the router decide)",
    "closed_book": "Closed book (evergreen, no research)",
    "hybrid": "Hybrid (evergreen + fresh examples)",
    "open_book": "Open book (news / weekly roundup)",
}

IMAGE_MD_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\(images/(?P<file>[^)]+)\)\n\*(?P<caption>[^*]*)\*"
)


def env_status():
    return {
        "GOOGLE_API_KEY": bool(os.environ.get("GOOGLE_API_KEY")),
        "HF_TOKEN": bool(os.environ.get("HF_TOKEN")),
        "TAVILY_API_KEY": bool(os.environ.get("TAVILY_API_KEY")),
    }


def render_markdown_with_images(md_text: str, base_dir: Path):
    """Split the final markdown on our image blocks and interleave
    st.markdown / st.image so generated images actually render."""
    pos = 0
    for m in IMAGE_MD_RE.finditer(md_text):
        if m.start() > pos:
            st.markdown(md_text[pos : m.start()])
        img_path = base_dir / "images" / m.group("file")
        if img_path.exists():
            st.image(str(img_path), caption=m.group("caption").strip())
        else:
            st.warning(f"Image file not found on disk: {img_path}")
        pos = m.end()
    if pos < len(md_text):
        st.markdown(md_text[pos:])


with st.sidebar:
    st.header("⚙️ Settings")

    topic = st.text_area(
        "Blog topic",
        placeholder="e.g. How vector databases work under the hood",
        height=90,
    )

    as_of = st.date_input("As-of date", value=date.today())

    force_mode_ui = st.selectbox(
        "Research mode",
        options=list(MODE_LABELS.keys()),
        format_func=lambda k: MODE_LABELS[k],
        index=0,
        help="Auto lets an LLM decide. Force a mode to override that decision.",
    )

    st.divider()
    st.subheader("🎨 Image generation")

    hf_model = st.text_input(
        "Hugging Face model",
        value=os.environ.get("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell"),
        help="Any text-to-image model id supported by huggingface_hub's InferenceClient.",
    )

    max_images = st.slider("Max images in the post", min_value=0, max_value=5, value=3)

    st.divider()
    with st.expander("🔑 Environment check"):
        status = env_status()
        for key, ok in status.items():
            st.write(("✅ " if ok else "❌ ") + key)
        if force_mode_ui in ("hybrid", "open_book") and not status["TAVILY_API_KEY"]:
            st.caption(
                "No TAVILY_API_KEY set - research will proceed with 0 evidence found."
            )

    generate_clicked = st.button(
        "🚀 Generate blog post",
        type="primary",
        use_container_width=True,
        disabled=not (status_ok := env_status())["GOOGLE_API_KEY"]
        or not status_ok["HF_TOKEN"],
    )
    if not status_ok["GOOGLE_API_KEY"] or not status_ok["HF_TOKEN"]:
        st.caption("Set GOOGLE_API_KEY and HF_TOKEN to enable generation.")


st.session_state.setdefault("final_md", None)
st.session_state.setdefault("blog_title", None)
st.session_state.setdefault("last_error", None)

st.title("📝 AI Blog Writer")
st.caption(
    "Router → Research → Orchestrator → Parallel section workers → Reducer + Hugging Face images"
)


def run_pipeline(initial_state: dict):
    log_box = st.status("Starting pipeline...", expanded=True)
    task_titles: dict[int, str] = {}
    final_md = None
    blog_title = None

    try:
        stream = bw.app.stream(initial_state, stream_mode="updates", subgraphs=True)
        for event in stream:
            if isinstance(event, tuple) and len(event) == 2:
                _namespace, chunk = event
            else:
                chunk = event

            if not isinstance(chunk, dict) or not chunk:
                continue

            for node_name, output in chunk.items():
                label = NODE_LABELS.get(node_name, f"⚙️ {node_name}")

                if node_name == "router" and isinstance(output, dict):
                    mode = output.get("mode")
                    n_q = len(output.get("queries") or [])
                    log_box.write(
                        f"{label} → mode=**{mode}**, {n_q} quer{'y' if n_q == 1 else 'ies'} planned"
                    )

                elif node_name == "research" and isinstance(output, dict):
                    n_ev = len(output.get("evidence") or [])
                    log_box.write(f"{label} → {n_ev} evidence item(s) found")

                elif node_name == "orchestrator" and isinstance(output, dict):
                    plan = output.get("plan")
                    if plan is not None:
                        blog_title = plan.blog_title
                        task_titles = {t.id: t.title for t in plan.tasks}
                        log_box.write(
                            f"{label} → **{plan.blog_title}** ({len(plan.tasks)} sections, kind={plan.blog_kind})"
                        )

                elif node_name == "worker" and isinstance(output, dict):
                    for task_id, section_md in output.get("sections", []):
                        title = task_titles.get(task_id, f"section {task_id}")
                        words = len(section_md.split())
                        log_box.write(f"{label}: **{title}** ({words} words)")

                elif node_name == "merge_content":
                    log_box.write(label)

                elif node_name == "decide_images" and isinstance(output, dict):
                    n_img = len(output.get("image_specs") or [])
                    log_box.write(f"{label} → {n_img} image(s) planned")

                elif node_name == "generate_and_place_images" and isinstance(
                    output, dict
                ):
                    log_box.write(f"{label} → done")
                    if "final" in output:
                        final_md = output["final"]

                else:
                    log_box.write(label)

        log_box.update(label="✅ Pipeline complete", state="complete", expanded=False)
        return final_md, blog_title, None

    except Exception as e:
        log_box.update(label="❌ Pipeline failed", state="error", expanded=True)
        return None, None, str(e)


if generate_clicked:
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        initial_state = {
            "topic": topic.strip(),
            "as_of": as_of.isoformat(),
            "force_mode": force_mode_ui,
            "hf_image_model": hf_model.strip() or None,
            "max_images": int(max_images),
        }
        final_md, blog_title, error = run_pipeline(initial_state)
        st.session_state["final_md"] = final_md
        st.session_state["blog_title"] = blog_title
        st.session_state["last_error"] = error


if st.session_state["last_error"]:
    st.error(f"Generation failed: {st.session_state['last_error']}")

if st.session_state["final_md"]:
    st.divider()
    st.subheader(st.session_state["blog_title"] or "Generated post")

    render_markdown_with_images(st.session_state["final_md"], base_dir=Path("."))

    st.download_button(
        "⬇️ Download as Markdown",
        data=st.session_state["final_md"],
        file_name=f"{bw._safe_slug(st.session_state['blog_title'] or 'blog')}.md",
        mime="text/markdown",
        use_container_width=False,
    )
elif not generate_clicked:
    st.info("Fill in the sidebar and click **Generate blog post** to start.")
