"""Build envoy/static/meridian.html from the template plus the design assets in the original Meridian prototype
(CSS, land map data, globe renderer), so those are reused verbatim rather than retyped.

  venv/bin/python scripts/build_ui.py ["~/Downloads/Meridian – demo.html"]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "Downloads" / "Meridian – demo.html").read_text()
css = src[src.index("<style>") + len("<style>"):src.index("</style>")]
land = src[src.index("var LAND=") + len("var LAND="):src.index("\nvar NET=")].rstrip().rstrip(";")
globe = src[src.index("function globeStart(){"):src.index("\nvar STAGES=")]
tpl = (ROOT / "envoy" / "static" / "meridian_template.html").read_text()
out = tpl.replace("/*__CSS__*/", css).replace("/*__LAND__*/", land).replace("/*__GLOBE__*/", globe)
(ROOT / "envoy" / "static" / "meridian.html").write_text(out)
print(f"built envoy/static/meridian.html ({len(out) // 1024} KB)")
