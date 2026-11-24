import os
import sys
from pathlib import Path

def configure():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    os.environ.setdefault("KIVY_GL_BACKEND", "angle_sdl2")
    os.environ.setdefault("KIVY_NO_ARGS", "1")
    os.environ.setdefault("KIVY_LOG_MODE", "PYTHON")
    os.environ.setdefault("KIVY_HOME", str(root / ".kivy"))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

if __name__ == "__main__":
    configure()
    sys.argv[1:] = ["--desktop-demo", "--debug"]
    from main import main
    main()

