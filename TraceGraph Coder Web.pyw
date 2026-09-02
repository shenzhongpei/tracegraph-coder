from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from tracegraph_coder.web_app import main


main()
