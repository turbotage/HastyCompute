
import sys
from pathlib import Path

def setup_test_environment():
	sys.path.append(str(Path(__file__).parent.parent.parent))

setup_test_environment()


