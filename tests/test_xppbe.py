import os
import tempfile
import pytest
import csv
import shutil
from xppbe import Simulation
from xppbe import Allrun,Allclean
import subprocess

import xppbe

def test_bash():
    print(xppbe.Molecules)
    os.system("pip list")
    print("\n\n\n")
    subprocess.Popen("pip list")