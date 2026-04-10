import sys
import runpy
import os

file_path = os.path.abspath(sys.argv[1])
lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
sys.path.insert(0, lib_dir)

runpy.run_path(file_path, run_name='__main__')
