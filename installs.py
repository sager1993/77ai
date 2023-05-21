import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'chroma'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'langchain'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'cohere'])
