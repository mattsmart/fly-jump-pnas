### Legacy scripts not used in primary analysis

Notes: 
- scripts are provided for legacy purposes
- some scripts assume a different fit data format and no longer function (provided for reference only)
- some scripts should be moved to the parent directory (`/python`) in order to run without path errors
- alternatively, the repo root can be added to path by adding block below to the top

`````
# Adds fly-jump/python to sys.path and change working directory  
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  
sys.path.append(ROOT)
os.chdir(ROOT)  # Change to python/ directory for relative paths to work
`````
