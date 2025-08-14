import nbformat
from nbconvert import PythonExporter

with open(r"C:\Plant_analysis\notebooks\Morphology\morphology_analysis.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

exporter = PythonExporter()
source, _ = exporter.from_notebook_node(nb)

with open("converted.py", "w", encoding="utf-8") as f:
    f.write(source)
