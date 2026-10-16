from pathlib import Path
import argparse,json
from IPython.core.interactiveshell import InteractiveShell
p=argparse.ArgumentParser();p.add_argument('--max-lesson',type=int,default=10);args=p.parse_args()
root=Path(__file__).resolve().parents[1];shell=InteractiveShell.instance()
for i,path in enumerate(sorted((root/'notebooks').glob('*.ipynb')),1):
    if i>args.max_lesson:break
    for cell in json.loads(path.read_text(encoding='utf-8'))['cells']:
        if cell['cell_type']=='code':
            source=''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']
            result=shell.run_cell(source)
            if result.error_before_exec or result.error_in_exec:raise RuntimeError(str(path))
