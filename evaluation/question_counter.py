import json
from app.config import BASE_DIR

json_dir = BASE_DIR / 'evaluation' / 'dataset.json'

with open (json_dir , 'r') as f:
    d = json.load(f)

scenariusze = d['scenarios']
ilosc_pytan = 0
for i in scenariusze:
    ilosc_pytan += len(i['turns'])


print(f'ilosc kategorii pytan: {len(scenariusze)}')
print(f'ilosc pytan: {ilosc_pytan}')