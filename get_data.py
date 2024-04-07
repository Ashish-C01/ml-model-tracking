import pandas as pd
url = 'https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/383116/rawdata_new.csv?sequence=1&isAllowed=y'
df = pd.read_csv(url)
df.to_csv('data_raw.csv', index=False)
