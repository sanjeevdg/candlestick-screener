# Clean_symbols.py

input_file = "datasets/symbols.csv"
output_file = "datasets/symbols_cleaned.csv"

remove_list = [
    'HFC', 'PXD', 'MYL', 'ABC', 'UTX', 'FLT', 'ADS', 'TIF', 'WCG', 'TWTR', 'BLL', 'XLNX', 'ABMD',
    'FISV', 'XEC', 'ALXN', 'DISH', 'ARNC', 'NBL', 'PEAK', 'JWN', 'RE', 'ANTM', 'SIVB', 'DISCK',
    'DISCA', 'RTN', 'MRO', 'WBA', 'GPS', 'CTXS', 'CTL', 'AGN', 'CERN', 'PBCT', 'DFS', 'HES',
    'FLIR', 'ETFC', 'WLTW', 'NLOK', 'JNPR', 'COG', 'FRC', 'BRK.B', 'ANSS', 'WRK', 'FBHS', 'NLSN',
    'DRE', 'VAR', 'CXO', 'MXIM', 'ATVI', 'VIAC', 'PKI', 'KSU'
]

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        symbol = line.strip().split(",")[0]
        if symbol not in remove_list:
            outfile.write(line)

print("✅ Cleaned list written to", output_file)
