You will be creating code to generate charts using python packages (matplotlib and Seaborn). These charts are going to be used ina slide deck for climate risk utility bond pricing. The charts should be minimalist and sleek, and make excellent use of the color scheme provided. I will direct you what charts to create. You should generate each chart in @charts.py that i can run and output the chart as a PNG. We should output with transparent=True if possible so they can be easily incorporated into the deck. Final_Report.pdf is the report we are creating the presentation for. Only load the final report if you need additional context.

## Colors
- Primary Color: #32006e
- Secondary Color: #4b2e83
- Tertiary Color: #e8e3d3
- Tertiary Color: #85754d
- Tertiary Color: #ffc700
- Accent Color: #aadb1e
- Accent Color: #2ad2c9
- Accent Color: #e93cac

## Data
- Data files are stored as CSVs in /data/
- Bond transaction Data: /data/msrb_transaction.csv.gz
- Bond reference data: /data/wa_ca_or_bond_reference_data.csv
- Utility area adjsuted gross income: /data/utility_service_area_average_adjusted_gross_income.csv
- Utility wildfire metrics: /data/utility_service_area_fire_metrics.csv.gz
- Do not read the data files directly, you should generate temporary scripts to load the data using pandas and output descriptions (df.decsribe(), df.columns, df.dtypes, etc...) to inform your chart making. If you are unsure of columns to use, ask me.