import pandas as pd

df = pd.read_csv('rmsd_all.csv')

thresholds = [10, 20, 30, 50, 100]
total = len(df)

print(f"总记录: {total}\n")
for t in thresholds:
    n = (df['rmsd'] > t).sum()
    print(f"RMSD > {t:>3} Å: {n:>8} 条  ({n/total*100:.1f}%)")
