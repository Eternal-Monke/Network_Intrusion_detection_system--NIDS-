(cybenv) ➜  6841 ./unsw_main.py
Train shape: (175341, 45)
Test shape: (82327, 45)
Training data head (first couple of entires):
   id       dur proto service state  spkts  dpkts  ...  ct_ftp_cmd  ct_flw_http_mthd  ct_src_ltm  ct_srv_dst  is_sm_ips_ports  attack_cat  label
0   1  0.121478   tcp       -   FIN      6      4  ...           0                 0           1           1                0      Normal      0
1   2  0.649902   tcp       -   FIN     14     38  ...           0                 0           1           6                0      Normal      0
2   3  1.623129   tcp       -   FIN      8     16  ...           0                 0           2           6                0      Normal      0
3   4  1.681642   tcp     ftp   FIN     12     12  ...           1                 0           2           1                0      Normal      0
4   5  0.449454   tcp       -   FIN     10      6  ...           0                 0           2          39                0      Normal      0

[5 rows x 45 columns]

Class distribution in training set:
attack_cat
Normal            56000
Generic           40000
Exploits          33393
Fuzzers           18184
DoS               12264
Reconnaissance    10491
Analysis           2000
Backdoor           1746
Shellcode          1133
Worms               130
Name: count, dtype: int64

Class distribution in test set:
attack_cat
Normal            36998
Generic           18871
Exploits          11131
Fuzzers            6061
DoS                4089
Reconnaissance     3495
Analysis            677
Backdoor            583
Shellcode           378
Worms                44
Name: count, dtype: int64

After preprocessing:
      proto  service  state
0  0.856061     0.00   0.25
1  0.856061     0.00   0.25
2  0.856061     0.00   0.25
3  0.856061     0.25   0.25
4  0.856061     0.00   0.25

=== Classification Report ===
                precision    recall  f1-score   support

      Analysis       0.00      0.00      0.00       677
      Backdoor       0.02      0.11      0.03       583
           DoS       0.59      0.12      0.20      4089
      Exploits       0.63      0.77      0.69     11131
       Fuzzers       0.29      0.59      0.39      6061
       Generic       1.00      0.97      0.98     18871
        Normal       0.97      0.76      0.85     36998
Reconnaissance       0.93      0.80      0.86      3495
     Shellcode       0.33      0.69      0.44       378
         Worms       0.54      0.16      0.25        44

      accuracy                           0.75     82327
     macro avg       0.53      0.50      0.47     82327
  weighted avg       0.84      0.75      0.78     82327


=== Confusion Matrix ===
/Users/og/6841/./unsw_main.py:184: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x=importances[indices][:10], y=features[indices][:10], palette='viridis')
