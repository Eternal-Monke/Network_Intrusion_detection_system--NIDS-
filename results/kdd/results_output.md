(cybenv) ➜  6841 ./main.py 
Train shape: (125973, 43)
Test shape: (22544, 43)
Training data head (first couple of entires):
   duration protocol_type   service flag  ...  dst_host_rerror_rate  dst_host_srv_rerror_rate    label  difficulty
0         0           tcp  ftp_data   SF  ...                  0.05                      0.00   normal          20
1         0           udp     other   SF  ...                  0.00                      0.00   normal          15
2         0           tcp   private   S0  ...                  0.00                      0.00  neptune          19
3         0           tcp      http   SF  ...                  0.00                      0.01   normal          21
4         0           tcp      http   SF  ...                  0.00                      0.00   normal          21

[5 rows x 43 columns]

Class distribution in training set:
label
normal             67343
neptune            41214
satan               3633
ipsweep             3599
portsweep           2931
smurf               2646
nmap                1493
back                 956
teardrop             892
warezclient          890
pod                  201
guess_passwd          53
buffer_overflow       30
warezmaster           20
land                  18
imap                  11
rootkit               10
loadmodule             9
ftp_write              8
multihop               7
phf                    4
perl                   3
spy                    2
Name: count, dtype: int64

Class distribution in test set:
label
normal             9711
neptune            4657
guess_passwd       1231
mscan               996
warezmaster         944
apache2             737
satan               735
processtable        685
smurf               665
back                359
snmpguess           331
saint               319
mailbomb            293
snmpgetattack       178
portsweep           157
ipsweep             141
httptunnel          133
nmap                 73
pod                  41
buffer_overflow      20
multihop             18
named                17
ps                   15
sendmail             14
rootkit              13
xterm                13
teardrop             12
xlock                 9
land                  7
xsnoop                4
ftp_write             3
worm                  2
loadmodule            2
perl                  2
sqlattack             2
udpstorm              2
phf                   2
imap                  1
Name: count, dtype: int64

After preprocessing:
   protocol_type   service  flag
0            0.5  0.289855   0.9
1            1.0  0.637681   0.9
2            0.5  0.710145   0.5
3            0.5  0.347826   0.9
4            0.5  0.347826   0.9

=== Classification Report ===
                 precision    recall  f1-score   support

        apache2       0.00      0.00      0.00       737
           back       0.93      0.96      0.94       359
buffer_overflow       0.00      0.00      0.00        20
      ftp_write       0.00      0.00      0.00         3
   guess_passwd       0.00      0.00      0.00      1231
     httptunnel       0.00      0.00      0.00       133
           imap       0.00      0.00      0.00         1
        ipsweep       0.58      0.98      0.73       141
           land       1.00      0.14      0.25         7
     loadmodule       0.00      0.00      0.00         2
       mailbomb       0.00      0.00      0.00       293
          mscan       0.00      0.00      0.00       996
       multihop       0.00      0.00      0.00        18
          named       0.00      0.00      0.00        17
        neptune       0.96      1.00      0.98      4657
           nmap       0.99      1.00      0.99        73
         normal       0.64      0.98      0.77      9711
           perl       0.00      0.00      0.00         2
            phf       1.00      0.50      0.67         2
            pod       0.72      0.93      0.81        41
      portsweep       0.75      0.97      0.84       157
   processtable       0.00      0.00      0.00       685
             ps       0.00      0.00      0.00        15
        rootkit       0.00      0.00      0.00        13
          saint       0.00      0.00      0.00       319
          satan       0.65      1.00      0.78       735
       sendmail       0.00      0.00      0.00        14
          smurf       0.99      1.00      1.00       665
  snmpgetattack       0.00      0.00      0.00       178
      snmpguess       0.00      0.00      0.00       331
      sqlattack       0.00      0.00      0.00         2
       teardrop       0.24      1.00      0.39        12
       udpstorm       0.00      0.00      0.00         2
    warezclient       0.00      0.00      0.00         0
    warezmaster       0.50      0.00      0.00       944
           worm       0.00      0.00      0.00         2
          xlock       0.00      0.00      0.00         9
         xsnoop       0.00      0.00      0.00         4
          xterm       0.00      0.00      0.00        13

       accuracy                           0.72     22544
      macro avg       0.25      0.27      0.23     22544
   weighted avg       0.57      0.72      0.62     22544


=== Confusion Matrix ===
/Users/og/6841/./main.py:178: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x=importances[indices][:10], y=features[indices][:10], palette='viridis')
