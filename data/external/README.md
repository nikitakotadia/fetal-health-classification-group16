CTU-UHB Intrapartum Cardiotocography Database used for the external validation feasibility test (see notebook 05). Not included in this repository due to size. Download from: https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/
To reproduce: download records 1001–1003 (or any subset) into this folder before running 05_ctu_feasibility_test.ipynb.

# CTU-UHB External Validation — Feasibility Notes

Feasibility test conducted on 3 CTU-UHB intrapartum recordings using the wfdb Python library (05_ctu_feasibility_test.ipynb).

Signal-derived STV analogue (mean absolute bpm difference) was computed successfully (e.g. STV-analogue = 0.65 for Record 1001). However, values are not on the same scale as the UCI dataset's SisPorto-derived ASTV, which reports variability as a percentage-of-time metric.

Reconciling this scale mismatch is the primary remaining task before full external validation can proceed (planned: Interim Report phase).