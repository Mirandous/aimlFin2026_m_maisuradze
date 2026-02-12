# Task 3: DDoS Detection Using Regression Analysis

**Author:** m_maisuradze_47631  
**Date:** February 13, 2026

---

## 📄 Main Report

**[ddos.md](./ddos.md)** - Complete comprehensive report (1,200+ lines, ~45KB)

This single document contains:
- **Attack Detection Results** - Time intervals and severity classification
- **Complete Analysis Report** - 9 main sections with methodology, results, and conclusions
- **Appendix A** - Quick start guide and file listing
- **Appendix B** - Complete program code documentation

---

## 🚨 Attack Summary

**DDoS Attack Detected:**

| Wave | Time | Duration | Peak Traffic | Z-Score | Severity |
|------|------|----------|--------------|---------|----------|
| #1 | 18:40-18:44 | 4 min | 12,292 req/min | 4.74 | **CRITICAL** |

**Pattern:** Single sustained high-intensity burst from distributed botnet causing complete server overload

---

## 📁 Repository Contents

### 📄 Documentation
- **ddos.md** (45KB) - Complete merged report with all documentation
- **README.md** - This file

### 💻 Python Programs (4 modules)
- **statistical_analysis.py** - Statistical data extraction and visualization
- **regression_analysis.py** - Polynomial regression modeling
- **attack_detection.py** - Attack detection and classification
- **enhanced_visualization.py** - Publication-quality 300 DPI dashboard

### 📊 Visualizations
- **enhanced_ddos_analysis.png** ⭐ - 8-panel dashboard (1.1MB, 300 DPI)
- **statistical_analysis.png** - 6-panel statistics (249KB)
- **regression_analysis.png** - 6-panel diagnostics (272KB)

### 📄 Data Files
- **m_maisuradze_47631_server.log** - Original log file (84,665 entries)
- **attack_detection_results.csv** - Detailed time-window analysis
- **regression_results.csv** - Regression predictions & residuals
- **attack_summary.txt** - Brief attack summary

---

## 🚀 Quick Start

### Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Run Complete Analysis
```bash
# Run all analyses in sequence
python statistical_analysis.py m_maisuradze_47631_server.log
python regression_analysis.py m_maisuradze_47631_server.log
python attack_detection.py m_maisuradze_47631_server.log
python enhanced_visualization.py m_maisuradze_47631_server.log
```

### Read Full Report
Open **[ddos.md](./ddos.md)** for complete documentation

---

## 📊 Key Results

- **Dataset:** 84,665 requests over 61 minutes
- **Baseline:** 1,388 requests/minute
- **Attack Peak:** 12,292 requests/minute (8.9x amplification)
- **Statistical Significance:** Z-score 4.74 (p < 0.0001)
- **Server Impact:** 57.3% error rate during peak attack
- **Severity:** CRITICAL (9/11 score)

---

## 📖 Documentation Structure

The **ddos.md** file is organized as follows:

1. **Attack Time Intervals** - Immediate results
2. **Executive Summary** - Key findings
3. **Main Report** (Sections 1-9):
   - Introduction
   - Methodology
   - Data Analysis  
   - Regression Analysis
   - Attack Detection Results
   - Visualizations
   - Code Implementation
   - Conclusions
   - References
4. **Appendix A** - Quick Start Guide
5. **Appendix B** - Program Documentation

---

## 🎯 For Reviewers

**Start here:** [ddos.md](./ddos.md) - Section: "🚨 DETECTED DDoS ATTACK TIME INTERVALS"

**For technical details:** See Section 2 (Methodology) and Section 4 (Regression Analysis)

**For code details:** See Appendix B (Program Code Documentation)

**For reproduction:** See Appendix A (Quick Start Guide)

---

## 📈 Analysis Highlights

### Methodology
- **Approach:** Polynomial regression (degree 2) with z-score anomaly detection
- **Threshold:** |z| > 2.0 (95% confidence)
- **Severity Scoring:** Multi-factor 0-11 point system

### Detection Results
- **Attack Start:** 2024-03-22 18:40:00 UTC+4
- **Attack End:** 2024-03-22 18:44:00 UTC+4
- **Duration:** 4 minutes continuous
- **Confidence:** Extremely high (p < 0.0001)

### Attack Characteristics
- **Type:** Volumetric DDoS
- **Source:** Distributed botnet (296 unique IPs)
- **Pattern:** Single sustained burst
- **Impact:** Complete server overload

---

**All documentation is consolidated into a single comprehensive ddos.md file**

**GitHub Repository:** Place all files from this directory into `task_3/` folder
