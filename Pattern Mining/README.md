# Pattern Mining Experiment Using US RCC Database

This directory contains an experiment conducting [target sequential rule mining (TaSRM)](https://arxiv.org/abs/2206.04728) for the US RCC.

TaSRM is built on top of the [RuleGrowth SRM algorithm](https://dl.acm.org/doi/abs/10.1145/1982185.1982394). It mainly provides efficiency improvements in a SRM case where a query containing some antecedent tokens and/or some consequent tokens is provided.

Since the RCC is in CSV format, it must be mapped to a lexical, token-based format for operability with the [TaSRM source code](https://github.com/DSI-Lab1/TaSRM/tree/main). This is done in [lexical_reduction.ipynb](lexical_reduction.ipynb). The notebook should be functional with any other CSV-format data via small changes to the DataFrame loading and filtering steps. Likewise, the lexical format is also applicable to the popular [SPMF data mining library](https://www.philippe-fournier-viger.com/spmf/) which has implementations of a wide variety of data mining algorithms.

## Setup and Use

*Assuming setup steps in main directory readme have been completed and Java is properly installed (OpenJDK 17 used)*

1. Run [lexical_reduction.ipynb](lexical_reduction.ipynb) using desired CSV dataset stored elsewhere

**I do not re-distribute the source code for TaSRM. Only the runnable JAR and source code for Main.java are provided in this directory. To re-compile the JAR for yourself from the TaSRM source, follow these steps:**

1. Clone TaSRM from [source](https://github.com/DSI-Lab1/TaSRM/tree/main)
1. Copy all files in ```TaSRM/code``` into [runtasrm/src/main/java/com/runtasrm](runtasrm/src/main/java/com/runtasrm)
1. In this directory rename ```TaSRM.java``` to ```TaSRM_V3.java```
1. Ensure that ```package com.runtasrm;``` is at the top of each file in this directory. JAR will not build properly otherwise.
1. Ensure Maven is installed and up-to-date, your Java version matches that of [runtasrm/pom.xml](runtasrm/pom.xml), and Java is correctly configured in your PATH and JAVA_HOME environment variables
1. Build the JAR and follow the steps for use
