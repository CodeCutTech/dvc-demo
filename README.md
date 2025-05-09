[![View the article](https://img.shields.io/badge/CodeCut-View%20Article-blue)](https://codecut.ai/introduction-to-dvc-data-version-control-tool-for-machine-learning-projects-2/) 

# DVC Demo

A demonstration of Data Version Control (DVC) for managing ML pipelines and data versioning.

## What is DVC?

[DVC](https://dvc.org/) is an open-source version control system for machine learning projects. It helps you:
- Version control large files, data sets, machine learning models, and metrics
- Track ML experiments
- Create reproducible ML pipelines
- Collaborate with team members

## Project Structure

```
.
├── data/              # Raw and processed data files
│   └── raw.dvc        # DVC file for raw data
├── src/               # Source code for data processing and model training
├── config/            # Configuration files
├── .dvc/              # DVC internal files
├── dvc.yaml           # DVC pipeline definition
├── dvc.lock           # DVC lock file for reproducible pipelines
└── .dvcignore         # Files/directories to be ignored by DVC
```

## Setup

1. Install project dependencies using uv:

```bash
uv sync dvc
```

2. Pull the data from remote storage:

```bash
dvc pull
```

3. Run the pipeline to reproduce all stages:

```bash
dvc repro
```

## Version Control

- Track data files: `dvc add <file>`
- Push data to remote storage: `dvc push`
- Pull data from remote storage: `dvc pull`
- Check status: `dvc status`