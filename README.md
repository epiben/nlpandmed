# nlpandmed

[![DOI](https://zenodo.org/badge/xxx.svg)](https://zenodo.org/badge/latestdoi/xxx)

**Repo contributors**: [Benjamin Skov Kaas-Hansen](http://github.com/epiben) [code author], Davide Placido [code review], [Cristina Rodriguez Leal](http://github.com/crlero) [code review]

The repo holds the public version of our full analytic pipeline of the paper *Eliciting side effects from clinical notes: language-agnostic pharmacovigilant text mining*. Symlinks and other internal files have been removed as they are non-essential for reading the code in the repo and wouldn't port anyway.

### Scope of study
Develop a machine learning pipeline for safety signal detection in textual data from electronic medical records.

### Design
Case-control-like, but not really. 

### Data sources (all Danish)
- In-hospital medication data
- In-hospital clinical notes from the first 48 hours of admissions

# Important notes for running the pipeline:
- The command `bash snakeqsub.sh` *must* be run from the bash shell.
- If you happed to automatically start another shell in `.basrc`, deactivate that to avoid invoking this alternative shell. Otherwise, processes will linger and clog the pipeline.