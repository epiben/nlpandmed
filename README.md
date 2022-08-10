# nlpandmed

[![DOI](https://zenodo.org/badge/421156768.svg)](https://zenodo.org/badge/latestdoi/421156768)

**Repo contributors**: [Benjamin Skov Kaas-Hansen](http://github.com/epiben) [code author], [Davide Placido](http://github.com/daplaci) [code review], [Cristina Rodriguez Leal](http://github.com/crlero) [code review]

The repo holds the public version of our full analytic pipeline of the paper *Eliciting side effects from clinical notes: language-agnostic pharmacovigilant text mining*. Symlinks and other internal files have been removed as they are non-essential for reading the code in the repo and wouldn't port anyway.

### Related publications
Kaas-Hansen, BS, Placido, D, Rodríguez, CL, et al. Language-agnostic pharmacovigilant text mining to elicit side effects from clinical notes and hospital medication records. Basic Clin Pharmacol Toxicol. 2022; 1-12. doi:[10.1111/bcpt.13773](https://doi.org/10.1111/bcpt.13773)

### Scope of study
Develop a machine learning pipeline for safety signal detection in textual data from electronic medical records.

### Design
Case-control-like, but not really. 

### Data sources (all Danish)
- In-hospital medication data
- In-hospital clinical notes from the first 48 hours of admissions

### Important notes for running the pipeline:
- The command `bash snakeqsub.sh` *must* be run from the bash shell.
- If you happed to automatically start another shell in `.basrc`, deactivate that to avoid invoking this alternative shell. Otherwise, processes will linger and clog the pipeline.
