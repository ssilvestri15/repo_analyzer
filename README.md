# Repository Analyzer 🔍

A powerful tool for analyzing Python code quality metrics commit by commit in Git repositories, with support for parallel analysis and automatic commit classification.

## 🌟 Key Features

- **Commit-by-Commit Analysis**: Extracts detailed metrics for each commit
- **Parallel Processing**: Supports simultaneous analysis of multiple repositories and commits
- **Advanced Metrics**: Cyclomatic complexity, code smells, warnings, bug density
- **Automatic Classification**: Categorizes commits (new feature, bug fix, enhancement, refactoring)
- **Resume Capability**: Resumes interrupted analysis using checkpoints
- **Comparative Reports**: Generates detailed reports for single repositories and multi-repo comparisons
- **Robust Error Handling**: Advanced error management with recovery strategies

## 📊 Extracted Metrics

### 🔢 Base Metrics

- **LOC**: Lines of code added/deleted
- **Modified files**: Number and list of changed files
- **Author and timestamp**: Author information and commit date

### 🧠 Complexity Metrics

- **Cyclomatic Complexity**: Per commit and per project (using Lizard)
- **Author Experience**: Number of previous commits by the author
- **Time Between Commits**: Time elapsed since the author's last commit

### 🐛 Quality Metrics

- **Code Smells**: Detection via CodeSmile
- **Warnings**: Using Pylint for static analysis
- **Smell Density**: Smells/LOC ratio
- **Bug Fix Detection**: Automatic identification of bug fix commits

### 🏷️ Commit Classification

- **New Feature**: New functionalities
- **Bug Fixing**: Error corrections
- **Enhancement**: Improvements and updates
- **Refactoring**: Code restructuring

## 🚀 Installation

### Prerequisites

```bash
# Python 3.6 or higher
python --version

# Git installed and accessible
git --version
```

### Dependency Installation

```bash
# Clone the Smell AI repository
git clone https://github.com/ssilvestri15/smell_ai.git

# Clone the repository
git clone https://github.com/ssilvestri15/repo-analyzer.git
cd repo-analyzer

python3.11 -m venv .venv
source ./venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Developer mode installation
pip install -e .
```

### Main Dependencies

- **GitPython**: Python interface for Git
- **Pandas**: Data manipulation and analysis
- **Lizard**: Cyclomatic complexity analysis
- **Pylint**: Static code analysis
- **CodeSmile**: Code smell detection (custom module)
- **psutil**: System resource monitoring

## 📖 Usage

### Single Repository Analysis

```bash
# Basic analysis
python -m repo_analyzer single https://github.com/user/repo.git

# With advanced options
python -m repo_analyzer single https://github.com/user/repo.git \
    --output-dir results/my-repo \
    --max-commits 100 \
    --resume
```

### Multi-Repository Analysis

```bash
# From URL list
python -m repo_analyzer multi \
    https://github.com/user/repo1.git \
    https://github.com/user/repo2.git \
    https://github.com/user/repo3.git \
    --output-dir multi-analysis \
    --max-commits 50

# From text file
echo "https://github.com/user/repo1.git" > repos.txt
echo "https://github.com/user/repo2.git" >> repos.txt

python -m repo_analyzer file repos.txt \
    --output-dir batch-analysis \
    --repo-workers 4 \
    --resume
```

### Common Options

| Option | Description |
|---------|-------------|
| `--output-dir` | Output directory for results |
| `--max-commits` | Maximum number of commits to analyze |
| `--resume` | Resume interrupted analysis |
| `--sequential` | Sequential mode (more stable) |
| `--max-workers` | Number of parallel workers |

## 📁 Output Structure

```
repo_analysis_results/
├── repository-name/
│   ├── commit_metrics.csv          # Detailed commit metrics
│   ├── commit_metrics.json         # Same content in JSON format
│   ├── file_frequencies.json       # File modification/bug frequencies
│   ├── report_summary.txt          # Summary report
│   ├── error_report.txt           # Error log
│   └── checkpoint.json            # State for resume
├── all_repos_metrics.csv          # Combined metrics (multi-repo)
├── comparative_report.txt         # Comparative report
└── commit_classification_summary.json  # Classification summary
```

## 📈 Metrics Example

### CSV Output (commit_metrics.csv)

```csv
commit_hash,author,date,message,LOC_added,LOC_deleted,files_changed,commit_cyclomatic_complexity,project_cyclomatic_complexity,author_experience,time_since_last_commit,smell_density,num_warnings,is_pr,is_bug_fix,new_feature,bug_fixing,enhancement,refactoring
abc123...,John Doe,2024-01-15T10:30:00,Add user authentication,45,12,3,8.5,156.2,15,2.5,0.0023,2,False,False,1,0,0,0
def456...,Jane Smith,2024-01-16T14:20:00,Fix login bug,8,15,2,3.2,158.9,8,3.1,0.0019,0,False,True,0,1,0,0
```

### Report Summary

```txt
Analysis report for: example-repo
Date: 2024-01-20 15:30:45
Total number of commits analyzed: 150

Average statistics:
- LOC added per commit: 23.45
- LOC deleted per commit: 18.32
- Files modified per commit: 2.8
- Average cyclomatic complexity of commits: 12.7
- Average cyclomatic complexity of project: 245.8
- Average author experience: 12.3 previous commits
- Average time since last commit: 4.2 hours
- Average smell density: 0.0031
- Average number of warnings: 1.2
- Average number of bug fixes: 18 (12.00%)

Number of Pull Requests: 25 (16.67%)
Number of distinct authors: 8
```

### Performance Optimization

```bash
# For very large repositories
python -m repo_analyzer single https://github.com/large/repo.git \
    --max-workers 2 \
    --max-commits 1000

# For optimized batch analysis
python -m repo_analyzer file repos.txt \
    --repo-workers 3 \
    --commit-workers 2 \
    --sequential
```

## 🔧 Troubleshooting

### Common Errors

**GitPython "read of closed file"**:

```bash
# Use sequential mode
python -m repo_analyzer single URL --sequential
```

**Corrupted repository or missing commits**:

```bash
# The analyzer automatically skips corrupted commits
# Check error_report.txt for details
```

**Insufficient memory**:

```bash
# Reduce parallel workers
python -m repo_analyzer single URL --max-workers 1
```

### Resume After Interruption

```bash
# Analysis automatically saves checkpoints
python -m repo_analyzer single URL --resume
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Simone Silvestri**

- Email: <s.silvestri15@studenti.unisa.it>
- University: University of Salerno

## 🙏 Acknowledgments

- [GitPython](https://github.com/gitpython-developers/GitPython) for Git interface
- [Lizard](https://github.com/terryyin/lizard) for complexity analysis
- [Pylint](https://github.com/PyCQA/pylint) for static analysis
- Open source community for inspiration and support

---

Made with ❤️ by **Simone Silvestri**
