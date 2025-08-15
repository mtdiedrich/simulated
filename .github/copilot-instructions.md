# Simulated Repository
A minimal Git repository with basic structure for development and experimentation.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively
- Bootstrap and explore the repository:
  - Clone: `git clone https://github.com/mtdiedrich/simulated.git`
  - Navigate: `cd simulated`
  - Check status: `git status` -- always shows current branch and uncommitted changes
  - List branches: `git branch -a` -- shows local and remote branches
  - View history: `git log --oneline -10` -- shows recent commits
- Basic file operations:
  - List files: `ls -la` -- shows all files including hidden ones
  - View file content: `cat README.md`
  - Edit files: Use `nano` or `vim` for text editing
- Git workflow:
  - Create branch: `git checkout -b feature/your-feature-name`
  - Stage changes: `git add .` or `git add specific-file.txt`
  - Check what's staged: `git status` -- shows staged vs unstaged changes
  - Commit changes: `git commit -m "Your commit message"`
  - Push changes: `git push origin branch-name`

## Validation
- Always run `git status` before and after making changes to understand repository state.
- ALWAYS test any new code or scripts you create by running them at least once.
- Always run `ls -la` to verify file structure after creating or modifying files.
- Test file operations by creating a temporary file, staging it, and then removing it:
  ```bash
  echo "test" > temp.txt
  git add temp.txt
  git status
  git reset HEAD temp.txt
  rm temp.txt
  git status
  ```

## Available Development Tools
The environment includes:
- Git version 2.50.1 (`git --version`)
- Node.js v20.19.4 (`node --version`)
- Python 3.12.3 (`python3 --version`)
- OpenJDK 17.0.16 (`java -version`)
- Text editors: nano, vim
- Standard Unix tools: ls, cat, find, grep, etc.

## Common Tasks
The following are validated commands and their expected outputs:

### Repository exploration
```bash
# View repository structure
ls -la
# Expected output:
# .git/        (Git metadata)
# .github/     (GitHub configuration)
# README.md    (Project documentation)

# Check git status
git status
# Expected output shows current branch and clean working tree when no changes

# View git history
git log --oneline -5
# Shows recent commits with short hashes and messages
```

### File operations
```bash
# Read the main documentation
cat README.md
# Contains: # simulated

# Create and test a new file
echo "Hello World" > test.txt
ls -la test.txt
cat test.txt
rm test.txt
```

### Development workflow validation
Always test these steps when making changes:
1. Create a new branch: `git checkout -b test-branch`
2. Make a small change: `echo "test content" > test-file.txt`
3. Stage the change: `git add test-file.txt`
4. Check status: `git status` -- should show staged file
5. Unstage if testing: `git reset HEAD test-file.txt`
6. Clean up: `rm test-file.txt`
7. Switch back: `git checkout main` or appropriate branch
8. Delete test branch: `git branch -d test-branch`

## Repository Expansion Guidelines
When adding new functionality to this repository:
- Always update this documentation with new build steps or requirements
- Add appropriate `.gitignore` entries for language-specific build artifacts
- Include validation steps for any new tools or dependencies
- Test all commands before documenting them
- Use imperative tone: "Run this command", "Do not do this"

## Known Limitations
- This is a minimal repository with only basic Git functionality
- No build system is currently configured
- No testing framework is currently set up
- No linting or formatting tools are configured

## Emergency Recovery
If you encounter repository state issues:
- Reset working directory: `git reset --hard HEAD`
- Clean untracked files: `git clean -fd` (use with caution)
- Check remote state: `git fetch origin && git status`
- Force sync with remote: `git reset --hard origin/main` (DANGER: loses local changes)