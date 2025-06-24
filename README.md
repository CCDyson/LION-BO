#LION-BO Git Help Sheet

This repository is set up so that we can both work on the code **independently** and then **merge our progress** without causing conflicts.

We will do this using **branches** — essentially different “versions” of the code that let us edit and test without touching the main copy (`main`) until we're ready.

---

## Getting Started

1. **Enter your work directory**

2. **Clone the repository:**

   ```bash
   git clone https://github.com/CCDyson/LION-BO.git #Copies the repository over from GitHub
   cd LION-BO #Enter the repository
   ```

---

## Day-to-Day Workflow

### 1. Make sure your local main branch is up to date

```bash
git checkout main
git pull origin main
```


### 2. Create your own branch from main

```bash
git checkout -b your-branch-name #Enters the branch
```

---

### 3. Make changes to the code

Edit, test, etc.
(When running the code I would run from your workspace, not the LION-BO folder.)

---

### 4. 📂 Save your work (commit your changes)

```bash
git add .      #Prepares the file for saving
git commit -m "Describe what you changed" #Saves the changes to your branch
```

You can do this often — it’s like saving snapshots of your progress, and very helpful for debugging

---

### 5. 🚀 Push your changes to GitHub

```bash
git push # Upload your branch’s changes to GitHub
```

This uploads your work to the shared repository, but *only* in your personal branch.

---

### 6. 🔄 Merge into `main` when your work is ready

When you've finished a task and everything works, merge your changes into `main`:

```bash
git checkout main               # Enter the main branch
git pull origin main            # Update your local main branch to match the online one
git merge your-branch-name      # Bring the changes you have made in your branch to the main           
git push origin main            # Update the online main with the new changes
```

Now the latest version of your work is available in the main branch. 🎉

---

## Example Scenario

> You’re editing the cost function in your branch.

1. `git checkout student`
2. Edit `cost_function.py`
3. `git add .`
4. `git commit -m "Improved cost function"`
5. `git push`
6. Once tested:
   ```bash
   git checkout main
   git pull origin main
   git merge your-branch-name
   git push origin main
   ```
7. I can now pull from `main` and use your updated code.

---

