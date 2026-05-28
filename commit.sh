#!/bin/bash

echo "Adding all files >100MB to .gitignore..."

# Find all files > 100MB and add to .gitignore
find . -type f -size +100M -not -path "./.git/*" | sed 's|^\./||' >> .gitignore

# Sort and remove duplicates
sort -u -o .gitignore .gitignore

echo "Running git add ..."

git add .

echo "Committing with message '$1'..."
git commit -m "$1"

echo "Pushing changes to main..."
git push origin main