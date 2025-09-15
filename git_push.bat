@echo off
setlocal enabledelayedexpansion
title CEGO Git Push Script
color 0A

echo ========================================
echo        CEGO Repository Push Script
echo ========================================
echo.

REM Check if we're in a git repository
if not exist ".git" (
    echo [ERROR] Not a git repository!
    echo Please run this script from the repository root.
    pause
    exit /b 1
)

echo [INFO] Checking repository status...
git status --porcelain > temp_status.txt
set /a file_count=0
for /f %%i in (temp_status.txt) do set /a file_count+=1
del temp_status.txt

if %file_count%==0 (
    echo [INFO] No changes to commit.
    echo Repository is up to date.
    pause
    exit /b 0
)

echo.
echo [INFO] Changes detected. Showing current status:
echo.
git status --short

echo.
echo ========================================
echo            Commit Message
echo ========================================

REM Get commit message from user
set "commit_msg="
if "%~1"=="" (
    echo Please enter a commit message:
    set /p commit_msg="Commit message: "
    if "!commit_msg!"=="" (
        echo [ERROR] Commit message cannot be empty!
        pause
        exit /b 1
    )
) else (
    set "commit_msg=%~1"
)

echo.
echo [INFO] Commit message: !commit_msg!
echo.

REM Confirm before proceeding
choice /M "Proceed with commit and push?"
if errorlevel 2 (
    echo [INFO] Operation cancelled by user.
    pause
    exit /b 0
)

echo.
echo ========================================
echo           Git Operations
echo ========================================

echo [STEP 1/4] Adding all changes...
git add .
if errorlevel 1 (
    echo [ERROR] Failed to add files!
    pause
    exit /b 1
)
echo [OK] Files staged successfully.

echo.
echo [STEP 2/4] Committing changes...
git commit -m "!commit_msg!"
if errorlevel 1 (
    echo [ERROR] Commit failed!
    pause
    exit /b 1
)
echo [OK] Commit successful.

echo.
echo [STEP 3/4] Checking remote repository...
git remote -v
if errorlevel 1 (
    echo [WARNING] No remote repository configured.
    echo Would you like to add a remote? (y/n)
    choice /M "Add remote repository?"
    if errorlevel 2 (
        echo [INFO] Skipping push to remote.
        goto :local_complete
    )
    echo.
    set /p remote_url="Enter remote repository URL: "
    if "!remote_url!"=="" (
        echo [ERROR] Remote URL cannot be empty!
        pause
        exit /b 1
    )
    git remote add origin "!remote_url!"
    if errorlevel 1 (
        echo [ERROR] Failed to add remote repository!
        pause
        exit /b 1
    )
    echo [OK] Remote repository added.
)

echo.
echo [STEP 4/4] Pushing to remote repository...
git push origin main
if errorlevel 1 (
    echo [WARNING] Push to 'main' failed. Trying 'master'...
    git push origin master
    if errorlevel 1 (
        echo [ERROR] Push failed! Possible causes:
        echo - Network connectivity issues
        echo - Authentication required
        echo - Branch doesn't exist on remote
        echo - Remote repository is not accessible
        echo.
        echo Try manual push: git push origin [branch-name]
        pause
        exit /b 1
    )
)

echo [OK] Push successful!
goto :success

:local_complete
echo [OK] Local commit completed successfully.
echo [INFO] To push later, use: git push origin [branch-name]

:success
echo.
echo ========================================
echo             SUCCESS!
echo ========================================
echo Repository has been updated successfully.
echo.
echo Summary:
echo - Changes committed with message: "!commit_msg!"
if not "%remote_url%"=="" echo - Remote repository configured
echo - All operations completed successfully
echo.

REM Show final status
echo Current repository status:
git log --oneline -5
echo.

echo Press any key to exit...
pause > nul
endlocal