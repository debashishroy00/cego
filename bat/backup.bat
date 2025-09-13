@echo off
echo === CEGO Repository Save Tool ===
echo.

REM Navigate to where the .git folder actually is
cd /d "C:\Users\DR\OneDrive\projects\cego\cego"

REM Check if we're in a git repository
git status >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Not in a git repository!
    echo Current directory: %cd%
    echo Searching for .git folder...
    cd ..
    if exist .git (
        echo Found .git in parent directory
    ) else (
        cd cego
        if exist .git (
            echo Found .git in cego subfolder
        )
    )
    pause
    goto end
)

echo [1] Quick save (timestamp)
echo [2] Save with message
echo [3] Feature commit (feat:)
echo [4] Fix commit (fix:)
echo [5] Docs update (docs:)
echo [6] View status only
echo [7] Save without push
echo [8] Cancel
echo.

choice /c 12345678 /n /m "Select option: "

if %errorlevel%==1 goto quick
if %errorlevel%==2 goto message
if %errorlevel%==3 goto feature
if %errorlevel%==4 goto fix
if %errorlevel%==5 goto docs
if %errorlevel%==6 goto status
if %errorlevel%==7 goto local
if %errorlevel%==8 goto end

:quick
git add -A
git commit -m "chore: Update %date% %time%"
git push origin main
echo === Quick Save Complete ===
pause
goto end

:message
set /p msg="Enter commit message: "
git add -A
git commit -m "%msg%"
git push origin main
echo === Save Complete ===
pause
goto end

:feature
set /p feat="Enter feature description: "
git add -A
git commit -m "feat: %feat%"
git push origin main
echo === Feature Commit Complete ===
pause
goto end

:fix
set /p fixmsg="Enter fix description: "
git add -A
git commit -m "fix: %fixmsg%"
git push origin main
echo === Fix Commit Complete ===
pause
goto end

:docs
set /p docmsg="Enter docs update description: "
git add -A
git commit -m "docs: %docmsg%"
git push origin main
echo === Documentation Update Complete ===
pause
goto end

:status
echo === Current Status ===
git status
echo.
echo === Recent Commits ===
git log --oneline -5
pause
goto end

:local
git add -A
git commit -m "chore: Local save %date% %time%"
echo === Local Save Complete (not pushed) ===
echo To push later, run: git push origin main
pause
goto end

:end