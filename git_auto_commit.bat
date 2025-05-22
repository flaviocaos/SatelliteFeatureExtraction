@echo off
echo ==============================
echo Commit Git Automático
echo ==============================
echo.

set /p commitMsg="Digite a mensagem do commit: "

git add .
git commit -m "%commitMsg%"
git push

echo.
echo ==============================
echo Commit enviado com sucesso!
echo ==============================
pause