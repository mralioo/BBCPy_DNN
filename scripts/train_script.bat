@echo off

setlocal enabledelayedexpansion

set "venv_path=..\bbcpy_env"
call %venv_path%\Scripts\activate

cd ..\src

for /L %%i in (1,1,2) do (
    set "arg=logger.mlflow.run_name=run_%%i"
    python baseline_train.py !arg!
)

endlocal