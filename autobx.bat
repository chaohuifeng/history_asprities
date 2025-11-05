@echo off
setlocal enabledelayedexpansion

:: 设置当前目录为根目录 / Set current directory as root directory
set "root_dir=%cd%"

:: 递归遍历所有子目录，并检查是否有更深层次的子目录 / Recursively traverse all subdirectories and check for deeper subdirectories
call :run_exsim_in_subdirs "%root_dir%"

echo All EXSIM instances have been started. / 所有EXSIM实例已启动。
pause
endlocal

:: 定义一个标签来处理递归逻辑 / Define a label to handle recursive logic
:run_exsim_in_subdirs
set "current_dir=%~1"

:: 对当前目录下的所有子文件夹进行操作 / Operate on all subfolders under current directory
for /d %%i in ("%current_dir%\*") do (
    :: 检查子目录中是否还有更深层次的子目录 / Check if there are deeper subdirectories in subdirectory
    set "has_subdirs=0"
    for /d %%j in ("%%i\*") do (
        set "has_subdirs=1"
        goto :continue_outer_loop
    )

    :continue_outer_loop
    :: 如果没有更深层次的子目录，则运行exsim_dmb.exe / If no deeper subdirectories, run exsim_dmb.exe
    if "!has_subdirs!"=="0" (
        echo Running EXSIM in directory: %%i / 在目录中运行EXSIM: %%i
        :: 切换到子目录 / Switch to subdirectory
        cd "%%i"
        
        :: 检查exsim_dmb.exe是否存在 / Check if exsim_dmb.exe exists
        if exist exsim_dmb.exe (
            echo Found exsim_dmb.exe in %%i. Running... / 在%%i中找到exsim_dmb.exe。正在运行...
            :: 运行exsim_dmb.exe并在新窗口自动输入参数文件名 / Run exsim_dmb.exe and automatically input parameter filename in new window
            start "" cmd /c "echo exsim_dmb.params | exsim_dmb.exe"
        ) else (
            echo exsim_dmb.exe not found in %%i. / 在%%i中未找到exsim_dmb.exe。
        )
        
        :: 返回上一层目录 / Return to parent directory
        cd "%current_dir%"
    ) else (
        :: 递归调用自身，处理有子目录的情况 / Recursively call itself to handle cases with subdirectories
        call :run_exsim_in_subdirs "%%i"
    )
)

goto :eof