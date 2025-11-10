# push.ps1 —— 一键提交并推送到 GitHub
# 位置：F:\1786kangding\history_asprities

$ErrorActionPreference = "Stop"
Set-Location -Path "F:\1786kangding\history_asprities"

Write-Host "`n[1/4] 检查 Git 仓库..."
if (-not (Test-Path ".git")) {
    git init
    git branch -M main
    git remote add origin git@github.com:chaohuifeng/history_asprities.git
    Write-Host "✅ 已初始化 Git 仓库并设置远程地址"
}

Write-Host "`n[2/4] 添加修改..."
git add .

$date = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$message = "update site - $date"

Write-Host "`n[3/4] 提交变更..."
try {
    git commit -m "$message"
} catch {
    Write-Host "⚠️ 没有新改动，跳过提交。"
}

Write-Host "`n[4/4] 推送到 GitHub..."
try {
    git pull --rebase --allow-unrelated-histories origin main 2>$null
} catch {}

git push -u origin main --force-with-lease

Write-Host "`n✅ 推送完成！"
Write-Host "👉 现在可以在浏览器打开：https://github.com/chaohuifeng/history_asprities"
