#!/bin/bash

LOG_DIR="logs"

# 检查 logs 文件夹是否存在
if [ ! -d "$LOG_DIR" ]; then
    echo "❌ 找不到文件夹: '$LOG_DIR'。请确认你在项目根目录下运行此脚本。"
    exit 1
fi

# 按时间倒序查找最新的 .out 和 .err 文件
LATEST_OUT=$(ls -t $LOG_DIR/*.out 2>/dev/null | head -n 1)
LATEST_ERR=$(ls -t $LOG_DIR/*.err 2>/dev/null | head -n 1)

echo "=================================================="
if [ -n "$LATEST_OUT" ]; then
    echo "📄 最新的标准输出 (LOG): $LATEST_OUT"
    echo "--- 倒数 30 行 ---"
    tail -n 30 "$LATEST_OUT"
else
    echo "⚠️ 在 $LOG_DIR 中没有找到 .out 文件。"
fi
echo "=================================================="

echo ""

echo "=================================================="
if [ -n "$LATEST_ERR" ]; then
    echo "🚨 最新的错误输出 (ERR): $LATEST_ERR"
    echo "--- 倒数 30 行 ---"
    tail -n 30 "$LATEST_ERR"
else
    echo "✅ 在 $LOG_DIR 中没有找到 .err 文件 (或者还没有报错)。"
fi
echo "=================================================="